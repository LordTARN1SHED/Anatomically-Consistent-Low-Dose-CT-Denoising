from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, sobel
from torch.utils.data import DataLoader
from tqdm import tqdm

from egldm.config import ProjectConfig
from egldm.data import build_dataloaders
from egldm.eval.metrics import FIDFeatureExtractor, compute_fid_from_features, psnr, rmse, ssim
from egldm.models.autoencoder_utils import decode_from_latent, encode_to_latent
from egldm.models.conditioning import latent_to_condition_tokens
from egldm.models.factory import build_models
from egldm.models.redcnn import REDCNN
from egldm.utils import ensure_dir, get_device, save_json, seed_everything


class EGLDMEvaluator:
    def __init__(self, cfg: ProjectConfig, checkpoint_path: str | Path, output_dir: str | Path) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = get_device()

        self.bundle = build_models(cfg.model, cfg.train)
        self.bundle.vae.to(self.device).eval()
        self.bundle.unet.to(self.device).eval()
        self.bundle.controlnet.to(self.device).eval()
        if self.bundle.condition_projector is not None:
            self.bundle.condition_projector.to(self.device).eval()
        self.anchor_model = self._load_anchor_model()

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.bundle.controlnet.load_state_dict(ckpt["controlnet"])
        if "unet" in ckpt:
            self.bundle.unet.load_state_dict(ckpt["unet"], strict=False)
        if "vae" in ckpt:
            self.bundle.vae.load_state_dict(ckpt["vae"], strict=False)
        if self.bundle.condition_projector is not None and "condition_projector" in ckpt:
            self.bundle.condition_projector.load_state_dict(ckpt["condition_projector"])

        self.out_dir = ensure_dir(output_dir)
        self.pred_dir = ensure_dir(self.out_dir / "preds")
        self.gt_dir = ensure_dir(self.out_dir / "gts")
        self.noisy_dir = ensure_dir(self.out_dir / "noisy")
        self.anchor_dir = ensure_dir(self.out_dir / "anchors")

    def _load_anchor_model(self) -> torch.nn.Module | None:
        if self.cfg.train.anchor_model_type.lower() != "redcnn":
            return None
        if not self.cfg.train.anchor_checkpoint_path:
            raise ValueError("train.anchor_checkpoint_path is required when train.anchor_model_type='redcnn'.")

        payload = torch.load(self.cfg.train.anchor_checkpoint_path, map_location="cpu")
        model_cfg = payload.get("config", {}).get("model", {})
        model = REDCNN(
            base_channels=int(model_cfg.get("base_channels", 96)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
        )
        model.load_state_dict(payload["model"])
        model.to(self.device).eval()
        return model

    def _cross_attention_dim(self) -> int:
        cross_dim = self.bundle.unet.config.cross_attention_dim
        if isinstance(cross_dim, int):
            return cross_dim
        return int(cross_dim[0])

    def _encode_condition_tokens(
        self,
        c_ldct: torch.Tensor,
        context_noisy_ct: torch.Tensor | None,
        anchor_ct: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz = c_ldct.shape[0]
        cross_dim = self._cross_attention_dim()
        downsample_factor = max(1, int(self.cfg.model.condition_token_downsample))
        if self.cfg.model.enable_ldct_condition:
            tokens = [
                latent_to_condition_tokens(
                    c_ldct,
                    cross_attention_dim=cross_dim,
                    projector=self.bundle.condition_projector,
                    downsample_factor=downsample_factor,
                )
            ]
        else:
            tokens = [torch.zeros((bsz, 1, cross_dim), device=self.device, dtype=c_ldct.dtype)]

        if context_noisy_ct is not None and context_noisy_ct.numel() > 0:
            bsz, n_ctx, h, w = context_noisy_ct.shape
            ctx = context_noisy_ct.reshape(bsz * n_ctx, 1, h, w)
            ctx_latent = encode_to_latent(self.bundle.vae, ctx, self.cfg.model.latent_scaling_factor)
            ctx_tokens = latent_to_condition_tokens(
                ctx_latent,
                cross_attention_dim=cross_dim,
                projector=self.bundle.condition_projector,
                downsample_factor=downsample_factor,
            )
            tokens.append(ctx_tokens.reshape(bsz, -1, ctx_tokens.shape[-1]))

        if anchor_ct is not None:
            anchor_latent = encode_to_latent(self.bundle.vae, anchor_ct, self.cfg.model.latent_scaling_factor)
            tokens.append(
                latent_to_condition_tokens(
                    anchor_latent,
                    cross_attention_dim=cross_dim,
                    projector=self.bundle.condition_projector,
                    downsample_factor=downsample_factor,
                )
            )
        return torch.cat(tokens, dim=1)

    @torch.no_grad()
    def _sample_latent(
        self,
        c_ldct: torch.Tensor,
        edge_map: torch.Tensor,
        context_noisy_ct: torch.Tensor | None,
        anchor_ct: torch.Tensor | None,
        num_steps: int = 50,
    ) -> torch.Tensor:
        encoder_hidden_states = self._encode_condition_tokens(c_ldct, context_noisy_ct, anchor_ct)
        bsz = c_ldct.shape[0]

        mode = self.cfg.eval.inference_mode
        cond_scale = float(self.cfg.eval.controlnet_conditioning_scale)

        if mode == "direct_x0":
            t_val = int(np.clip(self.cfg.eval.direct_timestep, 1, self.cfg.train.num_train_timesteps - 1))
            t = torch.full((bsz,), t_val, device=self.device, dtype=torch.long)

            down_res, mid_res = self.bundle.controlnet(
                c_ldct,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=edge_map,
                conditioning_scale=cond_scale,
                return_dict=False,
            )
            noise_pred = self.bundle.unet(
                c_ldct,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
                return_dict=False,
            )[0]

            alpha_bar = self.bundle.noise_scheduler.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
            z0_est = (c_ldct - torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar.clamp(min=1e-6))
            return z0_est

        self.bundle.noise_scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.bundle.noise_scheduler.timesteps

        if mode == "from_noise":
            z = torch.randn_like(c_ldct)
            active_timesteps = timesteps
        elif mode == "img2img":
            strength = float(np.clip(self.cfg.eval.strength, 0.0, 1.0))
            init_timestep = min(int(num_steps * strength), num_steps)
            if init_timestep < 1:
                return c_ldct

            t_start = max(num_steps - init_timestep, 0)
            active_timesteps = timesteps[t_start:]
            noise = torch.randn_like(c_ldct)
            t0 = torch.full((bsz,), int(active_timesteps[0].item()), device=self.device, dtype=torch.long)
            z = self.bundle.noise_scheduler.add_noise(c_ldct, noise, t0)
        else:
            raise ValueError(f"Unsupported inference_mode: {mode}")

        for t in active_timesteps:
            down_res, mid_res = self.bundle.controlnet(
                z,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=edge_map,
                conditioning_scale=cond_scale,
                return_dict=False,
            )
            noise_pred = self.bundle.unet(
                z,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
                return_dict=False,
            )[0]
            z = self.bundle.noise_scheduler.step(noise_pred, t, z).prev_sample

        return z

    @staticmethod
    def _to_numpy_2d(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().squeeze()

    @staticmethod
    def _save_png(array_2d: np.ndarray, path: Path, cmap: str = "gray") -> None:
        plt.figure(figsize=(4, 4))
        plt.imshow(array_2d, cmap=cmap, vmin=-1.0, vmax=1.0)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _edge_strength_map(self, noisy_2d: np.ndarray) -> np.ndarray:
        sigma = max(float(self.cfg.eval.edge_adaptive_blur_sigma), 0.0)
        base = gaussian_filter(noisy_2d, sigma=sigma) if sigma > 0 else noisy_2d
        grad_x = sobel(base, axis=1, mode="nearest")
        grad_y = sobel(base, axis=0, mode="nearest")
        grad = np.sqrt(grad_x**2 + grad_y**2)
        denom = float(np.max(grad))
        if denom <= 1e-6:
            return np.zeros_like(grad)
        return grad / denom

    def _postprocess_prediction(self, pred_2d: np.ndarray, noisy_2d: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.cfg.eval.output_blend_alpha, 0.0, 1.0))
        edge_strength = max(float(self.cfg.eval.edge_adaptive_blend_strength), 0.0)

        if alpha >= 1.0 - 1e-8 and edge_strength <= 1e-8:
            return np.clip(pred_2d, -1.0, 1.0)

        if edge_strength <= 1e-8:
            weight = alpha
        else:
            edge_map = self._edge_strength_map(noisy_2d)
            weight = np.clip(alpha - edge_strength * edge_map, 0.0, 1.0)

        fused = weight * pred_2d + (1.0 - weight) * noisy_2d
        return np.clip(fused, -1.0, 1.0)

    def _blend_with_anchor(self, pred_2d: np.ndarray, anchor_2d: np.ndarray | None) -> np.ndarray:
        if anchor_2d is None:
            return pred_2d
        alpha = float(np.clip(self.cfg.eval.anchor_blend_alpha, 0.0, 1.0))
        if alpha <= 1e-8:
            return np.clip(anchor_2d, -1.0, 1.0)
        return np.clip(alpha * pred_2d + (1.0 - alpha) * anchor_2d, -1.0, 1.0)

    def evaluate(self, num_samples: int, save_predictions: bool = True, compute_fid: bool = True) -> dict[str, float]:
        _, val_loader = build_dataloaders(self.cfg.data, seed=self.cfg.seed, eval_split=self.cfg.eval.dataset_split)

        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        rmse_vals: list[float] = []
        noisy_psnr_vals: list[float] = []
        noisy_ssim_vals: list[float] = []
        noisy_rmse_vals: list[float] = []

        fid_extractor = FIDFeatureExtractor(self.device) if compute_fid else None
        real_feats: list[np.ndarray] = []
        fake_feats: list[np.ndarray] = []
        manifest_entries: list[dict[str, Any]] = []

        seen = 0
        for batch in tqdm(val_loader, desc="Evaluate"):
            clean = batch["clean_ct"].to(self.device)
            noisy = batch["noisy_ct"].to(self.device)
            edge = batch["edge_map"].to(self.device)
            context_noisy = batch["context_noisy_ct"].to(self.device)
            if self.cfg.train.disable_edge_condition:
                edge = torch.zeros_like(edge)

            with torch.no_grad():
                c_ldct = encode_to_latent(self.bundle.vae, noisy, self.cfg.model.latent_scaling_factor)
                anchor = self.anchor_model(noisy).clamp(-1.0, 1.0) if self.anchor_model is not None else None
                pred_latent = self._sample_latent(
                    c_ldct,
                    edge,
                    context_noisy,
                    anchor,
                    num_steps=self.cfg.eval.num_inference_steps,
                )
                pred = decode_from_latent(
                    self.bundle.vae,
                    pred_latent,
                    self.cfg.model.latent_scaling_factor,
                    out_channels=clean.shape[1],
                )

            for i in range(clean.shape[0]):
                c_np = self._to_numpy_2d(clean[i])
                p_np = self._to_numpy_2d(pred[i])
                n_np = self._to_numpy_2d(noisy[i])
                a_np = self._to_numpy_2d(anchor[i]) if anchor is not None else None
                p_np = self._blend_with_anchor(p_np, a_np)
                p_np = self._postprocess_prediction(p_np, n_np)

                psnr_vals.append(psnr(p_np, c_np))
                ssim_vals.append(ssim(p_np, c_np))
                rmse_vals.append(rmse(p_np, c_np))
                noisy_psnr_vals.append(psnr(n_np, c_np))
                noisy_ssim_vals.append(ssim(n_np, c_np))
                noisy_rmse_vals.append(rmse(n_np, c_np))

                if save_predictions:
                    stem = f"sample_{seen:06d}"
                    np.save(self.pred_dir / f"{stem}.npy", p_np)
                    np.save(self.gt_dir / f"{stem}.npy", c_np)
                    np.save(self.noisy_dir / f"{stem}.npy", n_np)
                    if a_np is not None:
                        np.save(self.anchor_dir / f"{stem}.npy", a_np)
                    self._save_png(p_np, self.pred_dir / f"{stem}.png")
                    self._save_png(c_np, self.gt_dir / f"{stem}.png")
                    self._save_png(n_np, self.noisy_dir / f"{stem}.png")
                    if a_np is not None:
                        self._save_png(a_np, self.anchor_dir / f"{stem}.png")

                if fid_extractor is not None:
                    real_feats.append(fid_extractor.extract(clean[i : i + 1]))
                    fake_tensor = torch.from_numpy(p_np).unsqueeze(0).unsqueeze(0).to(self.device, dtype=clean.dtype)
                    fake_feats.append(fid_extractor.extract(fake_tensor))

                manifest_entries.append(
                    {
                        "sample_index": seen,
                        "source_path": str(batch["path"][i]),
                        "dataset_split": self.cfg.eval.dataset_split,
                    }
                )

                seen += 1
                if seen >= num_samples:
                    break

            if seen >= num_samples:
                break

        result = {
            "PSNR": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
            "SSIM": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
            "RMSE": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
            "LDCT_PSNR": float(np.mean(noisy_psnr_vals)) if noisy_psnr_vals else float("nan"),
            "LDCT_SSIM": float(np.mean(noisy_ssim_vals)) if noisy_ssim_vals else float("nan"),
            "LDCT_RMSE": float(np.mean(noisy_rmse_vals)) if noisy_rmse_vals else float("nan"),
        }
        if noisy_psnr_vals:
            result["Delta_PSNR"] = result["PSNR"] - result["LDCT_PSNR"]
            result["Delta_SSIM"] = result["SSIM"] - result["LDCT_SSIM"]
            result["Delta_RMSE"] = result["RMSE"] - result["LDCT_RMSE"]

        if fid_extractor is not None and real_feats and fake_feats:
            real = np.concatenate(real_feats, axis=0)
            fake = np.concatenate(fake_feats, axis=0)
            result["FID"] = compute_fid_from_features(real, fake)

        save_json({"samples": manifest_entries}, self.out_dir / "manifest.json")
        save_json({"metrics": result, "config": asdict(self.cfg)}, self.out_dir / "metrics.json")
        return result
