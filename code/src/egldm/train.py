from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from egldm.config import ProjectConfig
from egldm.data import build_dataloaders
from egldm.data.latent_cache import LatentCacheDataset
from egldm.models.autoencoder_utils import decode_from_latent, encode_to_latent
from egldm.models.conditioning import latent_to_condition_tokens
from egldm.models.factory import ModelBundle, build_models
from egldm.models.redcnn import REDCNN
from egldm.utils import ensure_dir, get_device, save_json, seed_everything


class ControlNetTrainer:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)

        self.device = get_device()
        self.amp_enabled = cfg.train.mixed_precision and self.device.type == "cuda"
        self.amp_dtype = torch.float32
        scaler_enabled = False
        if self.amp_enabled:
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
                scaler_enabled = True
        self.dtype = self.amp_dtype if self.amp_enabled else torch.float32

        self.bundle: ModelBundle = build_models(cfg.model, cfg.train)
        self.bundle.vae.to(self.device)
        self.bundle.unet.to(self.device)
        self.bundle.controlnet.to(self.device)
        if self.bundle.condition_projector is not None:
            self.bundle.condition_projector.to(self.device)
        self.anchor_model = self._load_anchor_model()

        params: list[nn.Parameter] = [p for p in self.bundle.controlnet.parameters() if p.requires_grad]
        params.extend([p for p in self.bundle.unet.parameters() if p.requires_grad])
        params.extend([p for p in self.bundle.vae.parameters() if p.requires_grad])
        if self.bundle.condition_projector is not None:
            params.extend([p for p in self.bundle.condition_projector.parameters() if p.requires_grad])
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found. Check freeze settings.")
        self.trainable_params = params

        self.optimizer = AdamW(self.trainable_params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

        self.out_dir = ensure_dir(cfg.output_dir)
        self.ckpt_dir = ensure_dir(self.out_dir / "checkpoints")
        self.log_dir = ensure_dir(self.out_dir / "logs")
        self.history_path = self.log_dir / "loss_history.json"
        self.global_step = 0
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.history: list[dict[str, float | int]] = []

        save_json(
            {
                "config": asdict(cfg),
                "device": str(self.device),
                "dtype": str(self.dtype),
                "mixed_precision_enabled": self.amp_enabled,
                "grad_scaler_enabled": self.scaler.is_enabled(),
            },
            self.log_dir / "run_meta.json",
        )

    def _load_anchor_model(self) -> nn.Module | None:
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
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _build_loaders(self) -> tuple[DataLoader, DataLoader, bool]:
        cache_root = Path(self.cfg.train.latent_cache_dir)
        needs_full_image_batches = (
            self.cfg.data.neighboring_slices > 0
            or self.cfg.train.aux_l1_weight > 0
            or self.cfg.train.aux_gradient_weight > 0
            or self.anchor_model is not None
        )
        use_cache = (
            self.cfg.train.cache_latents
            and not needs_full_image_batches
            and (cache_root / "train_index.json").exists()
            and (cache_root / "val_index.json").exists()
        )

        if use_cache:
            train_ds = LatentCacheDataset(cache_root, split="train")
            val_ds = LatentCacheDataset(cache_root, split="val")
            train_loader = DataLoader(
                train_ds,
                batch_size=self.cfg.data.batch_size,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.cfg.data.batch_size,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                drop_last=False,
            )
            return train_loader, val_loader, True

        tr, va = build_dataloaders(self.cfg.data, seed=self.cfg.seed)
        return tr, va, False

    def _cross_attention_dim(self) -> int:
        cross_dim = self.bundle.unet.config.cross_attention_dim
        if isinstance(cross_dim, int):
            return cross_dim
        if isinstance(cross_dim, (tuple, list)):
            return int(cross_dim[0])
        raise ValueError(f"Unsupported cross_attention_dim type: {type(cross_dim)}")

    def _prepare_latents(
        self,
        batch: dict[str, Any],
        cached: bool,
    ) -> dict[str, torch.Tensor]:
        if cached:
            return {
                "z0": batch["z0"].to(self.device),
                "c_ldct": batch["c_ldct"].to(self.device),
                "edge_map": batch["edge_map"].to(self.device),
                "context_noisy_ct": batch["context_noisy_ct"].to(self.device),
            }

        clean = batch["clean_ct"].to(self.device)
        noisy = batch["noisy_ct"].to(self.device)
        edge_map = batch["edge_map"].to(self.device)
        context_noisy = batch["context_noisy_ct"].to(self.device)

        with torch.no_grad():
            z0 = encode_to_latent(self.bundle.vae, clean, self.cfg.model.latent_scaling_factor)
            c_ldct = encode_to_latent(self.bundle.vae, noisy, self.cfg.model.latent_scaling_factor)

        return {
            "z0": z0,
            "c_ldct": c_ldct,
            "edge_map": edge_map,
            "clean_ct": clean,
            "noisy_ct": noisy,
            "context_noisy_ct": context_noisy,
        }

    def _encode_condition_tokens(
        self,
        c_ldct: torch.Tensor,
        context_noisy_ct: torch.Tensor | None,
        anchor_ct: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz = c_ldct.shape[0]
        cross_dim = self._cross_attention_dim()
        downsample_factor = max(1, int(self.cfg.model.condition_token_downsample))
        tokens: list[torch.Tensor] = []

        if self.cfg.model.enable_ldct_condition:
            tokens.append(
                latent_to_condition_tokens(
                    c_ldct,
                    cross_attention_dim=cross_dim,
                    projector=self.bundle.condition_projector,
                    downsample_factor=downsample_factor,
                )
            )
        else:
            tokens.append(torch.zeros((bsz, 1, cross_dim), device=self.device, dtype=c_ldct.dtype))

        if context_noisy_ct is not None and context_noisy_ct.numel() > 0:
            bsz, n_ctx, h, w = context_noisy_ct.shape
            ctx = context_noisy_ct.reshape(bsz * n_ctx, 1, h, w)
            with torch.no_grad():
                ctx_latent = encode_to_latent(self.bundle.vae, ctx, self.cfg.model.latent_scaling_factor)
            ctx_tokens = latent_to_condition_tokens(
                ctx_latent,
                cross_attention_dim=cross_dim,
                projector=self.bundle.condition_projector,
                downsample_factor=downsample_factor,
            )
            ctx_tokens = ctx_tokens.reshape(bsz, -1, ctx_tokens.shape[-1])
            tokens.append(ctx_tokens)

        if anchor_ct is not None:
            with torch.no_grad():
                anchor_latent = encode_to_latent(self.bundle.vae, anchor_ct, self.cfg.model.latent_scaling_factor)
            anchor_tokens = latent_to_condition_tokens(
                anchor_latent,
                cross_attention_dim=cross_dim,
                projector=self.bundle.condition_projector,
                downsample_factor=downsample_factor,
            )
            tokens.append(anchor_tokens)

        return torch.cat(tokens, dim=1)

    def _predict_anchor(self, noisy_ct: torch.Tensor) -> torch.Tensor | None:
        if self.anchor_model is None:
            return None
        anchor = self.anchor_model(noisy_ct).clamp(-1.0, 1.0)
        if self.cfg.train.anchor_condition_dropout > 0:
            keep_prob = 1.0 - float(self.cfg.train.anchor_condition_dropout)
            if keep_prob < 1.0:
                mask = torch.rand((anchor.shape[0], 1, 1, 1), device=anchor.device) < keep_prob
                anchor = torch.where(mask, anchor, noisy_ct)
        return anchor

    @staticmethod
    def _gradient_map(x: torch.Tensor) -> torch.Tensor:
        grad_x = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        return torch.cat([grad_x, grad_y], dim=1)

    def _forward_loss(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z0 = features["z0"]
        c_ldct = features["c_ldct"]
        edge_map = features["edge_map"]
        clean_ct = features.get("clean_ct")
        noisy_ct = features.get("noisy_ct")
        context_noisy_ct = features.get("context_noisy_ct")
        bsz = z0.shape[0]
        if self.cfg.train.disable_edge_condition:
            edge_map = torch.zeros_like(edge_map)

        anchor_ct = self._predict_anchor(noisy_ct) if noisy_ct is not None else None
        t = torch.randint(
            low=0,
            high=self.cfg.train.num_train_timesteps,
            size=(bsz,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(z0)
        zt = self.bundle.noise_scheduler.add_noise(z0, noise, t)

        encoder_hidden_states = self._encode_condition_tokens(c_ldct, context_noisy_ct, anchor_ct)

        down_res, mid_res = self.bundle.controlnet(
            zt,
            t,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=edge_map,
            conditioning_scale=1.0,
            return_dict=False,
        )

        noise_pred = self.bundle.unet(
            zt,
            t,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            return_dict=False,
        )[0]

        loss_noise = F.mse_loss(noise_pred, noise)
        total_loss = loss_noise
        losses: dict[str, torch.Tensor] = {"loss": loss_noise, "loss_noise": loss_noise}

        if clean_ct is not None and (self.cfg.train.aux_l1_weight > 0 or self.cfg.train.aux_gradient_weight > 0):
            alpha_bar = self.bundle.noise_scheduler.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
            z0_est = (zt - torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar.clamp(min=1e-6))
            recon = decode_from_latent(
                self.bundle.vae,
                z0_est,
                self.cfg.model.latent_scaling_factor,
                out_channels=clean_ct.shape[1],
            ).clamp(-1.0, 1.0)

            if self.cfg.train.aux_l1_weight > 0:
                loss_l1 = F.l1_loss(recon, clean_ct)
                total_loss = total_loss + float(self.cfg.train.aux_l1_weight) * loss_l1
                losses["loss_l1"] = loss_l1

            if self.cfg.train.aux_gradient_weight > 0:
                loss_grad = F.l1_loss(self._gradient_map(recon), self._gradient_map(clean_ct))
                total_loss = total_loss + float(self.cfg.train.aux_gradient_weight) * loss_grad
                losses["loss_gradient"] = loss_grad

        losses["loss"] = total_loss
        if anchor_ct is not None:
            losses["anchor_mean_abs"] = anchor_ct.abs().mean()
        return losses

    def _persist_history(self) -> None:
        save_json({"history": self.history}, self.history_path)

    def _save_checkpoint(self, name: str, step: int, epoch: int, epoch_completed: bool = False) -> None:
        payload: dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "epoch_completed": epoch_completed,
            "controlnet": self.bundle.controlnet.state_dict(),
            "unet": self.bundle.unet.state_dict(),
            "vae": self.bundle.vae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": asdict(self.cfg),
        }
        if self.bundle.condition_projector is not None:
            payload["condition_projector"] = self.bundle.condition_projector.state_dict()
        if self.scaler.is_enabled():
            payload["scaler"] = self.scaler.state_dict()
        torch.save(payload, self.ckpt_dir / name)

    def resume_from_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu")

        self.bundle.controlnet.load_state_dict(payload["controlnet"])
        self.bundle.unet.load_state_dict(payload["unet"])
        self.bundle.vae.load_state_dict(payload["vae"])
        if self.bundle.condition_projector is not None and "condition_projector" in payload:
            self.bundle.condition_projector.load_state_dict(payload["condition_projector"])

        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scaler.is_enabled() and "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])

        self.global_step = int(payload.get("step", 0))
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        self.history = list(payload.get("history", []))

        epoch = int(payload.get("epoch", 0))
        epoch_completed = bool(payload.get("epoch_completed", False))
        self.start_epoch = epoch + 1 if epoch_completed else max(1, epoch)
        if self.start_epoch < 1:
            self.start_epoch = 1

        self._persist_history()

    def _validate(self, val_loader: DataLoader, cached: bool) -> float:
        self.bundle.controlnet.eval()
        self.bundle.unet.eval()
        self.bundle.vae.eval()
        if self.bundle.condition_projector is not None:
            self.bundle.condition_projector.eval()

        losses: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                features = self._prepare_latents(batch, cached=cached)
                loss_dict = self._forward_loss(features)
                losses.append(float(loss_dict["loss"].item()))

        self.bundle.controlnet.train()
        self.bundle.unet.train()
        self.bundle.vae.train()
        if self.bundle.condition_projector is not None:
            self.bundle.condition_projector.train()

        return float(sum(losses) / max(1, len(losses)))

    def run(self) -> None:
        train_loader, val_loader, cached = self._build_loaders()

        grad_accum = max(1, int(self.cfg.train.gradient_accumulation_steps))

        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            self.bundle.controlnet.train()
            self.bundle.unet.train()
            self.bundle.vae.train()
            if self.bundle.condition_projector is not None:
                self.bundle.condition_projector.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.train.epochs}")
            self.optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(pbar, start=1):
                features = self._prepare_latents(batch, cached=cached)

                with torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                    loss_dict = self._forward_loss(features)
                    loss = loss_dict["loss"]
                    loss_to_backprop = loss / grad_accum

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Encountered non-finite loss at epoch={epoch}, step={self.global_step + 1}: {loss.item()}"
                    )

                self.scaler.scale(loss_to_backprop).backward()

                should_step = (batch_idx % grad_accum == 0) or (batch_idx == len(train_loader))
                if should_step:
                    if self.cfg.train.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.trainable_params, self.cfg.train.grad_clip_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                postfix = {"loss": float(loss.item())}
                if "loss_l1" in loss_dict:
                    postfix["l1"] = float(loss_dict["loss_l1"].item())
                if "loss_gradient" in loss_dict:
                    postfix["grad"] = float(loss_dict["loss_gradient"].item())
                pbar.set_postfix(**postfix)

                if self.global_step % self.cfg.train.log_every_steps == 0:
                    event: dict[str, float | int] = {
                        "step": self.global_step,
                        "train_loss": float(loss.item()),
                        "epoch": epoch,
                        "loss_noise": float(loss_dict["loss_noise"].item()),
                    }
                    if "loss_l1" in loss_dict:
                        event["loss_l1"] = float(loss_dict["loss_l1"].item())
                    if "loss_gradient" in loss_dict:
                        event["loss_gradient"] = float(loss_dict["loss_gradient"].item())
                    self.history.append(event)
                    self._persist_history()

                if should_step and self.global_step % self.cfg.train.save_every_steps == 0:
                    self._save_checkpoint(
                        f"controlnet_step_{self.global_step}.pt",
                        step=self.global_step,
                        epoch=epoch,
                        epoch_completed=False,
                    )

            val_loss = self._validate(val_loader, cached=cached)
            self.history.append({"step": self.global_step, "val_loss": float(val_loss), "epoch": epoch})
            self._persist_history()
            self._save_checkpoint(
                f"controlnet_epoch_{epoch}.pt",
                step=self.global_step,
                epoch=epoch,
                epoch_completed=True,
            )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("controlnet_best.pt", step=self.global_step, epoch=epoch, epoch_completed=True)

        self._save_checkpoint(
            "controlnet_last.pt",
            step=self.global_step,
            epoch=self.cfg.train.epochs,
            epoch_completed=True,
        )
        self._persist_history()

    @torch.no_grad()
    def vae_reconstruction_check(self, dataloader: DataLoader, n_batches: int = 1) -> dict[str, float]:
        self.bundle.vae.eval()
        mse_values: list[float] = []

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            clean = batch["clean_ct"].to(self.device)
            z = encode_to_latent(self.bundle.vae, clean, self.cfg.model.latent_scaling_factor)
            recon = decode_from_latent(self.bundle.vae, z, self.cfg.model.latent_scaling_factor, out_channels=clean.shape[1])
            mse = F.mse_loss(recon, clean)
            mse_values.append(float(mse.item()))

        mean_mse = float(sum(mse_values) / max(1, len(mse_values)))
        return {"recon_mse": mean_mse}
