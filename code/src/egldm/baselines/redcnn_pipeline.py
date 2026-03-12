from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from egldm.config import DataConfig, EvalConfig
from egldm.data import build_dataloaders
from egldm.eval.metrics import FIDFeatureExtractor, compute_fid_from_features, psnr, rmse, ssim
from egldm.models.redcnn import REDCNN
from egldm.utils import ensure_dir, get_device, save_json, seed_everything


@dataclass
class REDCNNModelConfig:
    base_channels: int = 96
    kernel_size: int = 5


@dataclass
class REDCNNTrainConfig:
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    mixed_precision: bool = False
    log_every_steps: int = 20
    save_every_steps: int = 200
    loss_type: str = "mse"  # mse | l1 | mixed
    l1_weight: float = 0.1


@dataclass
class REDCNNConfig:
    seed: int = 42
    output_dir: str = "outputs/redcnn"
    data: DataConfig = field(default_factory=DataConfig)
    model: REDCNNModelConfig = field(default_factory=REDCNNModelConfig)
    train: REDCNNTrainConfig = field(default_factory=REDCNNTrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _dataclass_from_dict(cls: Any, payload: dict[str, Any]) -> Any:
    fields = cls.__dataclass_fields__.keys()
    kwargs = {k: v for k, v in payload.items() if k in fields}
    return cls(**kwargs)


def load_redcnn_config(path: str | Path) -> REDCNNConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    top = _dataclass_from_dict(REDCNNConfig, raw)
    top.data = _dataclass_from_dict(DataConfig, raw.get("data", {}))
    top.model = _dataclass_from_dict(REDCNNModelConfig, raw.get("model", {}))
    top.train = _dataclass_from_dict(REDCNNTrainConfig, raw.get("train", {}))
    top.eval = _dataclass_from_dict(EvalConfig, raw.get("eval", {}))
    return top


class REDCNNTrainer:
    def __init__(self, cfg: REDCNNConfig) -> None:
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

        self.model = REDCNN(
            base_channels=cfg.model.base_channels,
            kernel_size=cfg.model.kernel_size,
        ).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

        self.out_dir = ensure_dir(cfg.output_dir)
        self.ckpt_dir = ensure_dir(self.out_dir / "checkpoints")
        self.log_dir = ensure_dir(self.out_dir / "logs")

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

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.train.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.cfg.train.loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.cfg.train.loss_type == "mixed":
            mse = F.mse_loss(pred, target)
            l1 = F.l1_loss(pred, target)
            return mse + float(self.cfg.train.l1_weight) * l1
        raise ValueError(f"Unsupported RED-CNN loss_type: {self.cfg.train.loss_type}")

    def _save_checkpoint(self, name: str, step: int, epoch: int) -> None:
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.cfg),
            },
            self.ckpt_dir / name,
        )

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                noisy = batch["noisy_ct"].to(self.device)
                clean = batch["clean_ct"].to(self.device)
                with torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                    pred = self.model(noisy)
                    loss = self._loss(pred, clean)
                losses.append(float(loss.item()))

        self.model.train()
        return float(sum(losses) / max(1, len(losses)))

    def run(self) -> None:
        train_loader, val_loader = build_dataloaders(self.cfg.data, seed=self.cfg.seed, eval_split="val")

        global_step = 0
        best_val_loss = float("inf")
        history: list[dict[str, float | int]] = []

        for epoch in range(1, self.cfg.train.epochs + 1):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.train.epochs}")
            for batch in pbar:
                noisy = batch["noisy_ct"].to(self.device)
                clean = batch["clean_ct"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                    pred = self.model(noisy)
                    loss = self._loss(pred, clean)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Encountered non-finite loss at epoch={epoch}, step={global_step + 1}: {loss.item()}"
                    )

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.cfg.train.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.cfg.train.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                global_step += 1
                pbar.set_postfix(loss=float(loss.item()))

                if global_step % self.cfg.train.log_every_steps == 0:
                    history.append({"step": global_step, "train_loss": float(loss.item()), "epoch": epoch})

                if global_step % self.cfg.train.save_every_steps == 0:
                    self._save_checkpoint(f"redcnn_step_{global_step}.pt", step=global_step, epoch=epoch)

            val_loss = self._validate(val_loader)
            history.append({"step": global_step, "val_loss": float(val_loss), "epoch": epoch})
            self._save_checkpoint(f"redcnn_epoch_{epoch}.pt", step=global_step, epoch=epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("redcnn_best.pt", step=global_step, epoch=epoch)

        self._save_checkpoint("redcnn_last.pt", step=global_step, epoch=self.cfg.train.epochs)
        save_json({"history": history}, self.log_dir / "loss_history.json")


class REDCNNEvaluator:
    def __init__(self, cfg: REDCNNConfig, checkpoint_path: str | Path, output_dir: str | Path) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = get_device()

        self.model = REDCNN(
            base_channels=cfg.model.base_channels,
            kernel_size=cfg.model.kernel_size,
        ).to(self.device).eval()

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

        self.out_dir = ensure_dir(output_dir)
        self.pred_dir = ensure_dir(self.out_dir / "preds")
        self.gt_dir = ensure_dir(self.out_dir / "gts")
        self.noisy_dir = ensure_dir(self.out_dir / "noisy")

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
        for batch in tqdm(val_loader, desc="Evaluate RED-CNN"):
            clean = batch["clean_ct"].to(self.device)
            noisy = batch["noisy_ct"].to(self.device)
            with torch.no_grad():
                pred = self.model(noisy).clamp(-1.0, 1.0)

            for i in range(clean.shape[0]):
                c_np = self._to_numpy_2d(clean[i])
                n_np = self._to_numpy_2d(noisy[i])
                p_np = self._to_numpy_2d(pred[i])

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
                    self._save_png(p_np, self.pred_dir / f"{stem}.png")
                    self._save_png(c_np, self.gt_dir / f"{stem}.png")
                    self._save_png(n_np, self.noisy_dir / f"{stem}.png")

                if fid_extractor is not None:
                    real_feats.append(fid_extractor.extract(clean[i : i + 1]))
                    fake_feats.append(fid_extractor.extract(pred[i : i + 1]))

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
