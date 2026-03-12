from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from skimage.metrics import structural_similarity as sk_ssim
from torchvision.models import Inception_V3_Weights, inception_v3


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    # pred/target expected as HxW in [-1, 1]
    return float(sk_ssim(pred, target, data_range=2.0))


def _to_inception_input(x: torch.Tensor) -> torch.Tensor:
    # x: [B,1,H,W] in [-1,1]
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x


class FIDFeatureExtractor(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        try:
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
        except Exception:
            # Fallback without pretrained weights when offline.
            model = inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Identity()
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        x = _to_inception_input(x.to(self.device))
        feat = self.model(x)
        return feat.detach().cpu().numpy()


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_fid_from_features(features_real: np.ndarray, features_fake: np.ndarray) -> float:
    mu_r = np.mean(features_real, axis=0)
    mu_f = np.mean(features_fake, axis=0)
    sigma_r = np.cov(features_real, rowvar=False)
    sigma_f = np.cov(features_fake, rowvar=False)
    return frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
