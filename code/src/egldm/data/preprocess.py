from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CTWindow:
    width: float
    level: float


def clip_hu(x: np.ndarray, hu_min: float = -1000.0, hu_max: float = 1000.0) -> np.ndarray:
    return np.clip(x, hu_min, hu_max)


def normalize_hu_to_minus1_1(x_hu: np.ndarray, hu_min: float = -1000.0, hu_max: float = 1000.0) -> np.ndarray:
    x = clip_hu(x_hu, hu_min, hu_max)
    x = (x - hu_min) / (hu_max - hu_min)
    return x * 2.0 - 1.0


def denormalize_minus1_1_to_hu(x_norm: np.ndarray, hu_min: float = -1000.0, hu_max: float = 1000.0) -> np.ndarray:
    x = (x_norm + 1.0) / 2.0
    return x * (hu_max - hu_min) + hu_min


def apply_window(x_hu: np.ndarray, window: CTWindow) -> np.ndarray:
    low = window.level - window.width / 2.0
    high = window.level + window.width / 2.0
    x = np.clip(x_hu, low, high)
    return (x - low) / (high - low + 1e-8)


def dual_window_channels(
    x_hu: np.ndarray,
    lung_window: CTWindow,
    soft_window: CTWindow,
) -> np.ndarray:
    lung = apply_window(x_hu, lung_window)
    soft = apply_window(x_hu, soft_window)
    mixed = np.clip(0.5 * (lung + soft), 0.0, 1.0)
    return np.stack([lung, soft, mixed], axis=0).astype(np.float32)


def resize_2d(x: np.ndarray, size: int = 256, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(x, (size, size), interpolation=interpolation)
