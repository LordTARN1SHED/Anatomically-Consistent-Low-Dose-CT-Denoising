from __future__ import annotations

import cv2
import numpy as np


def extract_canny_edge(
    x_norm: np.ndarray,
    blur_sigma: float = 1.0,
    low_threshold: int = 40,
    high_threshold: int = 120,
) -> np.ndarray:
    """Extract Canny edge map from [-1, 1] image, return [0, 1]."""
    x01 = np.clip((x_norm + 1.0) / 2.0, 0.0, 1.0)
    x_u8 = (x01 * 255.0).astype(np.uint8)

    if blur_sigma > 0:
        x_u8 = cv2.GaussianBlur(x_u8, (0, 0), blur_sigma)

    edges = cv2.Canny(x_u8, threshold1=low_threshold, threshold2=high_threshold)
    return (edges.astype(np.float32) / 255.0)[None, ...]
