from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from egldm.config import DataConfig
from egldm.data.edge import extract_canny_edge
from egldm.data.lidc import build_patient_splits, load_dicom_hu, load_prepared_split, scan_lidc_dicom_root
from egldm.data.noise import add_poisson_noise_image_domain, add_signal_dependent_gaussian_noise
from egldm.data.preprocess import CTWindow, dual_window_channels, normalize_hu_to_minus1_1, resize_2d


def _list_matching_pairs(clean_dir: Path, noisy_dir: Path) -> list[tuple[Path, Path]]:
    clean_files = sorted(clean_dir.glob("*.npy"))
    noisy_map = {p.name: p for p in noisy_dir.glob("*.npy")}
    pairs: list[tuple[Path, Path]] = []
    for c in clean_files:
        n = noisy_map.get(c.name)
        if n is not None:
            pairs.append((c, n))
    return pairs


def _split_indices(n: int, val_ratio: float = 0.1, seed: int = 42) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx


def _prepare_sample(
    clean_hu: np.ndarray,
    cfg: DataConfig,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    clean_hu = resize_2d(clean_hu, cfg.image_size)

    clean_norm = normalize_hu_to_minus1_1(clean_hu, cfg.hu_clip_min, cfg.hu_clip_max).astype(np.float32)

    if cfg.noise_model == "poisson":
        noisy_norm = add_poisson_noise_image_domain(clean_norm, peak=30.0, rng=rng)
    else:
        noisy_norm = add_signal_dependent_gaussian_noise(
            clean_norm,
            sigma_min=cfg.noise_sigma_min,
            sigma_max=cfg.noise_sigma_max,
            alpha=cfg.noise_alpha,
            rng=rng,
        )

    edge_map = extract_canny_edge(
        noisy_norm,
        blur_sigma=cfg.edge_blur_sigma,
        low_threshold=cfg.edge_low_threshold,
        high_threshold=cfg.edge_high_threshold,
    )

    window_stack = dual_window_channels(
        clean_hu,
        lung_window=CTWindow(cfg.lung_window_width, cfg.lung_window_level),
        soft_window=CTWindow(cfg.soft_window_width, cfg.soft_window_level),
    )

    return {
        "clean_ct": torch.from_numpy(clean_norm[None, ...]).float(),
        "noisy_ct": torch.from_numpy(noisy_norm[None, ...]).float(),
        "edge_map": torch.from_numpy(edge_map).float(),
        "window_stack": torch.from_numpy(window_stack).float(),
    }


def _prepare_context_slices(
    context_hus: list[np.ndarray],
    cfg: DataConfig,
    seed: int,
) -> torch.Tensor:
    context_noisy: list[np.ndarray] = []
    for offset, hu in enumerate(context_hus):
        clean_hu = resize_2d(hu, cfg.image_size)
        clean_norm = normalize_hu_to_minus1_1(clean_hu, cfg.hu_clip_min, cfg.hu_clip_max).astype(np.float32)
        rng = np.random.default_rng(seed + offset)
        if cfg.noise_model == "poisson":
            noisy_norm = add_poisson_noise_image_domain(clean_norm, peak=30.0, rng=rng)
        else:
            noisy_norm = add_signal_dependent_gaussian_noise(
                clean_norm,
                sigma_min=cfg.noise_sigma_min,
                sigma_max=cfg.noise_sigma_max,
                alpha=cfg.noise_alpha,
                rng=rng,
            )
        context_noisy.append(noisy_norm.astype(np.float32))

    if not context_noisy:
        return torch.zeros((0, cfg.image_size, cfg.image_size), dtype=torch.float32)
    return torch.from_numpy(np.stack(context_noisy, axis=0)).float()


class LIDCDicomSliceDataset(Dataset):
    def __init__(
        self,
        entries: list[dict[str, object]],
        cfg: DataConfig,
        seed: int = 42,
    ) -> None:
        self.entries = entries
        self.cfg = cfg
        self.seed = seed
        self._series_lookup: dict[tuple[str, str], list[dict[str, object]]] = {}
        for entry in self.entries:
            key = (str(entry.get("patient_id", "")), str(entry.get("series_instance_uid", "")))
            self._series_lookup.setdefault(key, []).append(entry)
        for items in self._series_lookup.values():
            items.sort(key=lambda item: int(item.get("series_slice_index", 0)))

    def __len__(self) -> int:
        return len(self.entries)

    def _neighbor_entries(self, entry: dict[str, object]) -> list[dict[str, object]]:
        k = max(0, int(self.cfg.neighboring_slices))
        if k <= 0:
            return []

        key = (str(entry.get("patient_id", "")), str(entry.get("series_instance_uid", "")))
        series_items = self._series_lookup.get(key, [])
        if not series_items:
            return []

        center_idx = int(entry.get("series_slice_index", 0))
        neighbors: list[dict[str, object]] = []
        for offset in range(-k, k + 1):
            if offset == 0:
                continue
            target_idx = min(max(center_idx + offset, 0), len(series_items) - 1)
            neighbors.append(series_items[target_idx])
        return neighbors

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[idx]
        path = Path(str(entry["dicom_path"]))
        hu = load_dicom_hu(path)
        sample = _prepare_sample(hu, self.cfg, np.random.default_rng(self.seed + idx))
        neighbors = self._neighbor_entries(entry)
        context_hus = [load_dicom_hu(Path(str(item["dicom_path"]))) for item in neighbors]
        sample["context_noisy_ct"] = _prepare_context_slices(context_hus, self.cfg, seed=self.seed + idx * 17 + 1000)
        sample["path"] = str(path)
        sample["patient_id"] = str(entry.get("patient_id", ""))
        return sample


class NpyPairDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], cfg: DataConfig, seed: int = 42) -> None:
        self.pairs = pairs
        self.cfg = cfg
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        clean_path, noisy_path = self.pairs[idx]
        clean_hu = np.load(clean_path).astype(np.float32)
        noisy_hu = np.load(noisy_path).astype(np.float32)

        clean_hu = resize_2d(clean_hu, self.cfg.image_size)
        noisy_hu = resize_2d(noisy_hu, self.cfg.image_size)

        clean_norm = normalize_hu_to_minus1_1(clean_hu, self.cfg.hu_clip_min, self.cfg.hu_clip_max).astype(np.float32)
        noisy_norm = normalize_hu_to_minus1_1(noisy_hu, self.cfg.hu_clip_min, self.cfg.hu_clip_max).astype(np.float32)

        edge_map = extract_canny_edge(
            noisy_norm,
            blur_sigma=self.cfg.edge_blur_sigma,
            low_threshold=self.cfg.edge_low_threshold,
            high_threshold=self.cfg.edge_high_threshold,
        )

        window_stack = dual_window_channels(
            clean_hu,
            lung_window=CTWindow(self.cfg.lung_window_width, self.cfg.lung_window_level),
            soft_window=CTWindow(self.cfg.soft_window_width, self.cfg.soft_window_level),
        )

        return {
            "clean_ct": torch.from_numpy(clean_norm[None, ...]).float(),
            "noisy_ct": torch.from_numpy(noisy_norm[None, ...]).float(),
            "edge_map": torch.from_numpy(edge_map).float(),
            "window_stack": torch.from_numpy(window_stack).float(),
            "context_noisy_ct": torch.zeros((0, self.cfg.image_size, self.cfg.image_size), dtype=torch.float32),
            "path": str(clean_path),
        }


class SyntheticCTDataset(Dataset):
    def __init__(self, n_samples: int, cfg: DataConfig, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.cfg = cfg
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def _make_hu(self, rng: np.random.Generator) -> np.ndarray:
        size = self.cfg.image_size
        img = np.full((size, size), -900.0, dtype=np.float32)

        # Soft-tissue like ellipses.
        for _ in range(4):
            center = (
                int(rng.integers(size // 4, 3 * size // 4)),
                int(rng.integers(size // 4, 3 * size // 4)),
            )
            axes = (
                int(rng.integers(size // 10, size // 4)),
                int(rng.integers(size // 12, size // 5)),
            )
            angle = float(rng.uniform(0, 180))
            hu_val = float(rng.uniform(-300, 200))
            cv2.ellipse(img, center, axes, angle, 0, 360, hu_val, thickness=-1)

        # Vessel-like thin lines.
        for _ in range(10):
            p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            hu_val = float(rng.uniform(-100, 200))
            cv2.line(img, p1, p2, hu_val, thickness=int(rng.integers(1, 3)))

        img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
        return np.clip(img, self.cfg.hu_clip_min, self.cfg.hu_clip_max)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(self.seed + idx)
        hu = self._make_hu(rng)
        sample = _prepare_sample(hu, self.cfg, rng)
        sample["context_noisy_ct"] = torch.zeros((0, self.cfg.image_size, self.cfg.image_size), dtype=torch.float32)
        sample["path"] = f"synthetic_{idx:06d}"
        return sample


class _SubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


def _load_dicom_entries(cfg: DataConfig, split: str, seed: int) -> list[dict[str, object]]:
    if cfg.prepared_index_dir is not None:
        index_path = Path(cfg.prepared_index_dir) / f"{split}_index.json"
        if index_path.exists():
            return load_prepared_split(cfg.prepared_index_dir, split)

    if cfg.dicom_root is None:
        raise ValueError("data.dicom_root is required for mode='dicom'.")

    records, _ = scan_lidc_dicom_root(
        cfg.dicom_root,
        hu_clip_min=cfg.hu_clip_min,
        hu_clip_max=cfg.hu_clip_max,
        strict_ct_only=cfg.strict_ct_only,
        min_slice_size=cfg.min_slice_size,
    )
    split_records, _ = build_patient_splits(
        records,
        val_ratio=cfg.val_split_ratio,
        test_ratio=cfg.test_split_ratio,
        seed=cfg.split_seed if cfg.split_seed is not None else seed,
    )
    return split_records[split]


def _build_dataset(cfg: DataConfig, split: str, seed: int) -> Dataset:
    if cfg.mode == "synthetic":
        n = cfg.max_train_samples if split == "train" else cfg.max_val_samples
        n = 128 if n is None else int(n)
        return SyntheticCTDataset(n_samples=n, cfg=cfg, seed=seed)

    if cfg.mode == "dicom":
        entries = _load_dicom_entries(cfg, split=split, seed=seed)
        max_n = cfg.max_train_samples if split == "train" else cfg.max_val_samples
        if max_n is not None:
            entries = entries[: int(max_n)]
        return LIDCDicomSliceDataset(entries=entries, cfg=cfg, seed=seed)

    if cfg.mode == "npy_pairs":
        if cfg.clean_dir is None or cfg.noisy_dir is None:
            raise ValueError("data.clean_dir and data.noisy_dir are required for mode='npy_pairs'.")
        pairs = _list_matching_pairs(Path(cfg.clean_dir), Path(cfg.noisy_dir))
        if len(pairs) == 0:
            raise FileNotFoundError(f"No matching .npy pairs found in {cfg.clean_dir} and {cfg.noisy_dir}")
        tr, va = _split_indices(len(pairs), val_ratio=0.1, seed=seed)
        selected = tr if split == "train" else va
        max_n = cfg.max_train_samples if split == "train" else cfg.max_val_samples
        if max_n is not None:
            selected = selected[: int(max_n)]
        pairs_sel = [pairs[i] for i in selected]
        return NpyPairDataset(pairs=pairs_sel, cfg=cfg, seed=seed)

    raise ValueError(f"Unsupported data.mode: {cfg.mode}")


def _make_loader(dataset: Dataset, cfg: DataConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def build_dataloaders(cfg: DataConfig, seed: int = 42, eval_split: str = "val") -> tuple[DataLoader, DataLoader]:
    train_ds = _build_dataset(cfg, split="train", seed=seed)
    val_ds = _build_dataset(cfg, split=eval_split, seed=seed)
    return _make_loader(train_ds, cfg, shuffle=True), _make_loader(val_ds, cfg, shuffle=False)
