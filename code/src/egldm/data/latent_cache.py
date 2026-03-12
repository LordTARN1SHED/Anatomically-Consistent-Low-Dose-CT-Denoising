from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from egldm.utils import ensure_dir, save_json


class LatentCacheDataset(Dataset):
    def __init__(self, cache_root: str | Path, split: str = "train") -> None:
        self.cache_root = Path(cache_root)
        index_path = self.cache_root / f"{split}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing cache index: {index_path}")

        import json

        self.items = json.loads(index_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.items[idx]
        npz = np.load(item["npz_path"])
        return {
            "z0": torch.from_numpy(npz["z0"]).float(),
            "c_ldct": torch.from_numpy(npz["c_ldct"]).float(),
            "edge_map": torch.from_numpy(npz["edge_map"]).float(),
            "context_noisy_ct": torch.zeros((0, 1, 1), dtype=torch.float32),
            "path": item["path"],
        }


def build_latent_cache(
    dataloader: DataLoader,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    output_dir: str | Path,
    split: str,
    device: torch.device,
) -> None:
    out_root = ensure_dir(output_dir)
    split_dir = ensure_dir(out_root / split)
    entries: list[dict[str, str]] = []

    idx = 0
    for batch in tqdm(dataloader, desc=f"Caching latents ({split})"):
        clean = batch["clean_ct"].to(device)
        noisy = batch["noisy_ct"].to(device)
        edge = batch["edge_map"].cpu().numpy()
        paths = batch["path"]

        with torch.no_grad():
            z0 = encode_fn(clean).cpu().numpy()
            c_ldct = encode_fn(noisy).cpu().numpy()

        bs = z0.shape[0]
        for b in range(bs):
            npz_path = split_dir / f"sample_{idx:07d}.npz"
            np.savez_compressed(npz_path, z0=z0[b], c_ldct=c_ldct[b], edge_map=edge[b])
            entries.append({"npz_path": str(npz_path), "path": str(paths[b])})
            idx += 1

    save_json(entries, out_root / f"{split}_index.json")
