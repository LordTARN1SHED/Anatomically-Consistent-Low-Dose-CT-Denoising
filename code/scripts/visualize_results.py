#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.eval.visualization import load_npy, save_comparison, save_roi_zoom


def _find_common_stems(paths: list[Path]) -> list[str]:
    stem_sets = [set(p.stem for p in d.glob("*.npy")) for d in paths]
    common = set.intersection(*stem_sets)
    return sorted(common)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison/error-map/ROI figures")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--ldct_dir", type=str, required=False)
    parser.add_argument("--redcnn_dir", type=str, required=False)
    parser.add_argument("--redcnn_label", type=str, default="RED-CNN")
    parser.add_argument("--ldm_dir", type=str, required=False)
    parser.add_argument("--ldm_label", type=str, default="Standard LDM")
    parser.add_argument("--pred_label", type=str, default="EG-LDM")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--roi", type=int, nargs=4, default=[90, 90, 140, 140])
    parser.add_argument("--max_cases", type=int, default=20)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [pred_dir, gt_dir]
    stems = _find_common_stems(required)
    stems = stems[: args.max_cases]

    for stem in stems:
        pred = load_npy(pred_dir / f"{stem}.npy")
        gt = load_npy(gt_dir / f"{stem}.npy")

        ldct = load_npy(Path(args.ldct_dir) / f"{stem}.npy") if args.ldct_dir else pred
        redcnn = load_npy(Path(args.redcnn_dir) / f"{stem}.npy") if args.redcnn_dir else None
        ldm = load_npy(Path(args.ldm_dir) / f"{stem}.npy") if args.ldm_dir else None

        save_comparison(
            ldct=ldct,
            redcnn=redcnn,
            ldm=ldm,
            egldm=pred,
            gt=gt,
            out_path=out_dir / f"{stem}_comparison.png",
            redcnn_label=args.redcnn_label,
            ldm_label=args.ldm_label,
            egldm_label=args.pred_label,
        )

        image_map = {"LDCT": ldct, args.pred_label: pred, "HDCT": gt}
        if redcnn is not None:
            image_map[args.redcnn_label] = redcnn
        if ldm is not None:
            image_map[args.ldm_label] = ldm

        save_roi_zoom(
            image_map=image_map,
            roi=tuple(args.roi),
            out_path=out_dir / f"{stem}_roi.png",
        )

    print(f"Saved {len(stems)} visualization cases to {out_dir}")


if __name__ == "__main__":
    main()
