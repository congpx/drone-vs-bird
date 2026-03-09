#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def polygon_to_bbox(vals: list[float]) -> tuple[float, float, float, float]:
    xs = vals[0::2]
    ys = vals[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_c = (x_min + x_max) / 2.0
    y_c = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return x_c, y_c, w, h


def convert_label(seg_label_path: Path, det_label_path: Path) -> None:
    det_label_path.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    if not seg_label_path.exists():
        det_label_path.write_text("")
        return

    for line in seg_label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = parts[0]
        coords = list(map(float, parts[1:]))
        x_c, y_c, w, h = polygon_to_bbox(coords)
        out_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    det_label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


def copy_images(src_img_dir: Path, dst_img_dir: Path) -> None:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for img_path in src_img_dir.glob("*.*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            shutil.copy2(img_path, dst_img_dir / img_path.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="YOLO segmentation dataset root")
    ap.add_argument("--dst", required=True, help="YOLO detection dataset root")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    for split in ["train", "val", "test"]:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        copy_images(src_img_dir, dst_img_dir)

        for img_path in src_img_dir.glob("*.*"):
            stem = img_path.stem
            seg_label = src_lbl_dir / f"{stem}.txt"
            det_label = dst_lbl_dir / f"{stem}.txt"
            convert_label(seg_label, det_label)

        print(f"[OK] Converted split={split}")

    print(f"[DONE] Detection dataset written to: {dst_root}")


if __name__ == "__main__":
    main()
