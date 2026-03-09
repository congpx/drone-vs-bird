#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def yolo_seg_line_to_mask(line: str, h: int, w: int) -> tuple[int, np.ndarray] | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    cls_id = int(float(parts[0]))
    coords = list(map(float, parts[1:]))
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = np.round(pts).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return cls_id, mask


def get_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_features(mask: np.ndarray) -> dict:
    mask_u8 = (mask > 0).astype(np.uint8)
    area = int(mask_u8.sum())
    bbox = get_bbox_from_mask(mask_u8)
    if bbox is None or area == 0:
        return {
            "area": 0, "bbox_area": 0, "fill_ratio": 0.0, "aspect_ratio": 0.0,
            "perimeter_compactness": 999.0, "solidity": 0.0,
        }
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    bbox_area = bw * bh
    fill_ratio = area / max(1, bbox_area)
    aspect_ratio = max(bw, bh) / max(1, min(bw, bh))

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        perimeter = 0.0
        solidity = 0.0
    else:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(cnt)
        solidity = float(cnt_area / hull_area) if hull_area > 0 else 0.0

    compactness = (perimeter * perimeter) / (4.0 * math.pi * max(1.0, area)) if perimeter > 0 else 999.0
    return {
        "area": area,
        "bbox_area": bbox_area,
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect_ratio,
        "perimeter_compactness": compactness,
        "solidity": solidity,
    }


def is_bird_like_shape(feat: dict, pred_cls: int) -> bool:
    # Chỉ suppress khi model đang dự đoán là drone nhưng hình dạng lại khá giống chim.
    if pred_cls != 0:
        return False

    fill_ratio = feat["fill_ratio"]
    aspect_ratio = feat["aspect_ratio"]
    compactness = feat["perimeter_compactness"]
    solidity = feat["solidity"]
    area = feat["area"]

    rule1 = (fill_ratio < 0.42 and compactness > 2.40)
    rule2 = (solidity < 0.82 and compactness > 2.15)
    rule3 = (aspect_ratio > 2.60 and fill_ratio < 0.48)
    rule4 = (area > 15 and fill_ratio < 0.38 and solidity < 0.86)
    return rule1 or rule2 or rule3 or rule4


def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def load_gt_masks(label_path: Path, h: int, w: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    items = []
    for line in label_path.read_text().splitlines():
        parsed = yolo_seg_line_to_mask(line, h, w)
        if parsed is not None:
            items.append(parsed)
    return items


def evaluate_one_image(preds: list[dict], gts: list[tuple[int, np.ndarray]], iou_thr: float = 0.5) -> dict:
    matched_gt = set()
    tp = []
    fp = []
    confusion = defaultdict(int)

    preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)
    for pred in preds_sorted:
        best_iou = 0.0
        best_j = -1
        best_gt_cls = None
        for j, (gt_cls, gt_mask) in enumerate(gts):
            if j in matched_gt:
                continue
            iou = iou_masks(pred["mask"], gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_j = j
                best_gt_cls = gt_cls

        if best_iou >= iou_thr and best_j >= 0:
            matched_gt.add(best_j)
            if pred["cls"] == best_gt_cls:
                tp.append(pred)
                confusion[(best_gt_cls, pred["cls"])] += 1
            else:
                fp.append(pred)
                confusion[(best_gt_cls, pred["cls"])] += 1
        else:
            fp.append(pred)
            confusion[(-1, pred["cls"])] += 1

    fn = []
    for j, (gt_cls, gt_mask) in enumerate(gts):
        if j not in matched_gt:
            fn.append({"cls": gt_cls, "mask": gt_mask})
            confusion[(gt_cls, -1)] += 1

    return {"tp": tp, "fp": fp, "fn": fn, "confusion": confusion}


def summarize(results: list[dict]) -> dict:
    total_tp = sum(len(r["tp"]) for r in results)
    total_fp = sum(len(r["fp"]) for r in results)
    total_fn = sum(len(r["fn"]) for r in results)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)

    cls_stats = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}
    confusion = defaultdict(int)

    for r in results:
        for p in r["tp"]:
            cls_stats[p["cls"]]["tp"] += 1
        for p in r["fp"]:
            cls_stats[p["cls"]]["fp"] += 1
        for f in r["fn"]:
            cls_stats[f["cls"]]["fn"] += 1
        for k, v in r["confusion"].items():
            confusion[str(k)] += int(v)

    cls_metrics = {}
    for cls_id, s in cls_stats.items():
        p = s["tp"] / max(1, s["tp"] + s["fp"])
        r = s["tp"] / max(1, s["tp"] + s["fn"])
        cls_f1 = (2 * p * r) / max(1e-12, p + r)
        cls_metrics[cls_id] = {
            "precision": p,
            "recall": r,
            "f1": cls_f1,
            **s,
        }

    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_class": cls_metrics,
        "confusion": dict(confusion),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in Path(args.images).glob("*.*") if p.suffix.lower() in IMG_EXTS])
    model = YOLO(args.model)

    raw_results = []
    filtered_results = []
    debug_rows = []

    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_path = Path(args.labels) / f"{img_path.stem}.txt"
        gts = load_gt_masks(gt_path, h, w)

        pred = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        raw_preds: list[dict] = []
        filt_preds: list[dict] = []

        if pred.masks is not None and pred.boxes is not None:
            masks = pred.masks.data.cpu().numpy()
            clss = pred.boxes.cls.cpu().numpy().astype(int)
            confs = pred.boxes.conf.cpu().numpy()

            for i in range(len(masks)):
                mask = (masks[i] > 0.5).astype(np.uint8)
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0).astype(np.uint8)
                item = {"cls": int(clss[i]), "conf": float(confs[i]), "mask": mask}
                raw_preds.append(item)

                feat = mask_features(mask)
                suppress = is_bird_like_shape(feat, item["cls"])
                debug_rows.append({
                    "image": img_path.name,
                    "pred_cls": int(item["cls"]),
                    "conf": float(item["conf"]),
                    "suppress": bool(suppress),
                    **feat,
                })
                if not suppress:
                    filt_preds.append(item)

        raw_eval = evaluate_one_image(raw_preds, gts, iou_thr=args.iou)
        filtered_eval = evaluate_one_image(filt_preds, gts, iou_thr=args.iou)
        raw_results.append(raw_eval)
        filtered_results.append(filtered_eval)

        if idx % 50 == 0:
            print(f"[INFO] processed {idx}/{len(image_paths)} images")

    raw_summary = summarize(raw_results)
    filtered_summary = summarize(filtered_results)
    all_summary = {
        "raw": raw_summary,
        "filtered": filtered_summary,
        "settings": {
            "model": args.model,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
        },
    }

    (outdir / "metrics_summary.json").write_text(json.dumps(all_summary, indent=2, ensure_ascii=False))

    import csv
    with open(outdir / "shape_debug.csv", "w", newline="", encoding="utf-8") as f:
        if debug_rows:
            writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_rows)

    print(json.dumps(all_summary, indent=2, ensure_ascii=False))
    print(f"[DONE] results written to: {outdir}")


if __name__ == "__main__":
    main()
