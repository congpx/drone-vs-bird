import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml_simple(path):
    data = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    current = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.startswith("-"):
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if v == "":
                data[k] = {}
                current = k
            else:
                data[k] = v.strip('"').strip("'")
                current = None
        elif current is not None and ":" in line:
            sk, sv = line.split(":", 1)
            data[current][int(sk.strip())] = sv.strip().strip('"').strip("'")
    return data


def resolve_dirs(data_yaml, split="test"):
    data = load_yaml_simple(data_yaml)
    root = Path(data["path"])
    img_dir = root / data[split]
    lbl_dir = img_dir.parent / "labels"
    return img_dir, lbl_dir, data.get("names", {})


def polygon_to_bbox(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return [x1, y1, x2, y2]


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def parse_gt_file(label_path, w, h):
    objs = []
    if not label_path.exists():
        return objs
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return objs
    for line in txt.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        vals = [float(x) for x in p[1:]]
        if len(p) == 5:
            cx, cy, bw, bh = vals
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            bbox = [x1, y1, x2, y2]
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            pts = []
            for i in range(0, len(vals), 2):
                pts.append([vals[i] * w, vals[i + 1] * h])
            bbox = polygon_to_bbox(pts)
        else:
            continue
        objs.append({"cls": cls, "bbox": bbox})
    return objs


def mask_features(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None

    area = abs(cv2.contourArea(pts))
    peri = float(cv2.arcLength(pts, True))
    x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    bbox_area = max(1.0, float(bw * bh))
    extent = area / bbox_area
    aspect = float(bw) / max(float(bh), 1.0)

    hull = cv2.convexHull(pts.astype(np.int32))
    hull_area = max(1.0, abs(cv2.contourArea(hull)))
    solidity = area / hull_area

    circularity = 0.0 if peri <= 1e-6 else 4.0 * math.pi * area / (peri * peri)

    eccentricity = 0.0
    if len(pts) >= 5:
        try:
            (_, _), (ma1, ma2), _ = cv2.fitEllipse(pts.astype(np.int32))
            major = max(ma1, ma2)
            minor = min(ma1, ma2)
            if major > 1e-6:
                eccentricity = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major)))
        except Exception:
            pass

    return {
        "area": area,
        "aspect_ratio": aspect,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "eccentricity": eccentricity,
    }


def reject_bird_like_v2(feat, conf):
    """
    V2: bảo toàn recall hơn v1
    - Không filter prediction conf cao
    - Chỉ filter nhóm conf thấp / trung bình thấp
    - Cần nhiều dấu hiệu bird-like cùng lúc
    """
    if feat is None:
        return False, "keep_invalid_feat"

    # Giữ prediction rất tự tin
    if conf >= 0.60:
        return False, "keep_highconf"

    # Tránh giết TP ở đối tượng nhỏ nhưng hợp lệ
    if feat["area"] < 12:
        return False, "keep_tiny"

    aspect = feat["aspect_ratio"]
    elongated = (aspect > 2.8 or aspect < 0.36)
    sparse = feat["extent"] < 0.34
    irregular = feat["solidity"] < 0.74
    very_ecc = feat["eccentricity"] > 0.965
    roundish = feat["circularity"] > 0.72

    # Rule mạnh: chỉ reject khi conf khá thấp và có >=3 dấu hiệu
    if conf < 0.35:
        score = int(elongated) + int(sparse) + int(irregular) + int(very_ecc)
        if score >= 3:
            return True, "reject_lowconf_strongbird"

    # Rule phụ: conf trung bình thấp + hình rất kéo dài và rỗng
    if conf < 0.45 and elongated and sparse and (irregular or very_ecc):
        return True, "reject_midconf_elongated"

    # Không reject các shape tròn/gọn hơn vì dễ là drone ở xa
    if roundish and feat["solidity"] > 0.82:
        return False, "keep_compact"

    return False, "keep"


def main(args):
    img_dir, lbl_dir, names = resolve_dirs(args.data, args.split)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    drone_tp_raw = drone_fp_raw = drone_fn_raw = 0
    drone_tp_fil = drone_fp_fil = drone_fn_fil = 0
    bird_fp_as_drone_raw = 0
    bird_fp_as_drone_fil = 0

    debug_rows = []

    for idx, img_path in enumerate(files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = parse_gt_file(lbl_dir / f"{img_path.stem}.txt", w, h)

        gt_drones = [g for g in gts if g["cls"] == 0]
        gt_birds = [g for g in gts if g["cls"] == 1]

        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou_nms,
            device=args.device,
            retina_masks=True,
            verbose=False
        )
        r = results[0]

        raw_drone_preds = []
        filtered_drone_preds = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            polys = r.masks.xy if (r.masks is not None and r.masks.xy is not None) else [None] * len(xyxy)

            for i in range(len(xyxy)):
                pred_cls = int(cls[i])
                conf = float(confs[i])

                if pred_cls != 0:
                    continue

                if i < len(polys) and polys[i] is not None and len(polys[i]) >= 3:
                    poly = polys[i].tolist() if hasattr(polys[i], "tolist") else polys[i]
                else:
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                feat = mask_features(poly)
                pred = {
                    "bbox": [float(v) for v in xyxy[i].tolist()],
                    "conf": conf,
                    "feat": feat,
                }
                raw_drone_preds.append(pred)

                reject, reason = reject_bird_like_v2(feat, conf)
                keep = not reject
                if keep:
                    filtered_drone_preds.append(pred)

                debug_rows.append({
                    "image": img_path.name,
                    "conf": conf,
                    "kept": keep,
                    "reason": reason,
                    "area": None if feat is None else feat["area"],
                    "aspect_ratio": None if feat is None else feat["aspect_ratio"],
                    "extent": None if feat is None else feat["extent"],
                    "solidity": None if feat is None else feat["solidity"],
                    "circularity": None if feat is None else feat["circularity"],
                    "eccentricity": None if feat is None else feat["eccentricity"],
                })

        def eval_preds(preds):
            tp = fp = fn = 0
            bird_fp = 0
            matched = set()

            for p in sorted(preds, key=lambda x: x["conf"], reverse=True):
                best_iou = -1.0
                best_j = -1
                for j, g in enumerate(gt_drones):
                    if j in matched:
                        continue
                    iou = bbox_iou(p["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_j >= 0 and best_iou >= args.iou_match:
                    matched.add(best_j)
                    tp += 1
                else:
                    fp += 1
                    if any(bbox_iou(p["bbox"], b["bbox"]) >= args.iou_match for b in gt_birds):
                        bird_fp += 1

            fn = len(gt_drones) - len(matched)
            return tp, fp, fn, bird_fp

        tp, fp, fn, bird_fp = eval_preds(raw_drone_preds)
        drone_tp_raw += tp
        drone_fp_raw += fp
        drone_fn_raw += fn
        bird_fp_as_drone_raw += bird_fp

        tp, fp, fn, bird_fp = eval_preds(filtered_drone_preds)
        drone_tp_fil += tp
        drone_fp_fil += fp
        drone_fn_fil += fn
        bird_fp_as_drone_fil += bird_fp

        if idx % 200 == 0:
            print(f"[INFO] Processed {idx}/{len(files)}")

    def metrics(tp, fp, fn):
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }

    summary = {
        "settings": vars(args),
        "raw": {
            "drone_metrics": metrics(drone_tp_raw, drone_fp_raw, drone_fn_raw),
            "bird_induced_false_drone_alarms": int(bird_fp_as_drone_raw),
        },
        "filtered": {
            "drone_metrics": metrics(drone_tp_fil, drone_fp_fil, drone_fn_fil),
            "bird_induced_false_drone_alarms": int(bird_fp_as_drone_fil),
        },
    }

    (outdir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (outdir / "shape_debug.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "conf", "kept", "reason",
                "area", "aspect_ratio", "extent", "solidity", "circularity", "eccentricity"
            ],
        )
        writer.writeheader()
        for row in debug_rows:
            writer.writerow(row)

    print("[DONE] Saved:")
    print(" -", outdir / "metrics_summary.json")
    print(" -", outdir / "shape_debug.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-nms", type=float, default=0.5)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    main(args)