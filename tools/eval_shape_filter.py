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


def polygon_area(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    return abs(cv2.contourArea(pts))


def polygon_to_bbox(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return [x1, y1, x2, y2]


def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def load_yaml_simple(path):
    # đủ dùng cho data.yaml rất đơn giản của YOLO
    data = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    current_dict_key = None
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
                current_dict_key = k
            else:
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                data[k] = v
                current_dict_key = None
        elif current_dict_key is not None:
            # parse names:
            #   0: Drone
            sub = line
            if ":" in sub:
                sk, sv = sub.split(":", 1)
                sk = sk.strip()
                sv = sv.strip().strip('"').strip("'")
                data[current_dict_key][int(sk)] = sv
    return data


def resolve_split_dir(data_yaml, split_name="test"):
    data = load_yaml_simple(data_yaml)
    root = Path(data["path"])
    split_rel = data[split_name]
    split_img_dir = root / split_rel
    split_lbl_dir = split_img_dir.parent.parent / "labels" / split_img_dir.name
    # vì cấu trúc chuẩn path + test/images
    # parent.parent = <root>/test
    # cách trên không đúng cho mọi trường hợp, nên tính lại:
    # split_img_dir = root/test/images
    # split_lbl_dir = root/test/labels
    split_root = split_img_dir.parent
    split_lbl_dir = split_root / "labels"
    return root, split_img_dir, split_lbl_dir, data.get("names", {})


def parse_gt_label_file(label_path, img_w, img_h):
    gts = []
    if not label_path.exists():
        return gts

    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return gts

    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            continue

        pts = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_w
            y = coords[i + 1] * img_h
            pts.append([x, y])

        bbox = polygon_to_bbox(pts)
        gts.append({
            "cls": cls_id,
            "poly": pts,
            "bbox": bbox,
        })
    return gts


def mask_features_from_polygon(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None

    area = abs(cv2.contourArea(pts))
    perimeter = float(cv2.arcLength(pts, True))
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    bbox_area = max(1.0, float(w * h))
    aspect_ratio = float(w) / max(float(h), 1.0)
    extent = area / bbox_area

    hull = cv2.convexHull(pts.astype(np.int32))
    hull_area = max(1.0, abs(cv2.contourArea(hull)))
    solidity = area / hull_area

    circularity = 0.0
    if perimeter > 1e-6:
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

    # fitEllipse cần >= 5 điểm
    eccentricity = 0.0
    if len(pts) >= 5:
        try:
            ellipse = cv2.fitEllipse(pts.astype(np.int32))
            (_, _), (MA, ma), _ = ellipse
            major = max(MA, ma)
            minor = min(MA, ma)
            if major > 1e-6:
                eccentricity = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major)))
        except Exception:
            eccentricity = 0.0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "bbox_w": float(w),
        "bbox_h": float(h),
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "solidity": float(solidity),
        "circularity": float(circularity),
        "eccentricity": float(eccentricity),
    }


def is_bird_like_for_drone_pred(feat, conf, min_area=20.0):
    """
    Luật hình dạng thực dụng:
    - loại mask quá nhỏ/nhiễu
    - nếu pred là Drone nhưng shape quá dài + lấp đầy kém + ít đặc => có xu hướng giống chim
    """
    if feat is None:
        return True, "invalid_feat"

    if feat["area"] < min_area:
        return True, "tiny_noise"

    elongated = (feat["aspect_ratio"] > 2.2 or feat["aspect_ratio"] < 0.45)
    sparse = feat["extent"] < 0.42
    irregular = feat["solidity"] < 0.82
    very_ecc = feat["eccentricity"] > 0.93

    # rule mạnh hơn khi conf thấp
    if conf < 0.45 and ((elongated and sparse) or (very_ecc and irregular)):
        return True, "bird_like_lowconf"

    if (elongated and sparse and irregular) or (very_ecc and irregular):
        return True, "bird_like_shape"

    return False, "keep"


def apply_shape_filter(det, drone_cls=0):
    """
    Chỉ filter các detection dự đoán là Drone.
    Bird giữ nguyên để tránh làm tụt recall bird quá mạnh.
    """
    if det["cls"] != drone_cls:
        det["filter_reason"] = "keep_non_drone"
        det["kept"] = True
        return det

    reject, reason = is_bird_like_for_drone_pred(det["feat"], det["conf"])
    det["kept"] = not reject
    det["filter_reason"] = reason
    return det


def match_predictions_to_gt(preds, gts, iou_thr=0.5):
    """
    Match greedy theo IoU.
    Trả về:
    - tp/fp/fn theo class
    - confusion matrix (gt_cls -> pred_cls)
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    matched_gt = set()
    preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)

    for p in preds_sorted:
        best_iou = -1.0
        best_j = -1

        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            iou = bbox_iou_xyxy(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            g = gts[best_j]
            matched_gt.add(best_j)
            confusion[g["cls"]][p["cls"]] += 1
            if g["cls"] == p["cls"]:
                tp[p["cls"]] += 1
            else:
                fp[p["cls"]] += 1
                fn[g["cls"]] += 1
        else:
            fp[p["cls"]] += 1

    for j, g in enumerate(gts):
        if j not in matched_gt:
            fn[g["cls"]] += 1

    return tp, fp, fn, confusion


def merge_counts(dst, src):
    for k, v in src.items():
        dst[k] += v


def merge_confusion(dst, src):
    for gk, row in src.items():
        for pk, v in row.items():
            dst[gk][pk] += v


def count_metrics(tp, fp, fn, class_ids):
    out = {}
    macro_p = []
    macro_r = []
    macro_f1 = []
    total_tp = total_fp = total_fn = 0

    for c in class_ids:
        t = tp.get(c, 0)
        f_p = fp.get(c, 0)
        f_n = fn.get(c, 0)

        prec = t / (t + f_p + 1e-9)
        rec = t / (t + f_n + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        out[c] = {
            "tp": int(t),
            "fp": int(f_p),
            "fn": int(f_n),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        macro_p.append(prec)
        macro_r.append(rec)
        macro_f1.append(f1)
        total_tp += t
        total_fp += f_p
        total_fn += f_n

    micro_p = total_tp / (total_tp + total_fp + 1e-9)
    micro_r = total_tp / (total_tp + total_fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    return {
        "per_class": out,
        "macro_precision": float(np.mean(macro_p) if macro_p else 0.0),
        "macro_recall": float(np.mean(macro_r) if macro_r else 0.0),
        "macro_f1": float(np.mean(macro_f1) if macro_f1 else 0.0),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
    }


def find_image_files(img_dir):
    files = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, img_dir, lbl_dir, names = resolve_split_dir(args.data, args.split)
    class_ids = sorted(list(names.keys())) if names else [0, 1]
    if not class_ids:
        class_ids = [0, 1]

    model = YOLO(args.model)
    images = find_image_files(img_dir)

    raw_tp = defaultdict(int)
    raw_fp = defaultdict(int)
    raw_fn = defaultdict(int)
    raw_conf = defaultdict(lambda: defaultdict(int))

    fil_tp = defaultdict(int)
    fil_fp = defaultdict(int)
    fil_fn = defaultdict(int)
    fil_conf = defaultdict(lambda: defaultdict(int))

    debug_rows = []

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_path = lbl_dir / f"{img_path.stem}.txt"
        gts = parse_gt_label_file(gt_path, w, h)

        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou_nms,
            imgsz=args.imgsz,
            device=args.device,
            retina_masks=True,
            verbose=False
        )
        r = results[0]

        preds_raw = []
        preds_filtered = []

        boxes = r.boxes
        masks = r.masks

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            mask_polys = []
            if masks is not None and masks.xy is not None:
                mask_polys = masks.xy
            else:
                mask_polys = [None] * len(xyxy)

            for i in range(len(xyxy)):
                poly = None
                if i < len(mask_polys) and mask_polys[i] is not None and len(mask_polys[i]) >= 3:
                    poly = mask_polys[i].tolist() if hasattr(mask_polys[i], "tolist") else mask_polys[i]
                else:
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                feat = mask_features_from_polygon(poly)
                det = {
                    "cls": int(cls[i]),
                    "conf": float(confs[i]),
                    "bbox": [float(v) for v in xyxy[i].tolist()],
                    "poly": poly,
                    "feat": feat,
                    "kept": True,
                    "filter_reason": "raw",
                }
                preds_raw.append(det)

                det2 = dict(det)
                det2 = apply_shape_filter(det2, drone_cls=args.drone_class)
                if det2["kept"]:
                    preds_filtered.append(det2)

                debug_rows.append({
                    "image": img_path.name,
                    "pred_cls": int(det["cls"]),
                    "pred_name": names.get(int(det["cls"]), str(det["cls"])),
                    "conf": float(det["conf"]),
                    "kept": bool(det2["kept"]),
                    "filter_reason": det2["filter_reason"],
                    "area": None if feat is None else feat["area"],
                    "aspect_ratio": None if feat is None else feat["aspect_ratio"],
                    "extent": None if feat is None else feat["extent"],
                    "solidity": None if feat is None else feat["solidity"],
                    "circularity": None if feat is None else feat["circularity"],
                    "eccentricity": None if feat is None else feat["eccentricity"],
                })

        tp, fp, fn, conf = match_predictions_to_gt(preds_raw, gts, iou_thr=args.iou_match)
        merge_counts(raw_tp, tp)
        merge_counts(raw_fp, fp)
        merge_counts(raw_fn, fn)
        merge_confusion(raw_conf, conf)

        tp, fp, fn, conf = match_predictions_to_gt(preds_filtered, gts, iou_thr=args.iou_match)
        merge_counts(fil_tp, tp)
        merge_counts(fil_fp, fp)
        merge_counts(fil_fn, fn)
        merge_confusion(fil_conf, conf)

        if idx % 200 == 0:
            print(f"[INFO] Processed {idx}/{len(images)} images")

    raw_metrics = count_metrics(raw_tp, raw_fp, raw_fn, class_ids)
    fil_metrics = count_metrics(fil_tp, fil_fp, fil_fn, class_ids)

    def conf_to_dict(confm):
        out = {}
        for g in class_ids:
            out[str(g)] = {}
            for p in class_ids:
                out[str(g)][str(p)] = int(confm[g][p])
        return out

    summary = {
        "names": {str(k): v for k, v in names.items()},
        "settings": {
            "model": args.model,
            "data": args.data,
            "split": args.split,
            "conf": args.conf,
            "iou_nms": args.iou_nms,
            "iou_match": args.iou_match,
            "imgsz": args.imgsz,
            "device": args.device,
            "drone_class": args.drone_class,
        },
        "raw": {
            "metrics": raw_metrics,
            "confusion": conf_to_dict(raw_conf),
        },
        "filtered": {
            "metrics": fil_metrics,
            "confusion": conf_to_dict(fil_conf),
        }
    }

    (outdir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    csv_path = outdir / "shape_debug.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "pred_cls", "pred_name", "conf", "kept", "filter_reason",
                "area", "aspect_ratio", "extent", "solidity", "circularity", "eccentricity"
            ]
        )
        writer.writeheader()
        for row in debug_rows:
            writer.writerow(row)

    print("[DONE] Saved:")
    print(" -", outdir / "metrics_summary.json")
    print(" -", outdir / "shape_debug.csv")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-nms", type=float, default=0.5)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--drone-class", type=int, default=0)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)