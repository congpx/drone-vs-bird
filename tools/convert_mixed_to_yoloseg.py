from pathlib import Path
import shutil

SRC_ROOT = Path(r"D:\chim\data\dronebird_seg_raw")
DST_ROOT = Path(r"D:\chim\data\dronebird_seg_clean")

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def parse_line(line: str):
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 5:
        return None
    if not parts[0].isdigit():
        return None
    if not all(is_float(p) for p in parts[1:]):
        return None

    vals = [float(x) for x in parts]
    cls = int(vals[0])
    nums = vals[1:]

    # bbox: class cx cy w h
    if len(parts) == 5:
        cx, cy, w, h = nums
        return ("bbox", cls, [cx, cy, w, h])

    # seg polygon: class x1 y1 x2 y2 ...
    if len(nums) >= 6 and len(nums) % 2 == 0:
        return ("seg", cls, nums)

    return None

def bbox_to_polygon(cx, cy, w, h):
    x1 = max(0.0, cx - w / 2.0)
    y1 = max(0.0, cy - h / 2.0)
    x2 = min(1.0, cx + w / 2.0)
    y2 = min(1.0, cy + h / 2.0)
    # rectangle polygon
    return [x1, y1, x2, y1, x2, y2, x1, y2]

def fmt_line(cls, coords):
    vals = [str(cls)] + [("{:.10f}".format(v)).rstrip("0").rstrip(".") if "." in "{:.10f}".format(v) else "{:.10f}".format(v) for v in coords]
    return " ".join(vals)

def clean_label_file(src_txt: Path, dst_txt: Path):
    lines = src_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    segs = []
    bboxes = []

    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            continue
        kind, cls, data = parsed
        if kind == "seg":
            segs.append((cls, data))
        elif kind == "bbox":
            bboxes.append((cls, data))

    out_lines = []

    if segs:
        # ưu tiên dùng polygon thật
        for cls, coords in segs:
            out_lines.append(fmt_line(cls, coords))
    elif bboxes:
        # nếu không có polygon thì đổi bbox -> polygon chữ nhật
        for cls, (cx, cy, w, h) in bboxes:
            poly = bbox_to_polygon(cx, cy, w, h)
            out_lines.append(fmt_line(cls, poly))

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(out_lines), encoding="utf-8")

def copy_images_and_labels():
    for split in SPLITS:
        src_img_dir = SRC_ROOT / split / "images"
        src_lbl_dir = SRC_ROOT / split / "labels"

        dst_img_dir = DST_ROOT / split / "images"
        dst_lbl_dir = DST_ROOT / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # copy images
        for img in src_img_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                shutil.copy2(img, dst_img_dir / img.name)

        # convert labels
        for txt in src_lbl_dir.glob("*.txt"):
            clean_label_file(txt, dst_lbl_dir / txt.name)

def write_yaml():
    yaml_path = DST_ROOT / "data.yaml"
    text = f"""path: {DST_ROOT.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  0: Drone
  1: Bird
"""
    yaml_path.write_text(text, encoding="utf-8")

def summarize():
    print("=== SUMMARY ===")
    for split in SPLITS:
        img_dir = DST_ROOT / split / "images"
        lbl_dir = DST_ROOT / split / "labels"
        n_img = len([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        n_lbl = len(list(lbl_dir.glob("*.txt")))
        empty = 0
        for f in lbl_dir.glob("*.txt"):
            if f.read_text(encoding="utf-8", errors="ignore").strip() == "":
                empty += 1
        print(f"{split}: images={n_img}, labels={n_lbl}, empty_labels={empty}")
    print(f"YAML: {DST_ROOT / 'data.yaml'}")

if __name__ == "__main__":
    if DST_ROOT.exists():
        print(f"[INFO] Output exists: {DST_ROOT}")
    copy_images_and_labels()
    write_yaml()
    summarize()
    print("[DONE] Converted mixed labels -> YOLO-seg labels.")