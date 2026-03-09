from pathlib import Path
import shutil

SRC_ROOT = Path(r"D:\chim\data\dronebird_seg_raw")
DST_ROOT = Path(r"D:\chim\data\dronebird_det_clean")
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

    if len(parts) == 5:
        cx, cy, w, h = nums
        return ("bbox", cls, [cx, cy, w, h])

    if len(nums) >= 6 and len(nums) % 2 == 0:
        return ("seg", cls, nums)

    return None

def polygon_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(1.0, max(xs))
    y2 = min(1.0, max(ys))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def fmt_line(cls, vals):
    out = [str(cls)]
    for v in vals:
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        out.append(s if s else "0")
    return " ".join(out)

def convert_label_file(src_txt: Path, dst_txt: Path):
    lines = src_txt.read_text(encoding="utf-8", errors="ignore").splitlines()

    out_lines = []
    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            continue
        kind, cls, data = parsed

        if kind == "bbox":
            out_lines.append(fmt_line(cls, data))
        elif kind == "seg":
            bbox = polygon_to_bbox(data)
            out_lines.append(fmt_line(cls, bbox))

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(out_lines), encoding="utf-8")

def main():
    for split in SPLITS:
        src_img_dir = SRC_ROOT / split / "images"
        src_lbl_dir = SRC_ROOT / split / "labels"
        dst_img_dir = DST_ROOT / split / "images"
        dst_lbl_dir = DST_ROOT / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in src_img_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                shutil.copy2(img, dst_img_dir / img.name)

        for txt in src_lbl_dir.glob("*.txt"):
            convert_label_file(txt, dst_lbl_dir / txt.name)

    yaml_path = DST_ROOT / "data.yaml"
    yaml_path.write_text(
f"""path: {DST_ROOT.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  0: Drone
  1: Bird
""",
        encoding="utf-8"
    )

    print("[DONE] Created YOLO detect dataset at:", DST_ROOT)

if __name__ == "__main__":
    main()