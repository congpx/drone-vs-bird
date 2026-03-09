from pathlib import Path
import shutil

SRC_ROOT = Path(r"D:\chim\data\dronebird_seg_raw")
DST_ROOT = Path(r"D:\chim\data\dronebird_seg_clean_v2")
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
        return ("bbox", cls, nums)   # cx cy w h

    if len(nums) >= 6 and len(nums) % 2 == 0:
        return ("seg", cls, nums)    # x1 y1 x2 y2 ...

    return None

def bbox_to_polygon(cx, cy, w, h):
    x1 = max(0.0, cx - w / 2.0)
    y1 = max(0.0, cy - h / 2.0)
    x2 = min(1.0, cx + w / 2.0)
    y2 = min(1.0, cy + h / 2.0)
    return [x1, y1, x2, y1, x2, y2, x1, y2]

def fmt_line(cls, coords):
    out = [str(cls)]
    for v in coords:
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
        if kind == "seg":
            out_lines.append(fmt_line(cls, data))
        elif kind == "bbox":
            cx, cy, w, h = data
            poly = bbox_to_polygon(cx, cy, w, h)
            out_lines.append(fmt_line(cls, poly))

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

    print("[DONE] Created:", DST_ROOT)

if __name__ == "__main__":
    main()
    