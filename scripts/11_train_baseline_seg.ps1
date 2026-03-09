param(
    [int]$Epochs = 40,
    [int]$Batch = 12,
    [int]$Workers = 4,
    [int]$Imgsz = 640,
    [string]$RunName = "baseline_seg_rgb"
)

$ErrorActionPreference = "Stop"
conda activate chim-yolo

yolo task=segment mode=train `
    model=yolo11n-seg.pt `
    data="D:\chim\data\dronebird_seg_clean_v2\data.yaml" `
    epochs=$Epochs `
    imgsz=$Imgsz `
    batch=$Batch `
    device=0 `
    workers=$Workers `
    project="D:\chim\runs" `
    name=$RunName `
    exist_ok=True

yolo task=segment mode=val `
    model="D:\chim\runs\$RunName\weights\best.pt" `
    data="D:\chim\data\dronebird_seg_clean_v2\data.yaml" `
    split=test `
    imgsz=$Imgsz `
    batch=$Batch `
    device=0 `
    workers=$Workers `
    project="D:\chim\runs" `
    name="${RunName}_test" `
    exist_ok=True