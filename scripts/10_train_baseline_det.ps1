param(
    [int]$Epochs = 40,
    [int]$Batch = 12,
    [int]$Workers = 4,
    [int]$Imgsz = 640,
    [string]$RunName = "baseline_det_rgb"
)

$ErrorActionPreference = "Stop"
conda activate chim-yolo

yolo task=detect mode=train `
    model=yolo11n.pt `
    data="D:\chim\data\dronebird_det_clean\data.yaml" `
    epochs=$Epochs `
    imgsz=$Imgsz `
    batch=$Batch `
    device=0 `
    workers=$Workers `
    project="D:\chim\runs" `
    name=$RunName `
    exist_ok=True

yolo task=detect mode=val `
    model="D:\chim\runs\$RunName\weights\best.pt" `
    data="D:\chim\data\dronebird_det_clean\data.yaml" `
    split=test `
    imgsz=$Imgsz `
    batch=$Batch `
    device=0 `
    workers=$Workers `
    project="D:\chim\runs" `
    name="${RunName}_test" `
    exist_ok=True