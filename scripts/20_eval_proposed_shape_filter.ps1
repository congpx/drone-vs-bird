param(
    [string]$Model = "D:\chim\runs\baseline_seg_rgb\weights\best.pt",
    [string]$Data  = "D:\chim\data\dronebird_seg_clean\data.yaml",
    [string]$OutDir = "D:\chim\runs\proposed_seg_shape_filter",
    [int]$Imgsz = 640,
    [double]$Conf = 0.25,
    [double]$IoUNms = 0.50,
    [double]$IoUMatch = 0.50,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
conda activate chim-yolo

python D:\chim\tools\eval_shape_filter.py `
    --model "$Model" `
    --data "$Data" `
    --split test `
    --outdir "$OutDir" `
    --imgsz $Imgsz `
    --conf $Conf `
    --iou-nms $IoUNms `
    --iou-match $IoUMatch `
    --device $Device `
    --drone-class 0