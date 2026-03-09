param(
    [int]$Imgsz = 640,
    [double]$Conf = 0.25,
    [double]$IoUNms = 0.50,
    [double]$IoUMatch = 0.50,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"

Write-Host "=== EVAL PROPOSED: YOLO-SEG + SHAPE FILTER ===" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\20_eval_proposed_shape_filter.ps1 `
    -Model "D:\chim\runs\baseline_seg_rgb\weights\best.pt" `
    -Data "D:\chim\data\dronebird_seg_clean\data.yaml" `
    -OutDir "D:\chim\runs\proposed_seg_shape_filter" `
    -Imgsz $Imgsz `
    -Conf $Conf `
    -IoUNms $IoUNms `
    -IoUMatch $IoUMatch `
    -Device $Device

Write-Host "[DONE] Proposed evaluation finished." -ForegroundColor Green