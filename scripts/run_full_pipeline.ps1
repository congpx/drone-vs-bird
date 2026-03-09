param(
    [int]$Epochs = 40,
    [int]$Batch = 12,
    [int]$Workers = 4,
    [int]$Imgsz = 640
)

$ErrorActionPreference = "Stop"

Write-Host "=== PREPARE DATA ===" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\00_prepare_data.ps1

Write-Host "=== TRAIN DETECT BASELINE ===" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\10_train_baseline_det.ps1 `
    -Epochs $Epochs -Batch $Batch -Workers $Workers -Imgsz $Imgsz -RunName "baseline_det_rgb"

Write-Host "=== TRAIN SEG BASELINE ===" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\11_train_baseline_seg.ps1 `
    -Epochs $Epochs -Batch $Batch -Workers $Workers -Imgsz $Imgsz -RunName "baseline_seg_rgb"

Write-Host "=== RUN PROPOSED FILTER ===" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\20_eval_proposed_false_alarm_filter.ps1 `
    -Model "D:\chim\runs\baseline_seg_rgb\weights\best.pt" `
    -Data "D:\chim\data\dronebird_seg_clean_v2\data.yaml" `
    -OutDir "D:\chim\runs\proposed_false_alarm_filter" `
    -Imgsz $Imgsz `
    -Conf 0.25 `
    -IoUNms 0.50 `
    -IoUMatch 0.50 `
    -Device "0"

Write-Host "[DONE] Full pipeline finished." -ForegroundColor Green