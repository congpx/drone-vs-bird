param(
    [int]$Epochs = 40,
    [int]$Batch = 12,
    [int]$Workers = 4,
    [int]$Imgsz = 640
)

$ErrorActionPreference = "Stop"

Write-Host "=== PREPARE DATA ===" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\00_prepare_data.ps1

Write-Host "=== TRAIN BASELINE DET ===" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\10_train_baseline_det.ps1 `
    -Epochs $Epochs -Batch $Batch -Workers $Workers -Imgsz $Imgsz -RunName "baseline_det_rgb"

Write-Host "=== TRAIN BASELINE SEG ===" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File D:\chim\scripts\11_train_baseline_seg.ps1 `
    -Epochs $Epochs -Batch $Batch -Workers $Workers -Imgsz $Imgsz -RunName "baseline_seg_rgb"

Write-Host "[DONE] Baselines finished." -ForegroundColor Green