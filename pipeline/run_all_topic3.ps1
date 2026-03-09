Param(
  [int]$Epochs = 80,
  [int]$Imgsz = 640,
  [int]$Batch = 16,
  [string]$Device = "0",
  [int]$Workers = 4
)
$ErrorActionPreference = "Stop"
$BASE_DIR = Split-Path -Parent $PSScriptRoot

& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "20_train_baseline_det.ps1") -Epochs $Epochs -Imgsz $Imgsz -Batch $Batch -Device $Device -Workers $Workers
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "21_train_seg.ps1") -Epochs $Epochs -Imgsz $Imgsz -Batch $Batch -Device $Device -Workers $Workers
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "30_eval_seg_raw_and_filtered.ps1") -Imgsz $Imgsz -Conf 0.25 -Iou 0.5 -Device $Device

$condaHook = & conda shell.powershell hook
Invoke-Expression $condaHook
conda activate topic3_yoloseg
python (Join-Path $PSScriptRoot "tools\collect_final_summary.py")
Write-Host "[DONE] Toàn bộ pipeline Đề tài 3 đã chạy xong."
Write-Host "[INFO] Xem kết quả tại: $(Join-Path $BASE_DIR 'runs\topic3')"
