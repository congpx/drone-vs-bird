Param(
  [int]$Epochs = 80,
  [int]$Imgsz = 640,
  [int]$Batch = 16,
  [string]$Device = "0",
  [int]$Workers = 4
)
$ErrorActionPreference = "Stop"
$BASE_DIR = Split-Path -Parent $PSScriptRoot

$condaHook = & conda shell.powershell hook
Invoke-Expression $condaHook
conda activate topic3_yoloseg

New-Item -ItemType Directory -Force -Path (Join-Path $BASE_DIR "runs\topic3") | Out-Null
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "10_make_data_yaml.ps1")

python (Join-Path $PSScriptRoot "tools\convert_seg_to_det.py") `
  --src (Join-Path $BASE_DIR "data\dronebird_seg") `
  --dst (Join-Path $BASE_DIR "data\dronebird_det")

Set-Location $BASE_DIR

yolo task=detect mode=train `
  model=yolo11n.pt `
  data=(Join-Path $PSScriptRoot "cfg\dronebird_det.yaml") `
  epochs=$Epochs imgsz=$Imgsz batch=$Batch device=$Device workers=$Workers pretrained=True cache=False `
  project=(Join-Path $BASE_DIR "runs\topic3") name="det_baseline" exist_ok=True
