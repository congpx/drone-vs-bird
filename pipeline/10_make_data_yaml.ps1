$ErrorActionPreference = "Stop"
$BASE_DIR = Split-Path -Parent $PSScriptRoot
$CFG_DIR = Join-Path $PSScriptRoot "cfg"
New-Item -ItemType Directory -Force -Path $CFG_DIR | Out-Null

$detPath = (Join-Path $BASE_DIR "data\dronebird_det").Replace("\", "/")
$segPath = (Join-Path $BASE_DIR "data\dronebird_seg").Replace("\", "/")

@"
path: $detPath
train: images/train
val: images/val
test: images/test
names:
  0: drone
  1: bird
"@ | Set-Content -Path (Join-Path $CFG_DIR "dronebird_det.yaml") -Encoding UTF8

@"
path: $segPath
train: images/train
val: images/val
test: images/test
names:
  0: drone
  1: bird
"@ | Set-Content -Path (Join-Path $CFG_DIR "dronebird_seg.yaml") -Encoding UTF8

Write-Host "[OK] Đã tạo YAML trong $CFG_DIR"
