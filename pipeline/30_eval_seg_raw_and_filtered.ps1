Param(
  [int]$Imgsz = 640,
  [double]$Conf = 0.25,
  [double]$Iou = 0.5,
  [string]$Device = "0"
)
$ErrorActionPreference = "Stop"
$BASE_DIR = Split-Path -Parent $PSScriptRoot

$condaHook = & conda shell.powershell hook
Invoke-Expression $condaHook
conda activate topic3_yoloseg

$MODEL = Join-Path $BASE_DIR "runs\topic3\seg_baseline\weights\best.pt"
$TEST_IMAGES = Join-Path $BASE_DIR "data\dronebird_seg\images\test"
$TEST_LABELS = Join-Path $BASE_DIR "data\dronebird_seg\labels\test"
$OUTDIR = Join-Path $BASE_DIR "runs\topic3\seg_shape_eval"

python (Join-Path $PSScriptRoot "tools\eval_seg_shape_filter.py") `
  --model $MODEL `
  --images $TEST_IMAGES `
  --labels $TEST_LABELS `
  --outdir $OUTDIR `
  --imgsz $Imgsz `
  --conf $Conf `
  --iou $Iou `
  --device $Device
