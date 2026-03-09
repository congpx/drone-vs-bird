$ErrorActionPreference = "Stop"
conda activate chim-yolo

python D:\chim\tools\convert_raw_to_yolodet.py
python D:\chim\tools\convert_mixed_to_yoloseg_v2.py

Write-Host "[DONE] Data prepared." -ForegroundColor Green