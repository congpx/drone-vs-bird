@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp000_setup_env.ps1"
powershell -ExecutionPolicy Bypass -File "%~dp0run_all_topic3.ps1" -Epochs 5 -Imgsz 640 -Batch 4 -Device 0 -Workers 2
endlocal
