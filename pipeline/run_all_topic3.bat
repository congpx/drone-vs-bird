@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_all_topic3.ps1" -Epochs 40 -Imgsz 640 -Batch 8 -Device 0 -Workers 2
endlocal
