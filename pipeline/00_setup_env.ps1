Param(
  [string]$EnvName = "topic3_yoloseg",
  [string]$PyVer = "3.11"
)
$ErrorActionPreference = "Stop"

$BASE_DIR = Split-Path -Parent $PSScriptRoot
$REQ_FILE = Join-Path $PSScriptRoot "requirements.txt"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  throw "Không tìm thấy conda trong PATH. Hãy cài Miniconda/Anaconda trước."
}

$condaHook = & conda shell.powershell hook
Invoke-Expression $condaHook

$envExists = conda env list | Select-String -Pattern "^$EnvName\s"
if (-not $envExists) {
  conda create -y -n $EnvName python=$PyVer
}

conda activate $EnvName
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r $REQ_FILE

python -c "import torch; print('[INFO] torch =', torch.__version__); print('[INFO] cuda available =', torch.cuda.is_available()); print('[INFO] device count =', torch.cuda.device_count()); print('[INFO] gpu name =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
