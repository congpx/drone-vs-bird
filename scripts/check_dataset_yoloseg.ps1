$DATASET_ROOT = "D:\chim\data\dronebird_seg_raw"

$imgExts = @(".jpg", ".jpeg", ".png", ".bmp", ".webp")
$splits = @("train", "valid", "test")

function Test-YoloSegLine {
    param([string]$line)

    $line = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($line)) { return $false }

    $parts = $line -split '\s+'
    if ($parts.Count -lt 7) { return $false }   # class + ít nhất 3 điểm (x y x y x y)

    if ($parts[0] -notmatch '^\d+$') { return $false }

    for ($i = 1; $i -lt $parts.Count; $i++) {
        if ($parts[$i] -notmatch '^[-+]?\d*\.?\d+$') { return $false }
        $v = [double]$parts[$i]
        if ($v -lt 0 -or $v -gt 1) { return $false }   # YOLO polygon normalized
    }

    if ((($parts.Count - 1) % 2) -ne 0) { return $false }  # số tọa độ phải chẵn
    return $true
}

Write-Host "=== KIEM TRA DATASET YOLO-SEG ===" -ForegroundColor Cyan
Write-Host "Root: $DATASET_ROOT"
Write-Host ""

if (!(Test-Path $DATASET_ROOT)) {
    Write-Host "KHONG TIM THAY DATASET_ROOT" -ForegroundColor Red
    exit 1
}

$yamlCandidates = Get-ChildItem -Path $DATASET_ROOT -Filter *.yaml -File -ErrorAction SilentlyContinue
if ($yamlCandidates.Count -gt 0) {
    Write-Host "Tim thay YAML:" -ForegroundColor Green
    $yamlCandidates | ForEach-Object { Write-Host " - $($_.FullName)" }
    Write-Host ""
} else {
    Write-Host "Khong tim thay file .yaml trong root dataset." -ForegroundColor Yellow
    Write-Host ""
}

$globalErrors = @()
$globalWarnings = @()

foreach ($split in $splits) {
    $imgDir = Join-Path $DATASET_ROOT "$split\images"
    $lblDir = Join-Path $DATASET_ROOT "$split\labels"

    Write-Host "----- SPLIT: $split -----" -ForegroundColor Cyan

    if (!(Test-Path $imgDir)) {
        $globalErrors += "Thieu thu muc images cho split $split : $imgDir"
        Write-Host "Thieu: $imgDir" -ForegroundColor Red
        continue
    }

    if (!(Test-Path $lblDir)) {
        $globalErrors += "Thieu thu muc labels cho split $split : $lblDir"
        Write-Host "Thieu: $lblDir" -ForegroundColor Red
        continue
    }

    $images = Get-ChildItem -Path $imgDir -File | Where-Object { $imgExts -contains $_.Extension.ToLower() }
    $labels = Get-ChildItem -Path $lblDir -File -Filter *.txt

    Write-Host ("So anh   : {0}" -f $images.Count)
    Write-Host ("So label : {0}" -f $labels.Count)

    $imageBaseMap = @{}
    foreach ($img in $images) {
        $imageBaseMap[$img.BaseName] = $img.FullName
    }

    $labelBaseMap = @{}
    foreach ($lbl in $labels) {
        $labelBaseMap[$lbl.BaseName] = $lbl.FullName
    }

    $missingLabels = @()
    foreach ($img in $images) {
        if (-not $labelBaseMap.ContainsKey($img.BaseName)) {
            $missingLabels += $img.FullName
        }
    }

    $orphanLabels = @()
    foreach ($lbl in $labels) {
        if (-not $imageBaseMap.ContainsKey($lbl.BaseName)) {
            $orphanLabels += $lbl.FullName
        }
    }

    $emptyLabels = @()
    $badFormatLabels = @()

    foreach ($lbl in $labels) {
        $content = Get-Content $lbl.FullName -ErrorAction SilentlyContinue
        if ($null -eq $content -or $content.Count -eq 0) {
            $emptyLabels += $lbl.FullName
            continue
        }

        $badLineFound = $false
        foreach ($line in $content) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            if (-not (Test-YoloSegLine $line)) {
                $badLineFound = $true
                break
            }
        }
        if ($badLineFound) {
            $badFormatLabels += $lbl.FullName
        }
    }

    if ($missingLabels.Count -gt 0) {
        Write-Host ("Anh thieu label: {0}" -f $missingLabels.Count) -ForegroundColor Yellow
        $missingLabels | Select-Object -First 10 | ForEach-Object { Write-Host "  MISSING_LABEL: $_" }
        if ($missingLabels.Count -gt 10) { Write-Host "  ... va them $($missingLabels.Count - 10) file" }
    } else {
        Write-Host "Anh thieu label: 0" -ForegroundColor Green
    }

    if ($orphanLabels.Count -gt 0) {
        Write-Host ("Label mo coi   : {0}" -f $orphanLabels.Count) -ForegroundColor Yellow
        $orphanLabels | Select-Object -First 10 | ForEach-Object { Write-Host "  ORPHAN_LABEL: $_" }
        if ($orphanLabels.Count -gt 10) { Write-Host "  ... va them $($orphanLabels.Count - 10) file" }
    } else {
        Write-Host "Label mo coi   : 0" -ForegroundColor Green
    }

    if ($emptyLabels.Count -gt 0) {
        Write-Host ("Label rong     : {0}" -f $emptyLabels.Count) -ForegroundColor Yellow
        $emptyLabels | Select-Object -First 10 | ForEach-Object { Write-Host "  EMPTY_LABEL: $_" }
        if ($emptyLabels.Count -gt 10) { Write-Host "  ... va them $($emptyLabels.Count - 10) file" }
    } else {
        Write-Host "Label rong     : 0" -ForegroundColor Green
    }

    if ($badFormatLabels.Count -gt 0) {
        Write-Host ("Label sai format YOLO-seg: {0}" -f $badFormatLabels.Count) -ForegroundColor Red
        $badFormatLabels | Select-Object -First 10 | ForEach-Object { Write-Host "  BAD_FORMAT: $_" }
        if ($badFormatLabels.Count -gt 10) { Write-Host "  ... va them $($badFormatLabels.Count - 10) file" }
    } else {
        Write-Host "Label sai format YOLO-seg: 0" -ForegroundColor Green
    }

    if ($images.Count -eq 0) {
        $globalErrors += "Split $split khong co anh."
    }
    if ($labels.Count -eq 0) {
        $globalErrors += "Split $split khong co label."
    }
    if ($missingLabels.Count -gt 0) {
        $globalWarnings += "Split $split co $($missingLabels.Count) anh thieu label."
    }
    if ($orphanLabels.Count -gt 0) {
        $globalWarnings += "Split $split co $($orphanLabels.Count) label mo coi."
    }
    if ($emptyLabels.Count -gt 0) {
        $globalWarnings += "Split $split co $($emptyLabels.Count) label rong."
    }
    if ($badFormatLabels.Count -gt 0) {
        $globalErrors += "Split $split co $($badFormatLabels.Count) label sai format YOLO-seg."
    }

    Write-Host ""
}

Write-Host "===== TONG KET =====" -ForegroundColor Cyan

if ($globalWarnings.Count -gt 0) {
    Write-Host "Canh bao:" -ForegroundColor Yellow
    $globalWarnings | ForEach-Object { Write-Host " - $_" }
    Write-Host ""
}

if ($globalErrors.Count -gt 0) {
    Write-Host "Loi:" -ForegroundColor Red
    $globalErrors | ForEach-Object { Write-Host " - $_" }
    Write-Host ""
    Write-Host "KET LUAN: Dataset CHUA san sang de train YOLO-seg." -ForegroundColor Red
} else {
    Write-Host "KET LUAN: Dataset DAT yeu cau co ban va SAN SANG de train YOLO-seg." -ForegroundColor Green
}