$DATASET_ROOT = "D:\chim\data\dronebird_seg_raw"
$imgExts = @(".jpg", ".jpeg", ".png", ".bmp", ".webp")
$splits = @("train", "valid", "test")

function Get-LabelLineType {
    param([string]$line)

    $line = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($line)) { return "empty" }

    $parts = $line -split '\s+'
    if ($parts.Count -lt 5) { return "bad" }

    if ($parts[0] -notmatch '^\d+$') { return "bad" }

    for ($i = 1; $i -lt $parts.Count; $i++) {
        if ($parts[$i] -notmatch '^[-+]?\d*\.?\d+$') { return "bad" }
        $v = [double]$parts[$i]
        if ($v -lt 0 -or $v -gt 1) { return "bad" }
    }

    if ($parts.Count -eq 5) {
        return "detect"
    }

    $coordCount = $parts.Count - 1
    if (($coordCount % 2) -eq 0 -and $parts.Count -ge 7) {
        return "seg"
    }

    return "unknown"
}

Write-Host "=== AUTO CHECK DATASET FORMAT ===" -ForegroundColor Cyan
Write-Host "Root: $DATASET_ROOT"
Write-Host ""

foreach ($split in $splits) {
    $imgDir = Join-Path $DATASET_ROOT "$split\images"
    $lblDir = Join-Path $DATASET_ROOT "$split\labels"

    Write-Host "----- SPLIT: $split -----" -ForegroundColor Cyan

    $images = Get-ChildItem -Path $imgDir -File | Where-Object { $imgExts -contains $_.Extension.ToLower() }
    $labels = Get-ChildItem -Path $lblDir -File -Filter *.txt

    Write-Host "So anh   : $($images.Count)"
    Write-Host "So label : $($labels.Count)"

    $detectCount = 0
    $segCount = 0
    $emptyCount = 0
    $badCount = 0
    $unknownCount = 0

    $badFiles = @()

    foreach ($lbl in $labels) {
        $content = Get-Content $lbl.FullName -ErrorAction SilentlyContinue

        if ($null -eq $content -or $content.Count -eq 0) {
            $emptyCount++
            continue
        }

        $fileTypes = @()
        foreach ($line in $content) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $fileTypes += (Get-LabelLineType $line)
        }

        if ($fileTypes.Count -eq 0) {
            $emptyCount++
            continue
        }

        $uniq = $fileTypes | Select-Object -Unique

        if ($uniq.Count -eq 1) {
            switch ($uniq[0]) {
                "detect" { $detectCount++ }
                "seg"    { $segCount++ }
                "empty"  { $emptyCount++ }
                "bad"    { $badCount++; $badFiles += $lbl.FullName }
                default  { $unknownCount++; $badFiles += $lbl.FullName }
            }
        } else {
            $unknownCount++
            $badFiles += $lbl.FullName
        }
    }

    Write-Host "Label detect : $detectCount" -ForegroundColor Yellow
    Write-Host "Label seg    : $segCount" -ForegroundColor Green
    Write-Host "Label rong   : $emptyCount" -ForegroundColor Yellow
    Write-Host "Label loi    : $badCount" -ForegroundColor Red
    Write-Host "Label mixed/unknown: $unknownCount" -ForegroundColor Yellow

    if ($badFiles.Count -gt 0) {
        Write-Host "Vi du file can xem tay:" -ForegroundColor Yellow
        $badFiles | Select-Object -First 10 | ForEach-Object { Write-Host " - $_" }
    }

    Write-Host ""
}