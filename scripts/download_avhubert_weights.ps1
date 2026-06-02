param(
    [string]$Url = ""
)

# Script to download AVHubert checkpoint files referenced by the detectors.
# Usage examples:
# 1) Provide a direct URL:
#    pwsh .\scripts\download_avhubert_weights.ps1 -Url "https://.../base_lrs3_iter4.pt"
# 2) Edit the $defaultUrl variable below and run the script without parameters.


$repoRoot = (Split-Path -Parent $PSScriptRoot)

# Paths to place files
$baseRel = "detectors\auvire\src\avhubert\base_lrs3_iter4.pt"
$miscDirRel = "detectors\auvire\src\avhubert\misc"
$meanFaceRel = "$miscDirRel\20words_mean_face.npy"
$shapeRel = "$miscDirRel\shape_predictor_68_face_landmarks.dat.bz2"

$baseTarget = Join-Path $repoRoot $baseRel
$meanFaceTarget = Join-Path $repoRoot $meanFaceRel
$shapeTarget = Join-Path $repoRoot $shapeRel

# Common public URLs (official or widely used mirrors)
$defaults = @{
    base = 'https://dl.fbaipublicfiles.com/av_hubert/base_lrs3_iter4.pt'
    meanface = 'https://dl.fbaipublicfiles.com/av_hubert/misc/20words_mean_face.npy'
    shape = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
}

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
}

Ensure-Dir (Join-Path $repoRoot (Split-Path -Parent $baseRel))
Ensure-Dir (Join-Path $repoRoot $miscDirRel)

if (-not [string]::IsNullOrWhiteSpace($Url)) {
    # single-file direct download (assume checkpoint)
    $downloadList = @([pscustomobject]@{url=$Url; dest=$baseTarget})
} else {
    # use defaults
    $downloadList = @(
        [pscustomobject]@{url=$defaults.base; dest=$baseTarget},
        [pscustomobject]@{url=$defaults.meanface; dest=$meanFaceTarget},
        [pscustomobject]@{url=$defaults.shape; dest=$shapeTarget}
    )
}

foreach ($item in $downloadList) {
    $url = $item.url
    $dest = $item.dest

    if (Test-Path $dest) {
        Write-Host "Already exists: $dest"
        continue
    }

    Write-Host "Downloading $url → $dest"
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -Verbose -ErrorAction Stop
        Write-Host "Downloaded: $dest"
    } catch {
        Write-Warning "Failed to download $url : $_"
        if ($url -match 'drive.google.com') { Write-Host 'Google Drive links may need special handling (gdown).' }
    }
}

Write-Host "Download attempts finished."

# If Python checker exists, try to validate the checkpoint can be loaded.
$checker = Join-Path $repoRoot "scripts\check_load_checkpoint.py"
if (Test-Path $checker) {
    Write-Host "Running Python load-check on downloaded checkpoint..."
    $python = "python"
    $proc = Start-Process -FilePath $python -ArgumentList "`"$checker`"", "`"$baseTarget`"" -NoNewWindow -Wait -PassThru -ErrorAction SilentlyContinue
    if ($proc.ExitCode -eq 0) {
        Write-Host "Checkpoint load check passed."
    } else {
        Write-Host "Checkpoint load check failed (exit code $($proc.ExitCode)). Check Python output above." -ForegroundColor Yellow
    }
} else {
    Write-Host "No checkpoint checker found at $checker; skipping load test." -ForegroundColor Gray
}
