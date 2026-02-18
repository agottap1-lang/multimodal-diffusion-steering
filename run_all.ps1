# run_all.ps1 – collect → train → evaluate (Windows PowerShell)
$ErrorActionPreference = "Stop"

# Activate venv if not already active
if (-not $env:VIRTUAL_ENV) {
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        . .\.venv\Scripts\Activate.ps1
    } else {
        Write-Host "ERROR: .venv not found. Run: py -m venv .venv" -ForegroundColor Red
        exit 1
    }
}

Write-Host "===== Step 1/3: Collect demonstrations =====" -ForegroundColor Cyan
python scripts/collect_demos_twoblockpick.py `
    --episodes_left 100 --episodes_right 100 `
    --seed 0 --out data/demos/demos.npz

Write-Host ""
Write-Host "===== Step 2/3: Train diffusion policy =====" -ForegroundColor Cyan
python scripts/train_diffusion_policy.py --config configs/train.yaml

Write-Host ""
Write-Host "===== Step 3/3: Evaluate multimodality =====" -ForegroundColor Cyan
python scripts/eval_multimodality.py `
    --ckpt runs/latest/ckpt.pt `
    --K 10 --M 10 --n_videos 10 `
    --video_dir analysis/videos

Write-Host ""
Write-Host "===== Done! =====" -ForegroundColor Green
Write-Host "  Metrics:  analysis/metrics.json"
Write-Host "  CSV:      analysis/results.csv"
Write-Host "  Plot:     analysis/multimodality_bar.png"
Write-Host "  Videos:   analysis/videos/"
