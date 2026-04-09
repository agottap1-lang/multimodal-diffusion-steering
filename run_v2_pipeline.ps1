# ===================================================================
#  V2 Pipeline: Collect → Train → Eval
#  Usage:  .\run_v2_pipeline.ps1
# ===================================================================
$ErrorActionPreference = "Stop"

$PYTHON = ".venv\Scripts\python.exe"
$DEMO_PATH = "data\demos\demos_v2.npz"

# ── Step 1: Check if demos exist (collection should already be running) ──
if (-not (Test-Path $DEMO_PATH)) {
    Write-Host "[1/3] Demos not found — starting collection..." -ForegroundColor Yellow
    & $PYTHON scripts/collect_demos_v2.py --collect --seed 0 --out $DEMO_PATH
    if ($LASTEXITCODE -ne 0) { throw "Demo collection failed!" }
} else {
    Write-Host "[1/3] Demos found at $DEMO_PATH — skipping collection." -ForegroundColor Green
}

# ── Step 2: Train CFG policy (200 epochs) ──
Write-Host "`n[2/3] Training CFG diffusion policy..." -ForegroundColor Cyan
& $PYTHON scripts/train_cfg.py `
    --demo_path $DEMO_PATH `
    --epochs 200 `
    --batch_size 256 `
    --lr 1e-4 `
    --hidden_dim 256 `
    --n_blocks 6 `
    --horizon 32 `
    --n_diffusion_steps 100 `
    --cfg_dropout 0.15

if ($LASTEXITCODE -ne 0) { throw "Training failed!" }

# ── Step 3: Find latest checkpoint and evaluate ──
$LATEST_RUN = Get-ChildItem -Path "runs" -Directory | Where-Object { $_.Name -like "cfg_*" } | Sort-Object Name -Descending | Select-Object -First 1
if (-not $LATEST_RUN) { throw "No CFG run found!" }

$CKPT = Join-Path $LATEST_RUN.FullName "ckpt_ep200.pt"
if (-not (Test-Path $CKPT)) {
    $CKPT = Join-Path $LATEST_RUN.FullName "ckpt_best.pt"
}
Write-Host "`n[3/3] Evaluating: $CKPT" -ForegroundColor Cyan

& $PYTHON evaluation/eval_cfg.py `
    --checkpoint $CKPT `
    --behavior all `
    --n_episodes 20 `
    --cfg_lambda_leg 2.0 `
    --cfg_lambda_pred -2.0 `
    --eta 0.5

if ($LASTEXITCODE -ne 0) { throw "Evaluation failed!" }

Write-Host "`n=== V2 PIPELINE COMPLETE ===" -ForegroundColor Green
