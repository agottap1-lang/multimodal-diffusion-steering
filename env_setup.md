# Environment Setup

Follow these steps on **Windows PowerShell**. The same steps work on Linux
(replace `.venv\Scripts\activate` with `source .venv/bin/activate`).

## 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## 2. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Verify installations

```powershell
python -c "import pybullet; import pybullet_data; import torch; print('PyBullet OK, Torch', torch.__version__)"
```

## 4. (Optional) Install ffmpeg for higher-quality MP4 videos

- Download from <https://ffmpeg.org/download.html> and add `ffmpeg\bin` to PATH.
- `imageio-ffmpeg` ships a bundled binary that works without this step.

## 5. Run the full pipeline

```powershell
.\run_all.ps1
```

Or step by step:

```powershell
# Collect demonstrations (100 left + 100 right)
python scripts/collect_demos_twoblockpick.py --episodes_left 100 --episodes_right 100

# Train diffusion policy
python scripts/train_diffusion_policy.py --config configs/train.yaml

# Evaluate multimodality
python scripts/eval_multimodality.py --ckpt runs/latest/ckpt.pt --config configs/train.yaml
```
