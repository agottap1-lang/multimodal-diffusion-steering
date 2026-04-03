"""Quick checkpoint integrity check for demos_combined collection."""
import os, pickle, time, numpy as np
from pathlib import Path

ckpt = Path("data/demos/demos_combined_ckpt.pkl")
npz  = Path("data/demos/demos_combined.npz")

print("=" * 60)
print("  Demo Collection Monitor")
print("=" * 60)
print(f"  Time now : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ── final .npz already exists? ─────────────────────────────────
if npz.exists():
    d = np.load(str(npz), allow_pickle=True)
    n = len(d["obs"])
    sty = d["style_labels"]
    print(f"\n  COLLECTION COMPLETE!")
    print(f"  demos_combined.npz : {n} demos")
    print(f"    legible   : {(sty==0).sum()}")
    print(f"    neutral   : {(sty==1).sum()}")
    print(f"    deceptive : {(sty==2).sum()}")
    print(f"  obs shape  : {d['obs'].shape}")
    print(f"  act shape  : {d['actions'].shape}")
    print("=" * 60)
    raise SystemExit(0)

# ── checkpoint ─────────────────────────────────────────────────
if not ckpt.exists():
    print("\n  No checkpoint yet — collection may just have started.")
    raise SystemExit(0)

mtime = time.ctime(ckpt.stat().st_mtime)
age_s = time.time() - ckpt.stat().st_mtime

with open(ckpt, "rb") as f:
    d = pickle.load(f)

n   = len(d["all_obs"])
obs = d["all_obs"]
act = d["all_act"]
sty = d["all_styles"]
leg = sum(1 for s in sty if s == 0)
neu = sum(1 for s in sty if s == 1)
dec = sum(1 for s in sty if s == 2)

print(f"\n  Progress   : {n} / 400  ({n/4:.1f}%)")
print(f"  Checkpoint : last saved {mtime}  ({age_s/60:.1f} min ago)")
print(f"  Styles     : legible={leg}  neutral={neu}  deceptive={dec}")
print(f"  obs shape  : {np.array(obs[0]).shape}  (should be (T, 22))")
print(f"  act shape  : {np.array(act[0]).shape}  (should be (T, 5))")

# ── sanity checks ──────────────────────────────────────────────
issues = []
for i, (o, a) in enumerate(zip(obs, act)):
    o, a = np.array(o), np.array(a)
    if o.shape[-1] != 22:
        issues.append(f"demo {i}: obs dim={o.shape[-1]} (expected 22)")
    if a.shape[-1] != 5:
        issues.append(f"demo {i}: act dim={a.shape[-1]} (expected 5)")
    if np.isnan(o).any():
        issues.append(f"demo {i}: NaN in obs")
    if np.isnan(a).any():
        issues.append(f"demo {i}: NaN in actions")

if issues:
    print(f"\n  !! {len(issues)} ISSUES FOUND !!")
    for iss in issues[:10]:
        print(f"     {iss}")
else:
    print(f"\n  Data integrity : OK (no NaNs, correct dims in all {n} demos)")

# ── stale check ────────────────────────────────────────────────
if age_s > 40 * 60:   # last checkpoint > 40 min ago
    print(f"\n  WARNING: checkpoint is {age_s/60:.0f} min old — process may have stalled!")
else:
    print(f"  Staleness     : OK (checkpoint fresh)")

# ── video count ────────────────────────────────────────────────
vid_dir = Path("data/demos/demo_videos_combined")
if vid_dir.exists():
    vids = list(vid_dir.glob("*.mp4"))
    print(f"  Videos on disk: {len(vids)}")

# ── ETA ────────────────────────────────────────────────────────
rate = 59.0   # s/demo conservative
remaining = 400 - n
eta_min = remaining * rate / 60
import datetime
eta_time = datetime.datetime.now() + datetime.timedelta(minutes=eta_min)
print(f"\n  Remaining  : {remaining} demos  ~{eta_min/60:.1f}h")
print(f"  ETA        : {eta_time.strftime('%H:%M')} (at {rate}s/demo)")
print("=" * 60)
