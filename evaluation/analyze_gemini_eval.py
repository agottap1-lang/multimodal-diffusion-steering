"""
Analyze gemini_vlm_eval canonical prefix_frames results.
40 videos (cfg00): dec_left/right (5+5), leg_left/right (10+10), neu_left/right (5+5)
Each video evaluated at t=0..k with growing prefix window.

CORRECTED goal_gt: side=left→A, side=right→B
  ("pick the left block"=image-left; k=6 data confirms left-side videos → gripper→image-left)

Usage:
  python analyze_gemini_eval.py          # analyze k=3 results
  python analyze_gemini_eval.py k6       # analyze k=6 results
"""

import json, sys
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path(r"c:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick")
RESULTS_K3 = BASE / "outputs/combined_prefix_gemini_eval/results.jsonl"
RESULTS_K6 = BASE / "outputs/combined_prefix_gemini_eval_k6/results.jsonl"

mode = sys.argv[1] if len(sys.argv) > 1 else "k3"
RESULTS = RESULTS_K6 if mode == "k6" else RESULTS_K3

records = [json.loads(l) for l in RESULTS.read_text().strip().splitlines()]
K = max(r["t_sec"] for r in records)
print(f"=== {mode.upper()}: {RESULTS.name}  ({len(records)} records, max_t={K}s) ===")

# CORRECTED ground truth (manifest had side=right→A, side=left→B which was wrong)
def gt(video_id):
    return "A" if video_id.split("_")[2] == "left" else "B"

def side(video_id):
    return video_id.split("_")[2]   # left / right

for r in records:
    r["_gt"]       = gt(r["video_id"])
    r["_side"]     = side(r["video_id"])
    r["_correct"]  = (r["choice"] == r["_gt"])
    r["_decisive"] = (r["choice"] != "C")

# ─────────────────────────────────────────
# 1. Overall choice distribution
# ─────────────────────────────────────────
print("\n── 1. Overall choice distribution ──")
dist = Counter(r["choice"] for r in records)
for ch in sorted(dist): print(f"  {ch}: {dist[ch]:4d} ({100*dist[ch]/len(records):.1f}%)")

# ─────────────────────────────────────────
# 2. Accuracy at FINAL timestep (t=K) per traj_type x side
# ─────────────────────────────────────────
tK = [r for r in records if r["t_sec"] == K]
print(f"\n── 2. At t={K}s (max context, {len(tK)} entries): accuracy ──")
print(f"{'Type':<12} {'Side':<6} {'N':>4} {'C%':>6} {'DecN':>5} {'Dec_Acc':>8} {'MeanpA':>8}")
print("─" * 54)
by_ts = defaultdict(list)
for r in tK: by_ts[(r["traj_type"], r["_side"])].append(r)

for ty in ["deceptive", "neutral", "legible"]:
    for sd in ["left", "right"]:
        recs = by_ts.get((ty, sd), [])
        if not recs: continue
        c_pct  = 100 * sum(1 for r in recs if r["choice"] == "C") / len(recs)
        dec    = [r for r in recs if r["_decisive"]]
        d_acc  = 100 * sum(1 for r in dec if r["_correct"]) / len(dec) if dec else float("nan")
        mpA    = sum(r["pA"] for r in recs) / len(recs)
        print(f"{ty:<12} {sd:<6} {len(recs):>4} {c_pct:>5.0f}% {len(dec):>5} {d_acc:>7.0f}% {mpA:>8.3f}")

# ─────────────────────────────────────────
# 3. pA trajectory over time by traj_type x side
# ─────────────────────────────────────────
print(f"\n── 3. Mean pA over time ──")
timesteps = sorted(set(r["t_sec"] for r in records))
header = "".join(f"  t={t}" for t in timesteps)
print(f"{'Type':<12} {'Side':<6}{header}")
print("─" * (18 + 5*len(timesteps)))

by_ts_t = defaultdict(lambda: defaultdict(list))
for r in records:
    by_ts_t[(r["traj_type"], r["_side"])][r["t_sec"]].append(r["pA"])

for ty in ["deceptive", "neutral", "legible"]:
    for sd in ["left", "right"]:
        ts_data = by_ts_t.get((ty, sd), {})
        vals = "".join(
            f"  {(sum(ts_data[t])/len(ts_data[t])):>4.2f}" if ts_data.get(t) else "   N/A"
            for t in timesteps
        )
        print(f"{ty:<12} {sd:<6}{vals}")

# ─────────────────────────────────────────
# 4. Bias check
# ─────────────────────────────────────────
left_K  = [r for r in tK if r["_side"] == "left"]
right_K = [r for r in tK if r["_side"] == "right"]
left_pA  = sum(r["pA"] for r in left_K)  / len(left_K)
right_pA = sum(r["pA"] for r in right_K) / len(right_K)
print(f"\n── 4. Bias check at t={K}s ──")
print(f"  Left  mean pA = {left_pA:.3f}  (unbiased expect > 0.5 = going to A)")
print(f"  Right mean pA = {right_pA:.3f}  (unbiased expect < 0.5 = going to B)")
print(f"  Δ(left-right) = {left_pA - right_pA:.3f}  (large positive = BIASED toward A)")

# ─────────────────────────────────────────
# 5. All non-C responses
# ─────────────────────────────────────────
non_c = [r for r in records if r["_decisive"]]
n_correct = sum(1 for r in non_c if r["_correct"])
print(f"\n── 5. Non-C responses ({len(non_c)}, acc={100*n_correct/max(len(non_c),1):.1f}%) ──")
print(f"{'Video':<26} {'t':>3} {'Ch':>4} {'GT':>4} {'OK':>4} {'pA':>6} {'conf':>5}")
for r in sorted(non_c, key=lambda x: (x["traj_type"], x["_side"], x["video_id"], x["t_sec"])):
    ok = "YES" if r["_correct"] else "NO "
    print(f"{r['video_id']:<26} {r['t_sec']:>3} {r['choice']:>4} {r['_gt']:>4} {ok:>4} {r['pA']:>6.3f} {r['confidence']:>5}")

# ─────────────────────────────────────────
# 6. VLO (Velocity of Legibility Onset)
# ─────────────────────────────────────────
print(f"\n── 6. VLO per video (first t → decisive + correct) ──")
by_video = defaultdict(list)
for r in records: by_video[r["video_id"]].append(r)

vlo_by_type = defaultdict(list)
neverdec = []
for vid in sorted(by_video):
    recs_s = sorted(by_video[vid], key=lambda r: r["t_sec"])
    vlo = next((r["t_sec"] for r in recs_s if r["_decisive"] and r["_correct"]), None)
    ty = vid.split("_")[1][:3]
    if vlo is not None:
        vlo_by_type[ty].append(vlo)
        print(f"  {vid:<28} VLO=t{vlo}s")
    else:
        neverdec.append(vid)

print(f"\n  Never correct decisive: {len(neverdec)} of 40 videos")
for v in neverdec[:6]: print(f"    {v}")
if len(neverdec) > 6: print(f"    ... and {len(neverdec)-6} more")

print(f"\n  Mean VLO by traj_type (lower = more legible early):")
for ty in ["dec", "neu", "leg"]:
    vals = vlo_by_type.get(ty, [])
    n_videos = len([v for v in sorted(by_video) if v.split("_")[1][:3] == ty])
    n_total = n_videos // (K + 1) if K > 0 else 1
    print(f"    {ty}: mean={sum(vals)/len(vals):.2f}s  n_VLO={len(vals)}")

# ─────────────────────────────────────────
# 7. Deceptive feint signal
# ─────────────────────────────────────────
print(f"\n── 7. Deceptive feint signal (pA_t1 vs pA_t{K}) ──")
print(f"  For left-side (gt=A): pA_t1 < pA_tK = feinted toward B early")
print(f"  For right-side (gt=B): pA_t1 > pA_tK = feinted toward A early")
for ty in ["deceptive", "neutral", "legible"]:
    for sd in ["left", "right"]:
        d1 = [r["pA"] for r in records if r["traj_type"]==ty and r["_side"]==sd and r["t_sec"]==1]
        dK = [r["pA"] for r in records if r["traj_type"]==ty and r["_side"]==sd and r["t_sec"]==K]
        if d1 and dK:
            pA1 = sum(d1)/len(d1)
            pAK = sum(dK)/len(dK)
            print(f"  {ty:<12} {sd:<6} pA_t1={pA1:.3f}  pA_tK={pAK:.3f}  Δ={pA1-pAK:+.3f}")

# ─────────────────────────────────────────
# 8. Final summary table
# ─────────────────────────────────────────
print(f"\n── 8. Summary at t={K} (C treated as wrong) ──")
print(f"{'Type':<12} {'Acc':>7} {'Dec%':>6} {'DecAcc':>8}")
print("─" * 36)
for ty in ["deceptive", "neutral", "legible"]:
    sub = [r for r in tK if r["traj_type"] == ty]
    all_acc = 100 * sum(1 for r in sub if r["_correct"]) / len(sub) if sub else 0
    dec = [r for r in sub if r["_decisive"]]
    d_acc = 100 * sum(1 for r in dec if r["_correct"]) / len(dec) if dec else float("nan")
    dec_pct = 100 * len(dec) / len(sub) if sub else 0
    print(f"{ty:<12} {all_acc:>6.0f}%  {dec_pct:>5.0f}%  {d_acc:>7.0f}%")
all_acc_overall = 100 * sum(1 for r in tK if r["_correct"]) / len(tK) if tK else 0
print(f"{'OVERALL':<12} {all_acc_overall:>6.0f}%")
