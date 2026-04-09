#!/usr/bin/env python3
"""Fast CFG-only evaluation (no video, no VLM) for all 4 behaviors."""

import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pybullet as p
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_cfg_vlm import (
    load_policy, get_obs_v2, _make_obs_uncond,
    rollout_cfg_only, scripted_grasp, _close_to_block,
    compute_l_early, compute_path_efficiency, compute_clearance, compute_hover_dist,
    add_obstacle_visual, add_waypoint_blocks, remove_bodies,
    WAYPOINT_BLOCKS, TABLE_TOP_Z, OBSTACLE_HEIGHT, OBSTACLE_RADIUS,
)
from envs.twoblockpick_env import TwoBlockPickEnv

CKPT = "runs/cfg_20260406_005407/ckpt_ep200.pt"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--cube_jitter', type=float, default=0.0)
    ap.add_argument('--cfg_lambda', type=float, default=None,
                    help='Override CFG lambda for all behaviors (e.g. 0.0 for ablation)')
    ap.add_argument('--output', type=str, default=None,
                    help='Output directory (default: auto-named)')
    args = ap.parse_args()
    N = args.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sampler, stats, ckpt = load_policy(CKPT, device)
    env = TwoBlockPickEnv(
        render=False, cube_jitter=args.cube_jitter, cube_half=0.015,
        cube_mass=0.08, cube_lateral_friction=2.5, episode_length=600,
    )
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(0, 100000, size=1000)
    all_results = {"config": {"N": N, "cube_jitter": args.cube_jitter,
                               "cfg_lambda_override": args.cfg_lambda}}
    t0 = time.time()

    # ── 1. LEGIBILITY ───────────────────────────────────────────
    leg_lam = args.cfg_lambda if args.cfg_lambda is not None else 2.0
    print(f"=== LEGIBILITY (lambda={leg_lam:+.1f}, mode=+1.0) ===")
    res = []
    for i in range(N):
        s = int(seeds[i])
        tgt = "left" if i % 2 == 0 else "right"
        env.reset(seed=s)
        r = rollout_cfg_only(
            env, model, sampler, stats, device,
            context_pos=None, behavior_mode=1.0, cfg_lambda=leg_lam,
            target=tgt, max_steps=400, video_path=None,
            zero_context=False,
        )
        actual = r.get("approached", tgt) or tgt
        le = compute_l_early(r["ee_traj"], r["final_obs"][8:11],
                             r["final_obs"][15:18], actual)
        tag = "OK" if r["success"] else "FAIL"
        print(f"  [{i}] {tag}  tgt={tgt} appr={actual} L_early={le:.3f} steps={r['steps']}")
        res.append({"success": r["success"], "L_early": round(le, 4),
                     "target": tgt, "approached": actual, "seed": s})
    n_ok = sum(1 for x in res if x["success"])
    vals = [x["L_early"] for x in res]
    print(f"  => {n_ok}/{N} success, L_early={np.mean(vals):.3f} +/- {np.std(vals):.3f}\n")
    all_results["legibility"] = res

    # ── 2. PREDICTABILITY ───────────────────────────────────────
    pred_lam = args.cfg_lambda if args.cfg_lambda is not None else 2.0
    print(f"=== PREDICTABILITY (lambda={pred_lam:+.1f}, mode=-1.0) ===")
    res = []
    for i in range(N):
        s = int(seeds[200 + i])
        tgt = "left" if i % 2 == 0 else "right"
        env.reset(seed=s)
        r = rollout_cfg_only(
            env, model, sampler, stats, device,
            context_pos=None, behavior_mode=-1.0, cfg_lambda=pred_lam,
            target=tgt, max_steps=400, video_path=None,
            zero_context=False,
        )
        pe = compute_path_efficiency(r["ee_traj"])
        tag = "OK" if r["success"] else "FAIL"
        actual = r.get("approached", tgt) or tgt
        print(f"  [{i}] {tag}  tgt={tgt} appr={actual} eff={pe:.3f} steps={r['steps']}")
        res.append({"success": r["success"], "path_efficiency": round(pe, 4),
                     "target": tgt, "approached": actual, "seed": s})
    n_ok = sum(1 for x in res if x["success"])
    vals = [x["path_efficiency"] for x in res]
    print(f"  => {n_ok}/{N} success, eff={np.mean(vals):.3f} +/- {np.std(vals):.3f}\n")
    all_results["predictability"] = res

    # ── 3. SAFETY ───────────────────────────────────────────────
    print("=== SAFETY (alternating obstacle placements) ===")
    res = []
    for i in range(N):
        s = int(seeds[400 + i])
        tgt = "left" if i % 2 == 0 else "right"
        if i % 4 < 2:
            lam, mode = (args.cfg_lambda if args.cfg_lambda is not None else 2.0), 1.0
            oy = 0.02 if tgt == "left" else -0.02
            obs_pos = [0.46, oy, TABLE_TOP_Z + OBSTACLE_HEIGHT / 2]
        else:
            lam, mode = (args.cfg_lambda if args.cfg_lambda is not None else 2.0), -1.0
            oy = 0.11 if tgt == "left" else -0.11
            obs_pos = [0.38, oy, TABLE_TOP_Z + OBSTACLE_HEIGHT / 2]
        env.reset(seed=s)
        uid = add_obstacle_visual(env, obs_pos)
        for _ in range(30):
            p.stepSimulation(physicsClientId=env._cid)
        r = rollout_cfg_only(
            env, model, sampler, stats, device,
            context_pos=obs_pos, behavior_mode=mode, cfg_lambda=lam,
            target=tgt, max_steps=400, video_path=None,
            zero_context=True,
        )
        cl = compute_clearance(r["ee_traj"], np.array(obs_pos[:2]))
        collision = cl < OBSTACLE_RADIUS
        tag = "OK" if r["success"] else "FAIL"
        print(f"  [{i}] {tag}  lam={lam:+.0f} clearance={cl:.3f} collision={collision}")
        res.append({"success": r["success"], "clearance": round(cl, 4),
                     "collision": collision, "seed": s})
        remove_bodies(env, [uid])
    n_ok = sum(1 for x in res if x["success"])
    n_clear = sum(1 for x in res if not x["collision"])
    cl_vals = [x["clearance"] for x in res]
    print(f"  => {n_ok}/{N} success, {n_clear}/{N} no_collision, clearance={np.mean(cl_vals):.3f}\n")
    all_results["safety"] = res

    # ── 4. GROUNDING ────────────────────────────────────────────
    grd_lam = args.cfg_lambda if args.cfg_lambda is not None else 1.0
    print(f"=== GROUNDING (lambda={grd_lam:+.1f}, mode=0.0) ===")
    res = []
    for i in range(N):
        s = int(seeds[600 + i])
        tgt = "left" if i % 2 == 0 else "right"
        wp = WAYPOINT_BLOCKS[i % len(WAYPOINT_BLOCKS)]
        env.reset(seed=s)
        wp_uids = add_waypoint_blocks(env, WAYPOINT_BLOCKS)
        for _ in range(30):
            p.stepSimulation(physicsClientId=env._cid)
        r = rollout_cfg_only(
            env, model, sampler, stats, device,
            context_pos=wp["pos"], behavior_mode=0.0, cfg_lambda=grd_lam,
            target=tgt, max_steps=400, video_path=None,
            zero_context=True,
        )
        hd = compute_hover_dist(r["ee_traj"], wp["pos"])
        hovered = hd < 0.06
        tag = "OK" if r["success"] else "FAIL"
        print(f"  [{i}] {tag}  wp={wp['name']} hover_dist={hd:.3f} hovered={hovered}")
        res.append({"success": r["success"], "hover_dist": round(hd, 4),
                     "hovered": hovered, "waypoint": wp["name"], "seed": s})
        remove_bodies(env, wp_uids)
    n_ok = sum(1 for x in res if x["success"])
    n_hov = sum(1 for x in res if x["hovered"])
    hd_vals = [x["hover_dist"] for x in res]
    print(f"  => {n_ok}/{N} success, {n_hov}/{N} hovered, dist={np.mean(hd_vals):.3f}\n")
    all_results["grounding"] = res

    env.close()
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")

    if args.output:
        out = Path(args.output)
    else:
        out = Path("outputs/eval_cfg_fast")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {out / 'results.json'}")


if __name__ == "__main__":
    main()
