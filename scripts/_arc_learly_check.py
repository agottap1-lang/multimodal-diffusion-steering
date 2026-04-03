import numpy as np, math

# ── Timing facts ──────────────────────────────────────────────────────────
# _SUBSTEPS=20, dt=1/240 => each env.step() = 20/240 s physics time
# Video recorded at 30fps, 1 frame captured per step()
# => t=1s = step 30, t=2s = step 60, t=3s = step 90
# => 30% early window = first 90 steps = first 3 video-seconds
STEPS_PER_VIDEO_SEC = 30

demos = np.load('data/demos/demos.npz', allow_pickle=True)
obs  = demos['obs']
labs = demos['labels']
ep_lens = demos['episode_lengths']


def posterior_curve(ee_traj, goals, true_goal_idx):
    """
    Returns per-step P(g* | xi_{0:t}) for two models:
      p_spatial[t] -- Gaussian spatial (IPF / Shi 2025)
      p_cost[t]    -- Boltzmann path-efficiency (Dragan HRI 2013)
    """
    d_min = np.linalg.norm(goals[0] - goals[1])
    sigma = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))
    c_bar = float(np.mean([np.linalg.norm(ee_traj[0] - g) for g in goals]))
    beta  = 1.0 / max(c_bar, 1e-6)

    H = len(ee_traj)
    p_spatial = np.zeros(H)
    p_cost    = np.zeros(H)
    cum_path  = 0.0

    for t in range(H):
        pos = ee_traj[t]
        # Gaussian spatial posterior
        log_sp = np.array([-float(np.sum((pos - g)**2)) / (2*sigma**2) for g in goals])
        log_sp -= log_sp.max()
        exp_sp = np.exp(log_sp)
        p_spatial[t] = float(exp_sp[true_goal_idx] / exp_sp.sum())

        # Boltzmann path-efficiency posterior (cumulative trajectory)
        if t > 0:
            cum_path += float(np.linalg.norm(ee_traj[t] - ee_traj[t-1]))
        log_co = np.array([
            beta * (float(np.linalg.norm(ee_traj[0] - g)) - cum_path - float(np.linalg.norm(pos - g)))
            for g in goals
        ])
        log_co -= log_co.max()
        exp_co = np.exp(log_co)
        p_cost[t] = float(exp_co[true_goal_idx] / exp_co.sum())

    return p_spatial, p_cost


# ── Compute curves for all 400 episodes ──────────────────────────────────
records = []
for ep in range(400):
    ep_len  = min(int(ep_lens[ep]), 400)
    ee      = obs[ep, :ep_len, 0:3]
    left_g  = obs[ep, 0, 8:11]
    right_g = obs[ep, 0, 15:18]
    goals   = np.stack([left_g, right_g])
    true_idx = 0 if labs[ep] == 'left' else 1
    sign     = 1 if labs[ep] == 'left' else -1
    # absolute arc magnitude: how far the EE sweeps toward the correct side
    abs_arc = abs(float((sign * obs[ep, :ep_len, 1]).max()))
    ps, pc   = posterior_curve(ee, goals, true_idx)
    records.append({'abs_arc': abs_arc, 'ps': ps, 'pc': pc, 'ep_len': ep_len})

records.sort(key=lambda r: r['abs_arc'])
n = len(records)
small_q = records[:n//4]   # smallest 25% arc magnitudes
large_q = records[-n//4:]  # largest 25% arc magnitudes

print("=== Spatial (Gaussian IPF) model: P(g* | x_t) per second ===")
print(f"  t      | small arcs | large arcs |  delta")
print("  -------|------------|------------|--------")
for t_sec in range(10):
    t_step = t_sec * STEPS_PER_VIDEO_SEC
    sm = np.mean([r['ps'][min(t_step, r['ep_len']-1)] for r in small_q])
    lg = np.mean([r['ps'][min(t_step, r['ep_len']-1)] for r in large_q])
    tag = "  <-- 30% early window end" if t_sec == 3 else ""
    print(f"  t={t_sec}s   |   {sm:.4f}   |   {lg:.4f}   | {lg-sm:+.4f}{tag}")

print()
print("=== Boltzmann path-efficiency model: P(g* | xi_0:t) per second ===")
print(f"  t      | small arcs | large arcs |  delta")
print("  -------|------------|------------|--------")
for t_sec in range(10):
    t_step = t_sec * STEPS_PER_VIDEO_SEC
    sm = np.mean([r['pc'][min(t_step, r['ep_len']-1)] for r in small_q])
    lg = np.mean([r['pc'][min(t_step, r['ep_len']-1)] for r in large_q])
    tag = "  <-- 30% early window end" if t_sec == 3 else ""
    print(f"  t={t_sec}s   |   {sm:.4f}   |   {lg:.4f}   | {lg-sm:+.4f}{tag}")

print()
print("=== L_early = mean P over t=0..3s (steps 0..90) ===")
for label, grp in [("Small arcs", small_q), ("Large arcs", large_q)]:
    sp_vals, co_vals = [], []
    for r in grp:
        early_end = min(90, r['ep_len'])
        sp_vals.append(float(r['ps'][:early_end].mean()))
        co_vals.append(float(r['pc'][:early_end].mean()))
    print(f"  {label}: spatial={np.mean(sp_vals):.4f}  cost-ratio={np.mean(co_vals):.4f}")

print()
print(f"Arc size range - small: {records[0]['abs_arc']:.3f} to {records[n//4-1]['abs_arc']:.3f} m")
print(f"Arc size range - large: {records[-n//4]['abs_arc']:.3f} to {records[-1]['abs_arc']:.3f} m")
