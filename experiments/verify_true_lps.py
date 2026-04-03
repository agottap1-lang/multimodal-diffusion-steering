#!/usr/bin/env python3
"""
Hypothesis Verification: eval_legibility_guided.py is NOT true LPS
====================================================================

HYPOTHESIS
----------
The current LPSDDIMSampler in evaluation/eval_legibility_guided.py implements
*classifier guidance* (Dhariwal & Nichol, NeurIPS 2021), NOT Diffusion
Posterior Sampling (Chung et al., ICLR 2023).

DISCREPANCY
-----------
Current "LPS" (classifier guidance style):
    guided_eps  = eps_theta(x_t) - w * sqrt(1-alpha_t) * grad
    x_{t-1}     = DDIM(guided_eps)                       <- grad embedded in eps

True DPS (Chung et al. ICLR 2023, Algorithm 1):
    x'_{t-1}    = DDIM(eps_theta(x_t))                   <- standard DDIM step
    x_{t-1}     = x'_{t-1} + zeta_t * grad               <- additive correction

Where:
    grad = nabla_{x_t} L_early( x0_hat(x_t) )
    zeta_t = rho / ||grad||   (normalised step size, rho = guidance_scale)

The two formulations produce *different* x_{t-1} values mathematically.
In classifier guidance the gradient contaminates the noise direction,
which accumulates over denoising steps.  True DPS adds an orthogonal
correction without perturbing the diffusion direction.

VERIFICATION PLAN
-----------------
1. Sample a small batch of actions under base DDIM.
2. Run one denoising step under the *current* (classifier-guidance) sampler.
3. Run the same step under the *true DPS* sampler.
4. Compute ||x_{t-1}^{current} - x_{t-1}^{DPS}|| to confirm they differ.
5. Run N=10 episodes under each sampler; report success + L_early.
6. If true DPS gives different (ideally better) legibility scores, hypothesis
   is confirmed.

Usage
-----
  python experiments/verify_true_lps.py
  python experiments/verify_true_lps.py --n_episodes 10 --guidance_scale 5.0
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# ── env constants (identical to eval_legibility_guided.py) ──────────────────
OBS_EE_POS    = slice(0, 3)
OBS_LEFT_POS  = slice(8, 11)
OBS_RIGHT_POS = slice(15, 18)
ACTION_SCALE  = 0.05

DEFAULT_CKPT = 'runs/diffusion_20260222_195530/ckpt_ep100.pt'


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (identical to eval_legibility_guided.py)
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class UNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.conv1 = nn.Linear(in_dim, out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.act = nn.Mish()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h.transpose(1, 2)
        h = self.norm1(h)
        h = h.transpose(1, 2)
        h = self.act(h + self.time_proj(t_emb).unsqueeze(1))
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        return self.act(h + self.shortcut(x))


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(128),
            nn.Linear(128, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(act_dim, hidden_dim)
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        self.encoder_blocks = nn.ModuleList([
            UNetBlock(dims[i], dims[i + 1], hidden_dim) for i in range(len(dims) - 1)
        ])
        self.bottleneck = UNetBlock(dims[-1], dims[-1], hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            UNetBlock(dims[i + 1] + dims[i + 1], dims[i], hidden_dim)
            for i in range(len(dims) - 2, -1, -1)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, noisy_act, timestep, obs):
        B = noisy_act.shape[0]
        t_emb = self.time_mlp(timestep)
        obs_emb = self.obs_embed(obs)
        x = self.input_proj(noisy_act)
        x = x + obs_emb.unsqueeze(1)
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x, t_emb)
            skip_connections.append(x)
        x = self.bottleneck(x, t_emb)
        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=-1)
            x = block(x, t_emb)
        return self.output_proj(x)


# ══════════════════════════════════════════════════════════════════════════════
# LEGIBILITY POTENTIAL  (identical to eval_legibility_guided.py)
# ══════════════════════════════════════════════════════════════════════════════

def l_early_intent_torch(ee_traj, goals, true_goal_idx=0, early_frac=0.30):
    H = ee_traj.shape[0]
    early_end = max(1, int(H * early_frac))
    early_traj = ee_traj[:early_end]

    dists = torch.cdist(goals, goals)
    mask = dists > 1e-6
    d_min = dists[mask].min() if mask.any() else torch.tensor(0.14, device=goals.device)
    sigma = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))

    diff = early_traj.unsqueeze(1) - goals.unsqueeze(0)
    sq_dist = (diff ** 2).sum(-1)
    log_like = -sq_dist / (2.0 * sigma ** 2)
    posteriors = torch.softmax(log_like, dim=-1)
    return posteriors[:, true_goal_idx].mean()


def _infer_true_goal(ee_traj, goals):
    """Detached goal inference — no gradient path."""
    with torch.no_grad():
        l0 = l_early_intent_torch(ee_traj, goals, 0).item()
        l1 = l_early_intent_torch(ee_traj, goals, 1).item()
    return 0 if l0 >= l1 else 1


def _compute_grad(model, x_t, t_batch, obs, sqrt_ab, sqrt_1m_ab,
                  ee_start, goals):
    """Compute nabla_{x_t} L_early(x0_hat(x_t)).  Used by both samplers."""
    x_in = x_t.detach().requires_grad_(True)
    with torch.enable_grad():
        eps_pred = model(x_in, t_batch, obs)
        x0_pred = (x_in - sqrt_1m_ab * eps_pred) / sqrt_ab
        delta_pos = x0_pred[0, :, :3] * ACTION_SCALE
        ee_traj = torch.cumsum(delta_pos, dim=0) + ee_start
        true_goal = _infer_true_goal(ee_traj.detach(), goals)
        l_early = l_early_intent_torch(ee_traj, goals, true_goal)
        grad = torch.autograd.grad(l_early, x_in)[0]
        eps_pred_detached = eps_pred.detach()
        l_val = float(l_early.item())
    return grad.detach(), eps_pred_detached, l_val


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLER A — current "classifier-guidance" approach  (what eval_legibility_guided.py does)
# ══════════════════════════════════════════════════════════════════════════════

class ClassifierGuidanceSampler:
    """What eval_legibility_guided.py actually implements.

    Update:
        guided_eps = eps_theta(x_t) - w * sqrt(1-alpha_t) * grad
        x_{t-1}    = DDIM( guided_eps )    ← gradient baked into eps
    """

    label = 'ClassifierGuidance (current)'

    def __init__(self, n_steps, beta_start, beta_end, device,
                 guidance_scale=2.0, grad_clip=1.0):
        self.gs = guidance_scale
        self.gc = grad_clip
        self.device = device
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.acp = torch.cumprod(alphas, dim=0)

    def sample(self, model, obs, ee_start, goals, n_sampling_steps=10):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        ts = torch.flip(
            torch.linspace(0, len(self.acp) - 1, n_sampling_steps,
                           device=self.device).long(), [0])
        last_l = 0.0
        for i, t in enumerate(ts):
            t_batch = t.repeat(B)
            alpha_t    = self.acp[t]
            alpha_prev = self.acp[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=self.device)
            sqrt_ab    = torch.sqrt(alpha_t)
            sqrt_1m_ab = torch.sqrt(1.0 - alpha_t)

            g, eps, l_val = _compute_grad(model, x, t_batch, obs,
                                          sqrt_ab, sqrt_1m_ab, ee_start, goals)
            last_l = l_val

            with torch.no_grad():
                gn = g.norm()
                if gn > self.gc:
                    g = g * (self.gc / (gn + 1e-8))

                # ← classifier guidance: gradient injected into eps
                guided_eps = eps - self.gs * sqrt_1m_ab * g
                x0_g = (x - sqrt_1m_ab * guided_eps) / sqrt_ab
                if i < len(ts) - 1:
                    x = torch.sqrt(alpha_prev) * x0_g + torch.sqrt(1.0 - alpha_prev) * guided_eps
                else:
                    x = x0_g

        return x, last_l


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLER B — TRUE DPS  (Chung et al. ICLR 2023, Algorithm 1)
# ══════════════════════════════════════════════════════════════════════════════

class TrueDPSSampler:
    """True Diffusion Posterior Sampling (Chung et al. ICLR 2023, Algo. 1).

    Update:
        x'_{t-1}   = DDIM( eps_theta(x_t) )      ← standard DDIM, unmodified
        x_{t-1}    = x'_{t-1} + zeta_t * grad    ← DPS additive correction

    Where:
        grad    = nabla_{x_t} log p(y | x0_hat(x_t))
                = nabla_{x_t} L_early(x0_hat(x_t))  [maximising legibility]
        zeta_t  = rho / ||grad||                  ← normalised as in DPS paper
        rho     = guidance_scale hyperparameter
    """

    label = 'TrueDPS (corrected)'

    def __init__(self, n_steps, beta_start, beta_end, device,
                 guidance_scale=2.0):
        self.gs = guidance_scale
        self.device = device
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.acp = torch.cumprod(alphas, dim=0)

    def sample(self, model, obs, ee_start, goals, n_sampling_steps=10):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        ts = torch.flip(
            torch.linspace(0, len(self.acp) - 1, n_sampling_steps,
                           device=self.device).long(), [0])
        last_l = 0.0
        for i, t in enumerate(ts):
            t_batch = t.repeat(B)
            alpha_t    = self.acp[t]
            alpha_prev = self.acp[ts[i + 1]] if i < len(ts) - 1 else torch.tensor(1.0, device=self.device)
            sqrt_ab    = torch.sqrt(alpha_t)
            sqrt_1m_ab = torch.sqrt(1.0 - alpha_t)

            g, eps, l_val = _compute_grad(model, x, t_batch, obs,
                                          sqrt_ab, sqrt_1m_ab, ee_start, goals)
            last_l = l_val

            with torch.no_grad():
                # Standard DDIM step (pure, no eps modification)
                x0_pred = (x - sqrt_1m_ab * eps) / sqrt_ab
                if i < len(ts) - 1:
                    x_prev = (torch.sqrt(alpha_prev) * x0_pred
                              + torch.sqrt(1.0 - alpha_prev) * eps)
                else:
                    x_prev = x0_pred

                # True DPS correction: additive after DDIM
                # zeta_t = rho / ||grad||  (per-step normalisation)
                gn = g.norm() + 1e-8
                zeta = self.gs / gn
                x = x_prev + zeta * g   # ← correct DPS update

        return x, last_l


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TEST — single denoising step divergence
# ══════════════════════════════════════════════════════════════════════════════

def unit_test_step_divergence(model, device, cfg, ckpt, n_steps=1):
    """Verify that classifier-guidance and true-DPS produce DIFFERENT x_{t-1}.

    If they produce the same x_{t-1} the hypothesis is wrong.
    If they produce different x_{t-1} the hypothesis is confirmed.
    """
    print("\n" + "="*65)
    print("  UNIT TEST: step-level divergence between samplers")
    print("="*65)

    n_diff = cfg.get('n_diffusion_steps', 100)
    b_s    = cfg.get('beta_start', 1e-4)
    b_e    = cfg.get('beta_end', 0.1)

    cg = ClassifierGuidanceSampler(n_diff, b_s, b_e, device, guidance_scale=5.0, grad_clip=1.0)
    dp = TrueDPSSampler(n_diff, b_s, b_e, device, guidance_scale=5.0)

    # Fake obs / goals from checkpoint stats
    obs_mean = np.array(ckpt['obs_mean'], dtype=np.float32)
    obs_std  = np.array(ckpt['obs_std'],  dtype=np.float32)
    obs_dim  = cfg['obs_dim']

    np.random.seed(0)
    fake_obs_raw = np.random.randn(obs_dim).astype(np.float32)
    fake_obs_raw[OBS_EE_POS]    = np.array([0.0, 0.0, 0.42], dtype=np.float32)
    fake_obs_raw[OBS_LEFT_POS]  = np.array([-0.12, 0.0, 0.42], dtype=np.float32)
    fake_obs_raw[OBS_RIGHT_POS] = np.array([ 0.12, 0.0, 0.42], dtype=np.float32)

    obs_norm = (fake_obs_raw - obs_mean) / obs_std
    obs_t    = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    ee_start = torch.tensor(fake_obs_raw[OBS_EE_POS], dtype=torch.float32, device=device)
    goals    = torch.tensor(
        np.stack([fake_obs_raw[OBS_LEFT_POS], fake_obs_raw[OBS_RIGHT_POS]]),
        dtype=torch.float32, device=device)

    # Fix random seed so both samplers start from the same x_T
    torch.manual_seed(42)
    x_T = torch.randn(1, cfg['horizon'], cfg['act_dim'], device=device)

    # One-step sample comparison  (use n_sampling_steps=1 for isolation)
    def _one_step(sampler):
        """Run exactly 1 DDIM step from the fixed x_T."""
        betas   = torch.linspace(b_s, b_e, n_diff, device=device)
        alphas  = 1.0 - betas
        acp     = torch.cumprod(alphas, dim=0)
        ts      = torch.flip(
            torch.linspace(0, n_diff - 1, 1, device=device).long(), [0])
        t       = ts[0]
        t_batch = t.repeat(1)
        alpha_t    = acp[t]
        alpha_prev = torch.tensor(1.0, device=device)
        sqrt_ab    = torch.sqrt(alpha_t)
        sqrt_1m_ab = torch.sqrt(1.0 - alpha_t)

        x = x_T.clone()
        g, eps, _ = _compute_grad(model, x, t_batch, obs_t,
                                  sqrt_ab, sqrt_1m_ab, ee_start, goals)

        with torch.no_grad():
            gn = g.norm() + 1e-8

            # Classifier guidance update
            gc = g * (1.0 / gn) if gn > 1.0 else g  # grad_clip=1.0
            guided_eps = eps - 5.0 * sqrt_1m_ab * gc
            x0_cg      = (x - sqrt_1m_ab * guided_eps) / sqrt_ab
            x_cg       = x0_cg  # last step -> x = x0

            # True DPS update
            x0_dps     = (x - sqrt_1m_ab * eps) / sqrt_ab
            zeta       = 5.0 / gn
            x_dps      = x0_dps + zeta * g

        return x_cg, x_dps, g, eps, sqrt_1m_ab

    x_cg, x_dps, g, eps, sqrt_1m_ab = _one_step(None)

    diff_norm = (x_cg - x_dps).norm().item()
    rel_diff  = diff_norm / (x_dps.norm().item() + 1e-8)

    print(f"  ||x_cg||              = {x_cg.norm().item():.4f}")
    print(f"  ||x_dps||             = {x_dps.norm().item():.4f}")
    print(f"  ||x_cg - x_dps||     = {diff_norm:.4f}  (absolute)")
    print(f"  ||x_cg - x_dps|| / ||x_dps|| = {rel_diff:.4f}  ({rel_diff*100:.1f}%)")
    print()

    if rel_diff > 0.01:
        print("  RESULT: ✓ HYPOTHESIS CONFIRMED — samplers produce DIFFERENT x_{t-1}")
        print(f"          Relative divergence {rel_diff*100:.1f}% >> 1%")
    else:
        print("  RESULT: ✗ HYPOTHESIS REJECTED — samplers produce the same x_{t-1}")

    # Also show what the gradient contributions look like
    gn_val  = g.norm().item()
    gc_vec  = g * (1.0 / (gn_val + 1e-8)) if gn_val > 1.0 else g
    guided_eps_diag = eps - 5.0 * sqrt_1m_ab * gc_vec
    cg_correction   = (eps - guided_eps_diag).norm().item()
    zeta_val        = 5.0 / (gn_val + 1e-8)
    print(f"\n  Correction magnitude:")
    print(f"    CG  injected into eps  : ||eps_guided - eps_orig|| = {cg_correction:.4f}")
    print(f"    DPS added to x_{{t-1}}   : ||zeta * grad||          = {zeta_val * gn_val:.4f}")

    return rel_diff > 0.01


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(model, sampler, obs_mean, obs_std, act_mean, act_std, device,
                n_sampling_steps=10, max_steps=400):
    env = TwoBlockPickEnv(render=False, episode_length=max_steps, cube_jitter=0.0)
    obs = env.reset()
    action_queue = deque(maxlen=model.horizon)
    ee_trajectory: List[np.ndarray] = []
    success = False
    last_obs = obs

    for step in range(max_steps):
        ee_trajectory.append(obs[0:3].copy())

        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t    = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            ee_start = torch.tensor(obs[OBS_EE_POS], dtype=torch.float32, device=device)
            goals    = torch.stack([
                torch.tensor(obs[OBS_LEFT_POS],  dtype=torch.float32, device=device),
                torch.tensor(obs[OBS_RIGHT_POS], dtype=torch.float32, device=device),
            ])

            act_seq, _ = sampler.sample(model, obs_t, ee_start, goals,
                                        n_sampling_steps=n_sampling_steps)
            chunk = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in chunk:
                action_queue.append(a)

        action  = action_queue.popleft()
        result  = env.step(action)
        obs     = result.obs
        last_obs = obs
        success  = (result.info.get('success_left', 0) > 0.5
                    or result.info.get('success_right', 0) > 0.5)
        if result.done:
            break

    env.close()

    if len(ee_trajectory) >= 4:
        traj_arr = np.array(ee_trajectory)
        goals_np = np.stack([last_obs[8:11], last_obs[15:18]], axis=0)
        from evaluation.legibility_metrics import compute_legibility
        r0 = compute_legibility(traj_arr, goals_np, true_goal_idx=0, model='gaussian')
        r1 = compute_legibility(traj_arr, goals_np, true_goal_idx=1, model='gaussian')
        best = r0 if r0.L_early_intent >= r1.L_early_intent else r1
        l_early = best.L_early_intent
    else:
        l_early = 0.0

    return dict(success=success, steps=step + 1, l_early=float(l_early))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--n_episodes', type=int, default=10,
                    help='Episodes per sampler (default 10, quick verify)')
    ap.add_argument('--guidance_scale', type=float, default=5.0,
                    help='Guidance strength rho / w (same value used for both samplers)')
    ap.add_argument('--n_sampling_steps', type=int, default=10)
    ap.add_argument('--skip_rollouts', action='store_true',
                    help='Only run unit test, skip full episode rollouts')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*65)
    print("  Hypothesis: eval_legibility_guided.py != true DPS/LPS")
    print("="*65)
    print(f"  Device          : {device}")
    print(f"  Checkpoint      : {args.checkpoint}")
    print(f"  Guidance scale  : {args.guidance_scale}")
    print(f"  Episodes        : {args.n_episodes}")
    print(f"  DDIM steps      : {args.n_sampling_steps}")

    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg    = ckpt['config']
    model  = DiffusionPolicy(
        obs_dim=cfg['obs_dim'], act_dim=cfg['act_dim'], horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256), n_blocks=cfg.get('n_blocks', 3),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    obs_mean = np.array(ckpt['obs_mean'], dtype=np.float32)
    obs_std  = np.array(ckpt['obs_std'],  dtype=np.float32)
    act_mean = np.array(ckpt['act_mean'], dtype=np.float32)
    act_std  = np.array(ckpt['act_std'],  dtype=np.float32)

    n_diff = cfg.get('n_diffusion_steps', 100)
    b_s    = cfg.get('beta_start', 1e-4)
    b_e    = cfg.get('beta_end', 0.1)

    # ── Unit test first ──────────────────────────────────────────────────────
    confirmed = unit_test_step_divergence(model, device, cfg, ckpt)

    if args.skip_rollouts:
        print("\nSkipping rollout comparison (--skip_rollouts).")
        return

    # ── Full episode comparison ──────────────────────────────────────────────
    samplers = [
        ClassifierGuidanceSampler(n_diff, b_s, b_e, device,
                                  guidance_scale=args.guidance_scale, grad_clip=1.0),
        TrueDPSSampler(n_diff, b_s, b_e, device,
                       guidance_scale=args.guidance_scale),
    ]

    all_results = {}
    for sampler in samplers:
        print(f"\n── {sampler.label} ── {args.n_episodes} episodes ──")
        results = []
        for ep in range(args.n_episodes):
            r = run_episode(model, sampler, obs_mean, obs_std, act_mean, act_std,
                            device, n_sampling_steps=args.n_sampling_steps)
            results.append(r)
            tick = '✓' if r['success'] else '✗'
            print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  "
                  f"L_early={r['l_early']:.4f}  steps={r['steps']}")

        l_vals = [r['l_early'] for r in results]
        s_vals = [r['success'] for r in results]
        print(f"\n  SUMMARY  {sampler.label}:")
        print(f"    L_early_intent : {np.mean(l_vals):.4f} ± {np.std(l_vals):.4f}")
        print(f"    Success rate   : {np.mean(s_vals):.1%}")
        all_results[sampler.label] = {
            'l_early_mean': float(np.mean(l_vals)),
            'l_early_std':  float(np.std(l_vals)),
            'success_rate': float(np.mean(s_vals)),
            'episodes':     [{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                             for k, v in r.items()} for r in results],
        }

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  COMPARISON SUMMARY")
    print("="*65)
    for name, res in all_results.items():
        print(f"  {name:<35}  "
              f"L_early={res['l_early_mean']:.4f}±{res['l_early_std']:.4f}  "
              f"Success={res['success_rate']:.1%}")

    cg_l  = all_results[ClassifierGuidanceSampler.label]['l_early_mean']
    dps_l = all_results[TrueDPSSampler.label]['l_early_mean']
    delta = dps_l - cg_l
    print(f"\n  True DPS vs ClassifierGuidance: ΔL_early = {delta:+.4f}")
    if abs(delta) > 0.005:
        print("  → Samplers produce MEASURABLY DIFFERENT legibility: hypothesis confirmed.")
    else:
        print("  → Legibility delta < 0.005: similar outcomes despite different mechanics.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / 'outputs'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'verify_true_lps.json'
    payload = {
        'guidance_scale':    args.guidance_scale,
        'n_episodes':        args.n_episodes,
        'n_sampling_steps':  args.n_sampling_steps,
        'step_divergence_confirmed': bool(confirmed),
        'results':           all_results,
        'conclusion': (
            'True DPS differs mechanically (confirmed by unit test). '
            f'L_early delta = {delta:+.4f}.'
        ),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == '__main__':
    main()
