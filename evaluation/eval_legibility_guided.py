#!/usr/bin/env python3
"""
Training-Free Classifier Guidance for Legible Diffusion Policy
===============================================================

NOTE: Despite legacy variable names (LPS*), this implements classifier
guidance (Dhariwal & Nichol 2021), NOT Diffusion Posterior Sampling.
True DPS was tested and has 0% success rate at practical scales.

Training-free legibility guidance at inference time.

The key idea
------------
Instead of Best-of-N (sample N trajectories, pick best), we inject
a legibility gradient into *every* DDIM denoising step. This follows
Diffusion Posterior Sampling (DPS; Chung et al., ICLR 2023):

    Îµ_guided = Îµ_Î¸(x_t) âˆ’ w Â· âˆš(1âˆ’Î±_t) Â· âˆ‡_{x_t} L_early(xÌ‚â‚€(x_t))

where xÌ‚â‚€ is the denoised prediction at step t, and L_early is the
Bayesian goal-inference legibility score from Dragan et al. (HRI 2013).

Why this is better than Best-of-N
-----------------------------------
- Best-of-N: O(N) inference cost, still random â€” if the policy rarely
  produces legible arcs in N tries, you get nothing.
- LPS: O(1) inference cost â€” every sample is pushed toward legibility
  via the gradient. Works even if the baseline rarely produces legible
  trajectories.

Why this is better than Legibility Diffuser (Bronars et al. RA-L 2024)
------------------------------------------------------------------------
- Bronars: requires a separately-trained guided policy (offline imitation
  of legible demonstration modes).
- LPS: zero retraining â€” works on any existing diffusion policy
  checkpoint. The guidance is injected purely at inference.

Novel contribution over DPS (Chung et al. 2022)
-------------------------------------------------
DPS was designed for image restoration (linear/nonlinear inverse
problems). We adapt it to robot trajectory legibility:
  - guidance potential = L_early_intent (Dragan 2013 Bayesian posterior,
    information-theoretic, not just pixel likelihood)
  - xÌ‚â‚€ is a *action* sequence, converted to EE trajectory via delta
    integration before computing the legibility gradient.

References
----------
[1] Dragan, Lee, Srinivasa. "Legibility and Predictability of Robot Motion."
    HRI 2013.
[2] Chung, Kim, McCann, Klasky, Ye. "Diffusion Posterior Sampling for
    General Noisy Inverse Problems." ICLR 2023. arXiv:2209.14687
[3] Bronars, Cheng, Xu. "Legibility Diffuser: Offline Imitation for
    Intent Expressive Motion." RA-L 2024.
[4] Shi, Grislain, Sigaud, Chetouani. "Controlling Intent Expressiveness
    in Robot Motion with Diffusion Models." arXiv:2510.12370, 2025.
[5] Dhariwal, Nichol. "Diffusion Models Beat GANs on Image Synthesis."
    NeurIPS 2021. (Classifier guidance â€” the special-case parent of DPS)

Usage
-----
  python scripts/eval_legibility_guided.py --checkpoint runs/latest/ckpt.pt
  python scripts/eval_legibility_guided.py --checkpoint runs/latest/ckpt.pt \\
      --guidance_scale 3.0 --n_episodes 30
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# â”€â”€ environment constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
OBS_EE_POS    = slice(0, 3)    # ee_pos in observation
OBS_LEFT_POS  = slice(8, 11)   # left_cube_pos in observation
OBS_RIGHT_POS = slice(15, 18)  # right_cube_pos in observation
ACTION_SCALE  = 0.05           # TwoBlockPickEnv: action_scale_pos

# â”€â”€ Default checkpoint (DiffusionPolicy / UNet â€” matches eval_with_videos.py) â”€
DEFAULT_CKPT = 'runs/diffusion_20260222_195530/ckpt_ep100.pt'


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODEL  (identical to eval_with_videos.py â€” DO NOT CHANGE)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
        return self.output_proj(x)   # (B, H, act_dim) â€” no tanh, raw noise



# DIFFERENTIABLE L_EARLY_INTENT  (Dragan 2013 Bayesian posterior)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def l_early_intent_torch(
    ee_traj: torch.Tensor,      # (H, 3)  predicted EE positions
    goals: torch.Tensor,        # (K, 3)  goal positions
    true_goal_idx: int = 0,
    early_frac: float = 0.30,
) -> torch.Tensor:
    """Differentiable L_early_intent â€” autograd flows through this.

    Implements the Bayesian goal-inference posterior from Dragan et al.
    (HRI 2013), approximated via Information Potential Field (Shi 2025):

        P(g | x) âˆ exp(âˆ’â€–x âˆ’ gâ€–Â² / 2ÏƒÂ²)
        L_early = mean P(g* | x_t) for t in [0, 0.3T]

    Ïƒ is auto-calibrated so half-maximum aligns with the Voronoi boundary
    between goals (same formula as evaluation/legibility_metrics.py).

    Returns a scalar tensor with gradient w.r.t. ee_traj.
    """
    H = ee_traj.shape[0]
    K = goals.shape[0]
    early_end = max(1, int(H * early_frac))
    early_traj = ee_traj[:early_end]   # (early_end, 3)

    # Auto-calibrate sigma from inter-goal distance
    dists = torch.cdist(goals, goals)  # (K, K)
    mask = dists > 1e-6
    if mask.any():
        d_min = dists[mask].min()
    else:
        d_min = torch.tensor(0.14, device=goals.device)
    sigma = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))

    # Gaussian log-likelihood: (early_end, K)
    diff = early_traj.unsqueeze(1) - goals.unsqueeze(0)   # (early_end, K, 3)
    sq_dist = (diff ** 2).sum(-1)                          # (early_end, K)
    log_like = -sq_dist / (2.0 * sigma ** 2)

    # Posterior via softmax over goals at each timestep
    posteriors = torch.softmax(log_like, dim=-1)           # (early_end, K)

    # L_early = mean P(g*) over early window
    l_early = posteriors[:, true_goal_idx].mean()
    return l_early




# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DDIM SAMPLER  (identical to eval_with_videos.py â€” baseline)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class DDIMSampler:
    """Standard DDIM sampler â€” matches eval_with_videos.py exactly."""

    def __init__(self, n_steps, beta_start, beta_end, device):
        self.device = device
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    @torch.no_grad()
    def sample(self, model, obs, n_sampling_steps=10):
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        timesteps = torch.linspace(
            0, len(self.alphas_cumprod) - 1, n_sampling_steps,
            device=self.device
        ).long()
        timesteps = torch.flip(timesteps, [0])
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            pred_noise = model(x, t_batch, obs)
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (self.alphas_cumprod[timesteps[i + 1]]
                          if i < len(timesteps) - 1
                          else torch.tensor(1.0, device=self.device))
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            if i < len(timesteps) - 1:
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = x0_pred
        return x


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLASSIFIER-GUIDED DDIM SAMPLER  (gradient injected at every step)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class LPSDDIMSampler:
    """DDIM with classifier-guidance gradient injection at every denoising step.

    At step t:
        ÎµÌ‚  = Îµ_Î¸(x_t) âˆ’ w Â· âˆš(1âˆ’á¾±_t) Â· âˆ‡_{x_t} L_early(xÌ‚â‚€(x_t), goals)
    Then apply standard DDIM update using ÎµÌ‚ instead of Îµ_Î¸.

    Parameters
    ----------
    guidance_scale : float  (w) â€” 0 = baseline, 2.0 = recommended
    grad_clip : float â€” clip â€–âˆ‡â€– for numerical stability
    """

    def __init__(self, n_steps, beta_start, beta_end, device,
                 guidance_scale: float = 2.0, grad_clip: float = 1.0):
        self.device = device
        self.guidance_scale = guidance_scale
        self.grad_clip = grad_clip
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def sample(
        self,
        model: DiffusionPolicy,
        obs: torch.Tensor,            # (1, obs_dim)
        ee_pos_start: torch.Tensor,   # (3,)
        goals: torch.Tensor,          # (2, 3)
        n_sampling_steps: int = 10,
    ) -> Tuple[torch.Tensor, float]:
        """Sample a guided action chunk.

        Returns (chunk, l_early_last) where l_early_last is the L_early_intent
        of the last predicted x0 (for diagnostics).
        """
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        timesteps = torch.linspace(
            0, len(self.alphas_cumprod) - 1, n_sampling_steps,
            device=self.device
        ).long()
        timesteps = torch.flip(timesteps, [0])

        final_l_early = 0.0

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (self.alphas_cumprod[timesteps[i + 1]]
                          if i < len(timesteps) - 1
                          else torch.tensor(1.0, device=self.device))
            sqrt_ab = torch.sqrt(alpha_t)
            sqrt_1m_ab = torch.sqrt(1.0 - alpha_t)

            # â”€â”€ LPS gradient â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                eps_pred = model(x_in, t_batch, obs)
                x0_pred = (x_in - sqrt_1m_ab * eps_pred) / sqrt_ab

                # Forward kinematics: Î”pos â†’ EE trajectory
                delta_pos = x0_pred[0, :, :3] * ACTION_SCALE   # (H, 3)
                ee_traj = torch.cumsum(delta_pos, dim=0) + ee_pos_start

                # Infer committed goal (detached â†’ no extra grad path)
                with torch.no_grad():
                    l0 = l_early_intent_torch(ee_traj.detach(), goals, 0).item()
                    l1 = l_early_intent_torch(ee_traj.detach(), goals, 1).item()
                true_goal = 0 if l0 >= l1 else 1

                l_early = l_early_intent_torch(ee_traj, goals, true_goal)
                grad = torch.autograd.grad(l_early, x_in)[0]

            final_l_early = float(l_early.item())

            with torch.no_grad():
                # Clip gradient norm
                g = grad.detach()
                gn = g.norm()
                if gn > self.grad_clip:
                    g = g * (self.grad_clip / (gn + 1e-8))

                # Guided noise: ÎµÌ‚ = Îµ_Î¸ âˆ’ wÂ·âˆš(1âˆ’á¾±_t)Â·âˆ‡
                guided_eps = eps_pred.detach() - self.guidance_scale * sqrt_1m_ab * g

                # Standard DDIM update with guided noise
                x0_guided = (x - sqrt_1m_ab * guided_eps) / sqrt_ab
                if i < len(timesteps) - 1:
                    x = torch.sqrt(alpha_prev) * x0_guided + torch.sqrt(1.0 - alpha_prev) * guided_eps
                else:
                    x = x0_guided

        return x, final_l_early


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# EPISODE RUNNER  (matches eval_with_videos.py for correct success rate)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_episode(
    model: DiffusionPolicy,
    sampler,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    act_mean: np.ndarray,
    act_std: np.ndarray,
    device,
    guided: bool = False,
    n_sampling_steps: int = 10,
    cube_jitter: float = 0.0,
    max_steps: int = 400,
) -> dict:
    """Episode runner â€” identical structure to eval_with_videos.py.

    cube_jitter=0.0 matches eval_with_videos.py for reproducible 88-92% baseline.
    """
    env = TwoBlockPickEnv(render=False, episode_length=max_steps,
                          cube_jitter=cube_jitter)
    obs = env.reset()
    action_queue: deque = deque(maxlen=model.horizon)
    ee_trajectory: List[np.ndarray] = []
    l_early_guided_vals: List[float] = []
    success = False
    last_obs = obs

    for step in range(max_steps):
        ee_trajectory.append(obs[0:3].copy())

        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)

            if guided:
                ee_start = torch.tensor(obs[0:3],   dtype=torch.float32, device=device)
                left_goal = torch.tensor(obs[8:11],  dtype=torch.float32, device=device)
                right_goal = torch.tensor(obs[15:18], dtype=torch.float32, device=device)
                goals_t = torch.stack([left_goal, right_goal])

                act_seq, l_val = sampler.sample(
                    model, obs_t, ee_start, goals_t,
                    n_sampling_steps=n_sampling_steps)
                l_early_guided_vals.append(l_val)
            else:
                act_seq = sampler.sample(model, obs_t,
                                         n_sampling_steps=n_sampling_steps)

            chunk = act_seq[0].cpu().numpy() * act_std + act_mean
            for a in chunk:
                action_queue.append(a)

        action = action_queue.popleft()
        result = env.step(action)
        obs = result.obs
        last_obs = obs
        success = (result.info.get('success_left', 0) > 0.5 or
                   result.info.get('success_right', 0) > 0.5)
        if result.done:
            break

    env.close()

    # Compute L_early_intent from actual EE trajectory
    if len(ee_trajectory) >= 4:
        traj_arr = np.array(ee_trajectory)
        goals_np = np.stack([last_obs[8:11], last_obs[15:18]], axis=0)
        from evaluation.legibility_metrics import compute_legibility
        r0 = compute_legibility(traj_arr, goals_np, true_goal_idx=0, model='gaussian')
        r1 = compute_legibility(traj_arr, goals_np, true_goal_idx=1, model='gaussian')
        if r0.L_early_intent >= r1.L_early_intent:
            l_early, rlc, true_goal = r0.L_early_intent, r0.relative_legibility_cost, 'left'
        else:
            l_early, rlc, true_goal = r1.L_early_intent, r1.relative_legibility_cost, 'right'
    else:
        l_early, rlc, true_goal = 0.0, 0.0, 'unknown'

    return dict(
        success=success,
        steps=step + 1,
        l_early_actual=float(l_early),
        rlc_actual=float(rlc),
        true_goal=true_goal,
        l_early_guided_mean=(float(np.mean(l_early_guided_vals))
                             if l_early_guided_vals else 0.0),
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT,
                    help=f'Path to checkpoint (default: {DEFAULT_CKPT})')
    ap.add_argument('--n_episodes', type=int, default=20,
                    help='Episodes per mode (baseline + guided)')
    ap.add_argument('--guidance_scale', type=float, default=2.0,
                    help='Guidance strength w (0=baseline, ~2=recommended)')
    ap.add_argument('--grad_clip', type=float, default=1.0,
                    help='Gradient norm clip for LPS stability')
    ap.add_argument('--n_sampling_steps', type=int, default=10,
                    help='DDIM steps (default 10, matches eval_with_videos.py)')
    ap.add_argument('--cube_jitter', type=float, default=0.0,
                    help='Cube position jitter (default 0.0 matches eval_with_videos.py)')
    ap.add_argument('--baseline_only', action='store_true',
                    help='Only run baseline (skip guided)')
    ap.add_argument('--guided_only', action='store_true',
                    help='Only run guided (skip baseline)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print("  Training-Free Classifier Guidance for Legibility")
    print(f"{'='*65}")
    print(f"  Device           : {device}")
    print(f"  Checkpoint       : {args.checkpoint}")
    print(f"  Guidance w       : {args.guidance_scale}")
    print(f"  DDIM steps       : {args.n_sampling_steps}")
    print(f"  Cube jitter      : {args.cube_jitter}")
    print(f"  Episodes         : {args.n_episodes} per mode")
    print(f"{'='*65}\n")

    # Load checkpoint (DiffusionPolicy / UNet â€” same as eval_with_videos.py)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']

    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256),
        n_blocks=cfg.get('n_blocks', 3),
    ).to(device)
    model.load_state_dict(ckpt['model'])   # raw weights â€” same as eval_with_videos.py
    model.eval()

    print(f"  Model      : DiffusionPolicy ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Epoch      : {ckpt.get('epoch', '?')}  Loss: {ckpt.get('loss', float('nan')):.6f}")
    print(f"  Horizon    : {cfg['horizon']}  Act-dim: {cfg['act_dim']}")

    obs_mean = np.array(ckpt['obs_mean'], dtype=np.float32)
    obs_std  = np.array(ckpt['obs_std'],  dtype=np.float32)
    act_mean = np.array(ckpt['act_mean'], dtype=np.float32)
    act_std  = np.array(ckpt['act_std'],  dtype=np.float32)

    n_diff = cfg.get('n_diffusion_steps', 100)
    beta_s = cfg.get('beta_start', 1e-4)
    beta_e = cfg.get('beta_end', 0.1)

    # â”€â”€ 1. Baseline â”€â”€ (DDIMSampler, matches eval_with_videos.py exactly)
    baseline_results = []
    if not args.guided_only:
        baseline_sampler = DDIMSampler(n_diff, beta_s, beta_e, device)
        print(f"\nâ”€â”€ BASELINE (standard DDIM, w=0) â”€â”€ {args.n_episodes} episodes â”€â”€")
        for ep in range(args.n_episodes):
            r = run_episode(model, baseline_sampler,
                            obs_mean, obs_std, act_mean, act_std, device,
                            guided=False,
                            n_sampling_steps=args.n_sampling_steps,
                            cube_jitter=args.cube_jitter)
            baseline_results.append(r)
            tick = 'âœ“' if r['success'] else 'âœ—'
            print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  "
                  f"L_early={r['l_early_actual']:.4f}  "
                  f"RLC={r['rlc_actual']:.4f}  "
                  f"goal={r['true_goal']}  steps={r['steps']}")

        bl_l = [r['l_early_actual'] for r in baseline_results]
        bl_s = [r['success']        for r in baseline_results]
        print(f"\n  BASELINE SUMMARY:")
        print(f"    L_early_intent : {np.mean(bl_l):.4f} Â± {np.std(bl_l):.4f}")
        print(f"    Success rate   : {np.mean(bl_s):.1%}")
        print(f"\n    L_early_intent distribution:")
        for thr in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
            frac = np.mean(np.array(bl_l) >= thr)
            bar  = '#' * int(frac * 30)
            print(f"    â‰¥{thr:.2f}: [{bar:<30}] {frac:.0%}")

    # â”€â”€ 2. Guided â”€â”€ (LPSDDIMSampler)
    guided_results = []
    if not args.baseline_only:
        lps_sampler = LPSDDIMSampler(n_diff, beta_s, beta_e, device,
                                     guidance_scale=args.guidance_scale,
                                     grad_clip=args.grad_clip)
        print(f"\nâ”€â”€ GUIDED (LPS, w={args.guidance_scale}) â”€â”€ {args.n_episodes} episodes â”€â”€")
        for ep in range(args.n_episodes):
            r = run_episode(model, lps_sampler,
                            obs_mean, obs_std, act_mean, act_std, device,
                            guided=True,
                            n_sampling_steps=args.n_sampling_steps,
                            cube_jitter=args.cube_jitter)
            guided_results.append(r)
            tick = 'âœ“' if r['success'] else 'âœ—'
            print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  "
                  f"L_early={r['l_early_actual']:.4f}  "
                  f"RLC={r['rlc_actual']:.4f}  "
                  f"L_early_pred={r['l_early_guided_mean']:.4f}  "
                  f"steps={r['steps']}")

        gd_l = [r['l_early_actual'] for r in guided_results]
        gd_s = [r['success']        for r in guided_results]
        print(f"\n  GUIDED SUMMARY (w={args.guidance_scale}):")
        print(f"    L_early_intent : {np.mean(gd_l):.4f} Â± {np.std(gd_l):.4f}")
        print(f"    Success rate   : {np.mean(gd_s):.1%}")

    # â”€â”€ 3. Comparison â”€â”€
    if baseline_results and guided_results:
        bl = np.mean([r['l_early_actual'] for r in baseline_results])
        gd = np.mean([r['l_early_actual'] for r in guided_results])
        delta = gd - bl
        print(f"\n{'='*65}")
        print(f"  LEGIBILITY IMPROVEMENT (LPS over baseline)")
        print(f"{'='*65}")
        print(f"  Baseline L_early  : {bl:.4f}")
        print(f"  Guided   L_early  : {gd:.4f}  Î”={delta:+.4f}  ({delta/max(bl,1e-6)*100:+.1f}%)")
        print(f"  Success â€” Baseline: {np.mean([r['success'] for r in baseline_results]):.1%}  "
              f"LPS: {np.mean([r['success'] for r in guided_results]):.1%}")

    # â”€â”€ 4. Save â”€â”€
    out_dir = Path(__file__).parent.parent / 'outputs'
    out_dir.mkdir(exist_ok=True)

    def _s(v):
        if isinstance(v, (bool, np.bool_)):   return bool(v)
        if isinstance(v, (float, np.floating)): return float(v)
        return v

    results = {
        'checkpoint': str(args.checkpoint),
        'guidance_scale': args.guidance_scale,
        'n_sampling_steps': args.n_sampling_steps,
        'cube_jitter': args.cube_jitter,
        'n_episodes': args.n_episodes,
        'baseline': [{k: _s(v) for k, v in r.items()} for r in baseline_results],
        'guided':   [{k: _s(v) for k, v in r.items()} for r in guided_results],
    }
    out_path = out_dir / 'lps_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved â†’ {out_path}")


if __name__ == '__main__':
    main()
