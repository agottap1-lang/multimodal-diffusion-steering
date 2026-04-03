#!/usr/bin/env python3
"""
Stage 1: VLM-Synthesized Guidance Functions for Diffusion Trajectory Planning
=============================================================================

This script implements the complete Stage 1 pipeline:

1. **Prompt Gemini** to generate a differentiable PyTorch legibility scoring
   function (Eureka-style code synthesis).
2. **Validate** the generated code: syntax, gradient flow, output range.
3. **Unit test** on known trajectories.
4. **Plug into LPSDDIMSampler** and run evaluation.
5. **Compare** against hand-crafted L_early_intent baseline.

Usage
-----
  # Full pipeline: generate + validate + evaluate
  python evaluation/stage1_vlm_guidance.py --api_key YOUR_KEY

  # Skip generation, use cached function
  python evaluation/stage1_vlm_guidance.py --use_cached

  # Evaluate only (function must exist in outputs/vlm_score_fn.py)
  python evaluation/stage1_vlm_guidance.py --eval_only --guidance_scale 10.0
"""

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import textwrap
import time
import traceback
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.twoblockpick_env import TwoBlockPickEnv

# ── re-use model + sampler from eval_legibility_guided ──────────────────
from evaluation.eval_legibility_guided import (
    DDIMSampler,
    DiffusionPolicy,
    LPSDDIMSampler,
    l_early_intent_torch,
)

# ── environment constants ───────────────────────────────────────────────
OBS_EE_POS    = slice(0, 3)
OBS_LEFT_POS  = slice(8, 11)
OBS_RIGHT_POS = slice(15, 18)
ACTION_SCALE  = 0.05
DEFAULT_CKPT  = 'runs/diffusion_20260222_195530/ckpt_ep100.pt'

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ═════════════════════════════════════════════════════════════════════════
# STAGE 1.1: VLM PROMPT FOR CODE SYNTHESIS
# ═════════════════════════════════════════════════════════════════════════

EUREKA_PROMPT = textwrap.dedent("""\
You are an expert robotics researcher specializing in trajectory legibility
and differentiable optimization. Your task is to write a PyTorch scoring
function that measures how "legible" a robot trajectory is — i.e., how
easily a human observer can infer which goal the robot is heading toward
by watching the early part of its motion.

## Task: TwoBlockPick

A Franka Panda robot arm picks up one of two red blocks on a table.
- Left block at approximately (0.50, -0.07, 0.42) in world coordinates
- Right block at approximately (0.50, +0.07, 0.42) in world coordinates
- Robot end-effector (EE) starts at approximately (0.40, 0.0, 0.55)
- The blocks are ~14cm apart (y-axis separation)

An observer watching the robot's early motion should be able to tell
which block the robot intends to pick. A "legible" trajectory curves
TOWARD the intended goal early, creating a clear visual signal.

## Function Signature

```python
def vlm_legibility_score(
    ee_traj: torch.Tensor,      # (H, 3) predicted EE positions (x,y,z)
    goals: torch.Tensor,        # (K, 3) goal positions (K=2 blocks)
    true_goal_idx: int = 0,     # index of committed goal in goals
    early_frac: float = 0.30,   # fraction of trajectory considered "early"
) -> torch.Tensor:
    \"\"\"Return a differentiable scalar score in [0, 1]. Higher = more legible.
    Must support torch.autograd.grad() backpropagation through ee_traj.\"\"\"
```

## Requirements

1. **Differentiable**: No argmax, no if/else on tensor values, no .item(),
   no numpy. Use only PyTorch operations that support autograd.
2. **Multi-criteria**: Combine at least 3 of these geometric cues:
   - Bayesian posterior P(g*|x) using Gaussian proximity
   - Directional commitment (velocity vector alignment with goal direction)
   - Lateral deviation (early motion away from non-goal, toward goal)
   - Approach curvature (trajectory curves toward intended goal)
   - Speed profile (faster approach = more committed)
3. **Auto-calibrated**: Scale parameters from inter-goal distance (no magic numbers)
4. **Numerically stable**: Use log-sum-exp, epsilon floors, torch.clamp
5. **Return scalar**: Single float tensor, higher = more legible, range [0,1]

## Reference: Existing Hand-Crafted Function

This is the current baseline that achieves L_early=0.952 with classifier guidance:

```python
def l_early_intent_torch(ee_traj, goals, true_goal_idx=0, early_frac=0.30):
    H = ee_traj.shape[0]
    K = goals.shape[0]
    early_end = max(1, int(H * early_frac))
    early_traj = ee_traj[:early_end]
    dists = torch.cdist(goals, goals)
    mask = dists > 1e-6
    d_min = dists[mask].min() if mask.any() else torch.tensor(0.14)
    sigma = d_min / (2.0 * math.sqrt(2.0 * math.log(2)))
    diff = early_traj.unsqueeze(1) - goals.unsqueeze(0)
    sq_dist = (diff ** 2).sum(-1)
    log_like = -sq_dist / (2.0 * sigma ** 2)
    posteriors = torch.softmax(log_like, dim=-1)
    l_early = posteriors[:, true_goal_idx].mean()
    return l_early
```

## Your Goal

Write a BETTER function that captures richer geometric cues beyond simple
Gaussian proximity. The function should reward trajectories that:
1. Move toward the goal EARLY (first 30%)
2. Show clear DIRECTIONAL commitment (velocity aligned with goal)
3. Create LATERAL separation from the non-goal
4. Have smooth, purposeful curvature toward the goal

Output ONLY the Python function code, starting with `def vlm_legibility_score(`.
No markdown fences, no imports (torch and math are available), no extra text.
""")


# ═════════════════════════════════════════════════════════════════════════
# STAGE 1.2: CALL GEMINI TO GENERATE CODE
# ═════════════════════════════════════════════════════════════════════════

def call_gemini_for_code(api_key: str, model: str = "gemini-2.5-flash",
                         max_retries: int = 3) -> str:
    """Call Gemini to generate the legibility scoring function."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        print(f"\n  [Gemini] Attempt {attempt+1}/{max_retries} ...")
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=4096),
            temperature=0.2 + attempt * 0.2,  # increase creativity on retries
        )
        try:
            resp = client.models.generate_content(
                model=model,
                contents=EUREKA_PROMPT,
                config=config,
            )
            raw_text = resp.text
            # Extract code — handle markdown fences if present
            code = extract_python_code(raw_text)
            if code and "def vlm_legibility_score(" in code:
                print(f"  [Gemini] Got valid function ({len(code)} chars)")
                return code
            else:
                print(f"  [Gemini] Response didn't contain valid function signature")
                print(f"  [Gemini] Raw preview: {raw_text[:200]}...")
        except Exception as e:
            print(f"  [Gemini] API error: {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError("Failed to get valid code from Gemini after all retries")


def extract_python_code(text: str) -> str:
    """Extract Python code from VLM response, handling markdown fences."""
    # Try to find code in markdown fences first
    fence_match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find the function definition directly
    fn_match = re.search(r'(def vlm_legibility_score\(.*)', text, re.DOTALL)
    if fn_match:
        return fn_match.group(1).strip()

    return text.strip()


# ═════════════════════════════════════════════════════════════════════════
# STAGE 1.3: VALIDATE GENERATED CODE
# ═════════════════════════════════════════════════════════════════════════

def validate_generated_code(code: str, device: torch.device) -> Tuple[bool, str, object]:
    """Validate the generated scoring function.

    Returns (success, message, function_object_or_None).
    """
    # 1. Syntax check
    try:
        compile(code, "<vlm_score>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}", None

    # 2. Execute in sandbox
    sandbox = {"torch": torch, "math": math, "nn": nn}
    try:
        exec(code, sandbox)
    except Exception as e:
        return False, f"Execution error: {e}", None

    fn = sandbox.get("vlm_legibility_score")
    if fn is None:
        return False, "Function 'vlm_legibility_score' not found in code", None

    # 3. Test basic call with gradient
    try:
        H, K = 32, 2
        ee_traj = torch.randn(H, 3, device=device, requires_grad=True)
        goals = torch.tensor([[0.50, -0.07, 0.42],
                               [0.50,  0.07, 0.42]], device=device)
        score = fn(ee_traj, goals, true_goal_idx=0, early_frac=0.30)

        # Check output properties
        if not isinstance(score, torch.Tensor):
            return False, f"Output is {type(score)}, expected torch.Tensor", None
        if score.ndim != 0:
            return False, f"Output has {score.ndim} dims, expected scalar (0-d)", None
        if not score.requires_grad:
            return False, "Output doesn't have grad — gradient won't flow", None

        # Check gradient exists
        grad = torch.autograd.grad(score, ee_traj, retain_graph=True)[0]
        if grad is None or grad.abs().sum() < 1e-12:
            return False, "Gradient is zero — no signal for guidance", None

        val = score.item()
        if not (0.0 <= val <= 1.0 + 0.01):
            return False, f"Score {val:.4f} outside expected [0, 1] range", None

    except Exception as e:
        return False, f"Runtime test failed: {e}\n{traceback.format_exc()}", None

    # 4. Test with "obviously legible" vs "ambiguous" trajectory
    try:
        goals = torch.tensor([[0.50, -0.07, 0.42],
                               [0.50,  0.07, 0.42]], device=device)

        # Legible: straight toward left goal
        legible = torch.zeros(32, 3, device=device, requires_grad=True)
        for i in range(32):
            frac = i / 31.0
            legible.data[i] = torch.tensor([
                0.40 + frac * 0.10,
                0.0 - frac * 0.07,
                0.55 - frac * 0.13
            ])

        # Ambiguous: goes straight down center
        ambiguous = torch.zeros(32, 3, device=device, requires_grad=True)
        for i in range(32):
            frac = i / 31.0
            ambiguous.data[i] = torch.tensor([
                0.40 + frac * 0.10,
                0.0,
                0.55 - frac * 0.13
            ])

        s_leg = fn(legible, goals, true_goal_idx=0, early_frac=0.30)
        s_amb = fn(ambiguous, goals, true_goal_idx=0, early_frac=0.30)

        print(f"    Legible score  : {s_leg.item():.4f}")
        print(f"    Ambiguous score: {s_amb.item():.4f}")

        if s_leg.item() <= s_amb.item():
            return False, (f"Legible ({s_leg.item():.4f}) should score HIGHER "
                         f"than ambiguous ({s_amb.item():.4f})"), None

    except Exception as e:
        return False, f"Discrimination test failed: {e}", None

    return True, "All validations passed", fn


# ═════════════════════════════════════════════════════════════════════════
# STAGE 1.4: VLM-GUIDED DDIM SAMPLER
# ═════════════════════════════════════════════════════════════════════════

class VLMGuidedDDIMSampler:
    """DDIM with VLM-generated guidance function.

    Same as LPSDDIMSampler but uses the VLM-generated scoring function
    instead of l_early_intent_torch.
    """

    def __init__(self, n_steps, beta_start, beta_end, device,
                 score_fn, guidance_scale: float = 10.0,
                 grad_clip: float = 1.0):
        self.device = device
        self.score_fn = score_fn
        self.guidance_scale = guidance_scale
        self.grad_clip = grad_clip
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def sample(
        self,
        model: DiffusionPolicy,
        obs: torch.Tensor,
        ee_pos_start: torch.Tensor,
        goals: torch.Tensor,
        n_sampling_steps: int = 10,
    ) -> Tuple[torch.Tensor, float]:
        B = obs.shape[0]
        H, A = model.horizon, model.act_dim
        x = torch.randn(B, H, A, device=self.device)
        timesteps = torch.linspace(
            0, len(self.alphas_cumprod) - 1, n_sampling_steps,
            device=self.device
        ).long()
        timesteps = torch.flip(timesteps, [0])

        final_score = 0.0

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (self.alphas_cumprod[timesteps[i + 1]]
                          if i < len(timesteps) - 1
                          else torch.tensor(1.0, device=self.device))
            sqrt_ab = torch.sqrt(alpha_t)
            sqrt_1m_ab = torch.sqrt(1.0 - alpha_t)

            # Gradient computation
            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                eps_pred = model(x_in, t_batch, obs)
                x0_pred = (x_in - sqrt_1m_ab * eps_pred) / sqrt_ab

                # Forward kinematics: delta → EE trajectory
                delta_pos = x0_pred[0, :, :3] * ACTION_SCALE
                ee_traj = torch.cumsum(delta_pos, dim=0) + ee_pos_start

                # Infer committed goal (detached)
                with torch.no_grad():
                    s0 = self.score_fn(ee_traj.detach(), goals, 0).item()
                    s1 = self.score_fn(ee_traj.detach(), goals, 1).item()
                true_goal = 0 if s0 >= s1 else 1

                score = self.score_fn(ee_traj, goals, true_goal)
                grad = torch.autograd.grad(score, x_in)[0]

            final_score = float(score.item())

            with torch.no_grad():
                g = grad.detach()
                gn = g.norm()
                if gn > self.grad_clip:
                    g = g * (self.grad_clip / (gn + 1e-8))

                guided_eps = eps_pred.detach() - self.guidance_scale * sqrt_1m_ab * g
                x0_guided = (x - sqrt_1m_ab * guided_eps) / sqrt_ab

                if i < len(timesteps) - 1:
                    x = torch.sqrt(alpha_prev) * x0_guided + torch.sqrt(1.0 - alpha_prev) * guided_eps
                else:
                    x = x0_guided

        return x, final_score


# ═════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER  (same as eval_legibility_guided.py)
# ═════════════════════════════════════════════════════════════════════════

def run_episode(
    model, sampler, obs_mean, obs_std, act_mean, act_std, device,
    guided: bool = False, n_sampling_steps: int = 10,
    cube_jitter: float = 0.0, max_steps: int = 400,
) -> dict:
    env = TwoBlockPickEnv(render=False, episode_length=max_steps,
                          cube_jitter=cube_jitter)
    obs = env.reset()
    action_queue: deque = deque(maxlen=model.horizon)
    ee_trajectory: List[np.ndarray] = []
    guided_score_vals: List[float] = []
    success = False
    last_obs = obs

    for step in range(max_steps):
        ee_trajectory.append(obs[0:3].copy())

        if len(action_queue) == 0:
            obs_norm = (obs - obs_mean) / obs_std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)

            if guided:
                ee_start = torch.tensor(obs[0:3], dtype=torch.float32, device=device)
                left_goal = torch.tensor(obs[8:11], dtype=torch.float32, device=device)
                right_goal = torch.tensor(obs[15:18], dtype=torch.float32, device=device)
                goals_t = torch.stack([left_goal, right_goal])

                act_seq, s_val = sampler.sample(
                    model, obs_t, ee_start, goals_t,
                    n_sampling_steps=n_sampling_steps)
                guided_score_vals.append(s_val)
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

    # Compute actual L_early from executed trajectory
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
        guided_score_mean=(float(np.mean(guided_score_vals))
                           if guided_score_vals else 0.0),
    )


# ═════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--api_key', type=str, default=None,
                    help='Gemini API key (or set GEMINI_API_KEY / GOOGLE_API_KEY)')
    ap.add_argument('--model', type=str, default='gemini-2.5-flash',
                    help='Gemini model name')
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--n_episodes', type=int, default=20)
    ap.add_argument('--guidance_scale', type=float, default=10.0)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--n_sampling_steps', type=int, default=10)
    ap.add_argument('--cube_jitter', type=float, default=0.0)
    ap.add_argument('--use_cached', action='store_true',
                    help='Use cached VLM function from outputs/vlm_score_fn.py')
    ap.add_argument('--eval_only', action='store_true',
                    help='Skip generation, evaluate existing function')
    ap.add_argument('--skip_baseline', action='store_true',
                    help='Skip baseline evaluation')
    ap.add_argument('--skip_handcrafted', action='store_true',
                    help='Skip hand-crafted LPS evaluation')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(__file__).parent.parent / 'outputs' / 'stage1'
    out_dir.mkdir(parents=True, exist_ok=True)
    fn_path = out_dir / 'vlm_score_fn.py'

    print(f"\n{'='*70}")
    print("  Stage 1: VLM-Synthesized Guidance Functions")
    print(f"{'='*70}")
    print(f"  Device        : {device}")
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Guidance w    : {args.guidance_scale}")
    print(f"  Episodes      : {args.n_episodes}")
    print(f"{'='*70}\n")

    # ── Step 1: Get/Generate the scoring function ────────────────────
    vlm_score_fn = None

    if args.eval_only or args.use_cached:
        if fn_path.exists():
            print(f"  Loading cached function from {fn_path}")
            code = fn_path.read_text(encoding='utf-8')
        else:
            print(f"  ERROR: No cached function at {fn_path}")
            sys.exit(1)
    else:
        # Call Gemini
        api_key = (args.api_key
                   or os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GOOGLE_API_KEY"))
        if not api_key:
            print("  ERROR: No API key. Set --api_key, GEMINI_API_KEY, or GOOGLE_API_KEY")
            sys.exit(1)

        print("  Step 1: Calling Gemini to generate legibility scoring function...")
        code = call_gemini_for_code(api_key, model=args.model)

        # Save generated code
        fn_path.write_text(code, encoding='utf-8')
        print(f"  Saved generated code → {fn_path}")

    # ── Step 2: Validate ─────────────────────────────────────────────
    print("\n  Step 2: Validating generated function...")
    success, msg, vlm_score_fn = validate_generated_code(code, device)

    if not success:
        print(f"\n  VALIDATION FAILED: {msg}")
        print("  Attempting auto-fix...")

        # Try to fix common issues and re-validate
        api_key = (args.api_key
                   or os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GOOGLE_API_KEY"))
        if api_key and not args.eval_only:
            fix_prompt = (
                f"The following PyTorch function has an error:\n\n"
                f"```python\n{code}\n```\n\n"
                f"Error: {msg}\n\n"
                f"Fix the function. It must:\n"
                f"- Be named vlm_legibility_score(ee_traj, goals, true_goal_idx=0, early_frac=0.30)\n"
                f"- Return a differentiable scalar tensor in [0,1]\n"
                f"- Support torch.autograd.grad()\n"
                f"- Score 'obviously legible' trajectories HIGHER than ambiguous ones\n\n"
                f"Output ONLY the corrected function code, no markdown."
            )
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            for retry in range(2):
                print(f"\n  [Fix attempt {retry+1}/2] ...")
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=4096),
                    temperature=0.3,
                )
                try:
                    resp = client.models.generate_content(
                        model=args.model, contents=fix_prompt, config=config
                    )
                    code = extract_python_code(resp.text)
                    success, msg, vlm_score_fn = validate_generated_code(code, device)
                    if success:
                        fn_path.write_text(code, encoding='utf-8')
                        print(f"  Fix succeeded! Saved → {fn_path}")
                        break
                    else:
                        print(f"  Fix attempt failed: {msg}")
                        fix_prompt = (
                            f"This STILL has an error:\n\n```python\n{code}\n```\n\n"
                            f"Error: {msg}\n\nFix it. Output ONLY the function code."
                        )
                except Exception as e:
                    print(f"  Fix API error: {e}")

        if not success:
            print(f"\n  FATAL: Could not generate valid function after all attempts.")
            print(f"  Last error: {msg}")
            sys.exit(1)

    print(f"\n  ✓ Validation passed: {msg}")

    # ── Step 3: Load model ───────────────────────────────────────────
    print("\n  Step 3: Loading diffusion policy...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']

    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'],
        act_dim=cfg['act_dim'],
        horizon=cfg['horizon'],
        hidden_dim=cfg.get('hidden_dim', 256),
        n_blocks=cfg.get('n_blocks', 3),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    obs_mean = np.array(ckpt['obs_mean'], dtype=np.float32)
    obs_std  = np.array(ckpt['obs_std'],  dtype=np.float32)
    act_mean = np.array(ckpt['act_mean'], dtype=np.float32)
    act_std  = np.array(ckpt['act_std'],  dtype=np.float32)

    n_diff = cfg.get('n_diffusion_steps', 100)
    beta_s = cfg.get('beta_start', 1e-4)
    beta_e = cfg.get('beta_end', 0.1)

    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Step 4: Evaluate ─────────────────────────────────────────────
    all_results = {}

    # 4a: Baseline (unguided DDIM)
    if not args.skip_baseline:
        baseline_sampler = DDIMSampler(n_diff, beta_s, beta_e, device)
        print(f"\n── BASELINE (DDIM, w=0) ── {args.n_episodes} episodes ──")
        baseline_results = []
        for ep in range(args.n_episodes):
            r = run_episode(model, baseline_sampler,
                            obs_mean, obs_std, act_mean, act_std, device,
                            guided=False,
                            n_sampling_steps=args.n_sampling_steps,
                            cube_jitter=args.cube_jitter)
            baseline_results.append(r)
            tick = '✓' if r['success'] else '✗'
            print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  "
                  f"L_early={r['l_early_actual']:.4f}  "
                  f"goal={r['true_goal']}  steps={r['steps']}")

        bl_l = [r['l_early_actual'] for r in baseline_results]
        bl_s = [r['success'] for r in baseline_results]
        print(f"\n  BASELINE: success={np.mean(bl_s):.1%}  "
              f"L_early={np.mean(bl_l):.4f} ± {np.std(bl_l):.4f}")
        all_results['baseline'] = baseline_results

    # 4b: Hand-crafted LPS
    if not args.skip_handcrafted:
        lps_sampler = LPSDDIMSampler(n_diff, beta_s, beta_e, device,
                                     guidance_scale=args.guidance_scale,
                                     grad_clip=args.grad_clip)
        print(f"\n── HAND-CRAFTED LPS (w={args.guidance_scale}) ── {args.n_episodes} episodes ──")
        handcrafted_results = []
        for ep in range(args.n_episodes):
            r = run_episode(model, lps_sampler,
                            obs_mean, obs_std, act_mean, act_std, device,
                            guided=True,
                            n_sampling_steps=args.n_sampling_steps,
                            cube_jitter=args.cube_jitter)
            handcrafted_results.append(r)
            tick = '✓' if r['success'] else '✗'
            print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  "
                  f"L_early={r['l_early_actual']:.4f}  "
                  f"score_pred={r['guided_score_mean']:.4f}  "
                  f"steps={r['steps']}")

        hc_l = [r['l_early_actual'] for r in handcrafted_results]
        hc_s = [r['success'] for r in handcrafted_results]
        print(f"\n  HAND-CRAFTED: success={np.mean(hc_s):.1%}  "
              f"L_early={np.mean(hc_l):.4f} ± {np.std(hc_l):.4f}")
        all_results['handcrafted_lps'] = handcrafted_results

    # 4c: VLM-generated guidance
    vlm_sampler = VLMGuidedDDIMSampler(n_diff, beta_s, beta_e, device,
                                        score_fn=vlm_score_fn,
                                        guidance_scale=args.guidance_scale,
                                        grad_clip=args.grad_clip)
    print(f"\n── VLM-GENERATED GUIDANCE (w={args.guidance_scale}) ── {args.n_episodes} episodes ──")
    vlm_results = []
    for ep in range(args.n_episodes):
        r = run_episode(model, vlm_sampler,
                        obs_mean, obs_std, act_mean, act_std, device,
                        guided=True,
                        n_sampling_steps=args.n_sampling_steps,
                        cube_jitter=args.cube_jitter)
        vlm_results.append(r)
        tick = '✓' if r['success'] else '✗'
        print(f"  Ep {ep+1:>2}/{args.n_episodes} {tick}  "
              f"L_early={r['l_early_actual']:.4f}  "
              f"vlm_score={r['guided_score_mean']:.4f}  "
              f"steps={r['steps']}")

    vl_l = [r['l_early_actual'] for r in vlm_results]
    vl_s = [r['success'] for r in vlm_results]
    print(f"\n  VLM-GUIDED: success={np.mean(vl_s):.1%}  "
          f"L_early={np.mean(vl_l):.4f} ± {np.std(vl_l):.4f}")
    all_results['vlm_guided'] = vlm_results

    # ── Step 5: Summary comparison ───────────────────────────────────
    print(f"\n{'='*70}")
    print("  STAGE 1 RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'Success':>8} {'L_early':>10} {'L_early_std':>12}")
    print(f"  {'-'*55}")

    for name, results in all_results.items():
        l_vals = [r['l_early_actual'] for r in results]
        s_vals = [r['success'] for r in results]
        print(f"  {name:<25} {np.mean(s_vals):>7.1%} {np.mean(l_vals):>10.4f} "
              f"{np.std(l_vals):>12.4f}")

    # Compute improvement
    if 'baseline' in all_results and 'vlm_guided' in all_results:
        bl = np.mean([r['l_early_actual'] for r in all_results['baseline']])
        vl = np.mean([r['l_early_actual'] for r in all_results['vlm_guided']])
        delta = vl - bl
        print(f"\n  VLM vs Baseline: Δ L_early = {delta:+.4f} ({delta/max(bl,1e-6)*100:+.1f}%)")

    if 'handcrafted_lps' in all_results and 'vlm_guided' in all_results:
        hc = np.mean([r['l_early_actual'] for r in all_results['handcrafted_lps']])
        vl = np.mean([r['l_early_actual'] for r in all_results['vlm_guided']])
        delta = vl - hc
        print(f"  VLM vs Hand-crafted: Δ L_early = {delta:+.4f} ({delta/max(hc,1e-6)*100:+.1f}%)")

    # ── Save results ─────────────────────────────────────────────────
    def _s(v):
        if isinstance(v, (bool, np.bool_)):   return bool(v)
        if isinstance(v, (float, np.floating)): return float(v)
        return v

    output = {
        'checkpoint': str(args.checkpoint),
        'guidance_scale': args.guidance_scale,
        'n_sampling_steps': args.n_sampling_steps,
        'n_episodes': args.n_episodes,
        'vlm_function_path': str(fn_path),
    }
    for name, results in all_results.items():
        l_vals = [r['l_early_actual'] for r in results]
        s_vals = [r['success'] for r in results]
        output[name] = {
            'success_rate': float(np.mean(s_vals)),
            'l_early_mean': float(np.mean(l_vals)),
            'l_early_std': float(np.std(l_vals)),
            'episodes': [{k: _s(v) for k, v in r.items()} for r in results],
        }

    results_path = out_dir / 'stage1_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {results_path}")
    print(f"  VLM function saved → {fn_path}")


if __name__ == '__main__':
    main()
