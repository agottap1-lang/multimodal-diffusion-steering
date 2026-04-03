# Comprehensive Research & Engineering Audit Prompt

> **Copy everything below this line into a fresh chat with Claude Opus 4.6 (or equivalent frontier model). Attach the full codebase as context.**

---

## PROMPT

You are a **senior machine learning and robotics research scientist** with deep expertise in diffusion policies, vision-language models, Bayesian intent inference, legibility theory, and human-robot interaction. You have published at ICRA, RSS, CoRL, and NeurIPS. You are advising a graduate student who is building a system to make robot motion **legible** — meaning a human observer can infer the robot's intended goal early in the trajectory — using a **VLM-in-the-loop diffusion policy** for a tabletop manipulation task. The long-term vision is to generalize this methodology to **Vision-Language-Action (VLA) models** and other policy architectures beyond the current testbed.

---

### 1. THE GOAL (one sentence)

**We aim to make a diffusion policy's generated robot trajectories *legible* — meaning a human observer can correctly identify which of two possible target objects the robot intends to pick up within the first 30–40% of the trajectory — by using a Vision-Language Model (Gemini) as a real-time legibility evaluator that steers action selection during closed-loop execution, without retraining the policy.**

---

### 2. COMPLETE SYSTEM DESCRIPTION — Read every file, miss nothing

The codebase implements the following end-to-end pipeline. Read every module described below and understand the data flow, the mathematical assumptions, and the experimental protocol.

#### 2.1 Environment: `envs/twoblockpick_env.py`
- **PyBullet** simulation with a **Franka Panda** arm and **two identical red cubes** on a table, one at y=+0.07m ("left") and one at y=−0.07m ("right"), with ±0.015m jitter.
- **Observation** (22-d): `ee_pos(3) + ee_quat(4) + gripper(1) + left_cube_pos(3) + left_cube_quat(4) + right_cube_pos(3) + right_cube_quat(4)`
- **Action** (5-d): `[dx, dy, dz, dyaw, grip]` each in `[-1, 1]`, scaled by `action_scale_pos=0.05m` and `action_scale_yaw=15°`.
- **Success**: cube lifted above z=0.52m (table at z=0.40, threshold +0.12m).
- Cubes are **visually identical** (both red) — the only distinguishing cue for an observer is the **trajectory shape** (spatial path of the end-effector).

#### 2.2 Training: `scripts/train.py` + `configs/train.yaml`
- **Architecture**: MLP-based U-Net with sinusoidal time embeddings, GroupNorm, Mish activations, skip connections. 3 encoder blocks, 1 bottleneck, 3 decoder blocks. Hidden dim 256. ~2.3M parameters.
- **Diffusion**: DDPM with 100 steps, β from 0.0001 to 0.1 (ensuring ᾱ_T ≈ 0). Predicts **noise** (not x₀). EMA model (decay 0.999) used for evaluation.
- **Data**: Demonstrations collected with Bézier-curve scripted policies. Each demo is a full episode with horizon H=32 action chunks. Z-score normalization for both observations and actions using global statistics.
- **Training details**: Batch size 256, AdamW (lr=1e-4), gradient clipping at 1.0, mixed precision (AMP), 500 epochs.

#### 2.3 Inference / Sampling: `scripts/eval_with_videos.py` → `DDIMSampler`
- **DDIM** sampling with 10 denoising steps (subsampled from 100 training steps).
- Supports `temperature` scaling and `initial_noise` injection for diversity.
- At each replanning step: observe, normalize, sample H=32 action chunk, denormalize, execute first `n_action_steps=8` actions, repeat.

#### 2.4 VLM Client: `scripts/vlm_client.py` → `LegibilityScorer`
- Wraps an external `gemini_vlm_eval` package with its own `GeminiClient`.
- Uses **Gemini 2.5 Flash** (configurable) via the Google GenAI API.
- **Prompt** (`gemini_vlm_eval/src/gemini_vlm_eval/prompt.py`): Asks the VLM to estimate `P(Goal A | frames)` and `P(Goal B | frames)` given either a single frame or a prefix of frames (ordered t=0..t_now). Outputs JSON with `pA, pB, cue, legible`.
- **Two modes**: `single_frame` (one snapshot) and `prefix_frames` (chronological sequence of annotated frames with timestamps).
- **Legibility score** = `max(pA, pB)` — higher means more confident goal inference by the VLM.
- Also supports `progressive` scoring: early (30%) frame weighted 60%, full (100%) frame weighted 40%, with consistency bonus.

#### 2.5 VLM-Guided Steering — Three Approaches Implemented

**Approach A: Reranking with Goal-Locked Noise Perturbation** (`scripts/vlm_guided_policy.py`, `scripts/vlm_guided_policy_goal_locked.py`)
- At each replanning step: (1) sample baseline trajectory, (2) determine target block from early dy-sign, (3) generate N-1 goal-locked variants by perturbing the action sequence in the lateral (Y) dimension with a Gaussian arc mask (peak at t=0.5), preserving early dy direction and endpoint, (4) visualize all N candidates as overlay images, (5) query VLM batch for legibility scores, (6) execute the most legible candidate.
- **Key mechanism**: Goal-locking ensures all candidates reach the same block; only the *arc style* (how much lateral curvature) varies. VLM selects the most readable arc.

**Approach B: Research-Backed Bézier Arc Warping** (`scripts/vlm_guided_policy_research.py`)
- Generates baseline trajectory from diffusion policy to determine goal/endpoint.
- Post-hoc warps trajectory using **Bézier curves**: `B(t) = (1-t)²P₀ + 2(1-t)t·P₁ + t²·P₂` with control point `P₁` pulled laterally by `arc_magnitude ∈ [0.05, 0.10, 0.15, 0.20, 0.25]m` toward the target block.
- Blends warped trajectory with baseline using a Gaussian warp profile (strongest in middle, zero at endpoints).
- VLM reranks different arc magnitudes.

**Approach C: Progressive MPC-Style Steering** (`evaluation/vlm_progressive_steering.py`) — **THE MAIN EXPERIMENT**
- **Condition 1 (Baseline)**: Run raw diffusion policy for 5 seconds (150 steps at 30 Hz). At t=1,2,3,5 seconds, capture frame, score with VLM passively. No trajectory modification.
- **Condition 2 (VLM Progressive Steering)**: At each checkpoint t ∈ {1s, 2s, 3s}:
  1. Score current trajectory prefix with VLM.
  2. If legibility ≥ θ (default 0.60): continue normally.
  3. If legibility < θ: **MPC-style look-ahead intervention**:
     - Save full PyBullet physics state (`p.saveState`).
     - Sample K candidate action chunks.
     - For each: restore state → simulate 2 seconds forward → capture hypothetical frames at each second boundary.
     - Score `[accumulated_real_frames + hypothetical_future_frames]` with VLM in `prefix_frames` mode.
     - Select the candidate with highest VLM legibility.
     - Restore state → execute only the first 30 steps (1 second) of the winner.
  - **Critical insight**: At t=1s the robot has barely moved — all candidates look alike from the camera. By simulating 2s ahead (to t=3s), the VLM can discriminate trajectories heading toward different blocks. The steering decision is made EARLY (at t=1), but the evaluation uses predicted future frames.
- **Target block enforcement**: Early actions have their Y-sign forced to match the intended block direction (`enforce_block_direction`).
- Uses `prefix_frames` VLM mode so the VLM sees the chronological progression.
- Metrics: progressive legibility curve, early-intent detection rate (% legible by t=1,2,3), interventions per episode, success rate, arc magnitude (emergent).

#### 2.6 Legibility Metrics: `evaluation/legibility_metrics.py` (1,244 lines of rigorous math)
- **Bayesian posterior**: $P(g | \xi_{0:t}) = L(g; \xi_{0:t}) P(g) / \sum_{g'} L(g'; \xi_{0:t}) P(g')$ with uniform prior.
- **Two observer models**:
  1. **Gaussian (IPF, Shi et al. 2025)**: $P(x|g) \propto \exp(-\|x-g\|^2 / 2\sigma^2)$ with auto-calibrated $\sigma = d_{\min} / (2\sqrt{2\ln 2})$.
  2. **Cost-Ratio (Dragan et al. HRI 2013)**: $P(\xi_{0:t}|g) \propto \exp[\beta(C^*(x_0,g) - C(\xi_{0:t}) - C^*(x_t,g))]$.
- **Nine named scores**: L_posterior, L_ipf, L_entropy_auc, L_intent_info_rate, L_half_time, L_early_intent, L_commitment, L_geometric, L_composite.
- **Frenet–Serret analysis**: Discrete curvature via Menger formula, goal-relative curvature $\kappa_g(t) = \kappa(t) \cos\angle(\mathbf{N}(t), g^* - x_t)$.
- **Composite score**: Weighted combination (posterior 25%, early_intent 25%, commitment 20%, geometric 15%, entropy_auc 10%, info_rate 5%).
- Fully **task-agnostic**: no hardcoded thresholds, auto-calibrates from goal geometry.

#### 2.7 Results (50-rollout experiment, `outputs/vlm_progressive_50/`)

| Metric | Baseline | VLM Steered |
|---|---|---|
| N rollouts | 50 | 50 |
| Success rate | 100% | 100% |
| Mean arc (m) | 0.0784 | 0.0749 |
| Arc 15-19 | 2% | 2% |
| Legibility @ t=1s | 0.568 | 0.536 |
| Legibility @ t=2s | 0.565 | 0.550 |
| Legibility @ t=3s | 0.690 | 0.584 |
| Legibility @ t=5s | 0.678 | 0.602 |
| Early intent (by t=1s) | 20% | 10% |
| Early intent (by t=2s) | 16% | 14% |
| Early intent (by t=3s) | 42% | 18% |
| Mean interventions | 0 | 2.58 |
| Total VLM calls | 200 | 845 |

**The steered condition performs WORSE than baseline on every legibility metric.** Steering is actively intervening (2.58 mean interventions per episode, 845 VLM calls vs 200 for passive scoring) but is degrading legibility rather than improving it.

---

### 3. QUESTIONS I NEED YOU TO ANSWER — BE THOROUGH, CITE PAPERS, SHOW EDGE CASES

#### 3.1 Are we utilizing the in-context learning capabilities of the VLM correctly?

Examine specifically:
- The **prompt engineering** in `gemini_vlm_eval/prompt.py`: Is it structured to leverage Gemini's multimodal reasoning? Are we providing enough visual context? Should we use chain-of-thought? Should we provide few-shot examples of legible vs. ambiguous trajectories?
- The **`prefix_frames` mode**: Are we sending frames in the right order, at the right resolution, with the right annotations? The current annotations are just "t = Xs" timestamps and goal legends. Should we overlay trajectory traces, block labels, or other visual aids?
- The **scoring interpretation**: We use `max(pA, pB)` as the legibility score. Is this the right proxy for human-perceived legibility? What about calibration — is the VLM's probability well-calibrated for this task?
- The **prompt itself** asks for probabilities. Should we instead ask "which block is the robot going to pick?" with a confidence scale? Or use chain-of-thought "First describe what you see the robot doing, then estimate probabilities"?
- Why does the VLM return 0.50/0.50 so often at early timesteps? Is this a limitation of the visual input (robot hasn't moved enough) or a prompting failure?

#### 3.2 Why is steering making things WORSE?

The results show baseline legibility is HIGHER than steered legibility across all timepoints. Diagnose:
- Is the look-ahead simulation (2s forward) producing visually unrealistic frames that confuse the VLM?
- Is `enforce_block_direction` corrupting the natural diffusion policy trajectories?
- Are the K candidates too similar to each other (all drawn from the same policy, same observation)?
- Is the VLM's ranking unreliable — i.e., it picks candidates that look legible to it but not to humans?
- Is there a **selection bias** in the evaluation: the baseline is scored passively on the trajectory that actually executed, while the steered condition is scored on an *intervened* trajectory that was chosen by a different VLM call?
- Is the intervention itself mechanically problematic (save/restore state, seed management, chunk boundary alignment)?

#### 3.3 How are we measuring legibility? (one sentence, then elaborate)

**One sentence**: We measure legibility as the VLM's confidence in identifying the intended goal from a time-ordered sequence of annotated trajectory frames scored at 1-second checkpoints during execution — operationalized as `max(P(Goal A), P(Goal B))` where probabilities are estimated by Gemini from visual observations alone.

Now elaborate: Is this a valid measure? Compare to:
- **Dragan et al. (2013)**: Legibility = $\int_0^T f(t) \cdot P(g^* | \xi_{0:t}) dt$ with Boltzmann-rational observer model. We have this in `legibility_metrics.py` but we're NOT using it for steering — we're using the VLM instead.
- **Human studies**: The gold standard. We have no human evaluation.
- **Shi et al. (2025)**: Information Potential Field. We have this implemented but not connected to steering.
- Are the legibility_metrics.py scores and the VLM scores correlated? We haven't measured this.
- The VLM is a **proxy** for human perception. How good is this proxy? What does the literature say about VLMs as human-perception surrogates for motion legibility?

#### 3.4 Would RL solve our problem? SAC? DPO/preference-based methods?

My professor suggested preference-based approaches. Evaluate:
- **SAC (Soft Actor-Critic)**: Could we train a legibility reward function and fine-tune the policy with RL? What would the reward be — the VLM score at each step? The analytical legibility metrics? What are the sample efficiency concerns?
- **DPO (Direct Preference Optimization)**: Could we collect preference pairs (trajectory A more legible than trajectory B, as judged by VLM or humans) and do preference-based fine-tuning of the diffusion policy? How does DPO apply to diffusion models — see DPPO (Diffusion Policy Policy Optimization) and related work?
- **RLHF for robotics**: Relevant papers: "Learning from Human Feedback for Robot Manipulation" (various 2023-2025). How do they handle the sparse reward / long horizon problem?
- **Reward hacking**: If we use VLM as the reward model for RL, will the policy learn to "trick" the VLM rather than be genuinely legible?
- **Comparison**: Reranking (current approach) vs. RL fine-tuning vs. classifier-free guidance vs. gradient-based guidance. Which is most appropriate for our setting?

#### 3.5 Input/Output of the Policy Pipeline — Step by Step

**Without steering:**
1. **Input**: Raw observation from PyBullet (22-d state vector).
2. **Normalize**: `obs_norm = (obs - obs_mean) / obs_std` using global training data statistics.
3. **DDIM Sample**: Start from Gaussian noise `x_T ~ N(0,I)` of shape `(1, 32, 5)`, denoise for 10 steps, get `x_0` = normalized action sequence `(1, 32, 5)`.
4. **Denormalize**: `action_real = x_0 * act_std + act_mean` → real-valued delta actions.
5. **Execute**: Take first 8 actions from the 32-step chunk, apply to environment.
6. **Repeat** from step 1 with new observation.

**With progressive VLM steering:**
1. Steps 1-5 as above for segment 0 (t=0→1s, 30 steps).
2. At t=1s: Render frame, annotate with timestamp, send [t=0, t=1] frames to VLM.
3. VLM returns `{pA, pB, cue, legible}`. Compute `legibility_score = max(pA, pB)`.
4. If `legibility_score >= 0.60`: continue normally (steps 1-5 for next segment).
5. If `legibility_score < 0.60` → **Intervene**:
   a. `p.saveState()` — checkpoint full physics simulation.
   b. For k=1..K: restore state, sample new action chunk, `enforce_block_direction`, simulate 60 steps (2s) forward, capture frames at each second boundary.
   c. Send `[accumulated_real_frames + hypothetical_frames]` to VLM for each candidate.
   d. Select candidate with highest VLM legibility.
   e. Restore state, execute only first 30 steps of winner.
6. Repeat at t=2s, t=3s checkpoints.
7. After t=3s: execute normally without VLM scoring until task completion.
8. Final VLM scoring at t=5s with all 6 frames.

#### 3.6 Inspect the gemini_vlm_eval results: what is observed?

Go through the actual result files in `outputs/vlm_progressive_50/`:
- Many episodes show legibility = 0.50 at t=1s and t=2s (VLM is maximally uncertain). This means the VLM **cannot tell** which block the robot is targeting from early frames.
- Some episodes jump from 0.50 to 0.99 at t=5s (VLM becomes certain only when the gripper is near/touching the block). This is **too late** for legibility — it's just goal recognition, not intent inference.
- The steered condition frequently shows legibility LOWER than baseline at t=3s (0.584 vs 0.690). This is paradoxical — steering is supposed to help.
- The `enforce_block_direction` forces Y-sign of early actions, which may create unnatural-looking motions that confuse the VLM.
- ~2.58 interventions per steered episode means VLM is intervening at nearly every checkpoint — but the interventions aren't working.

#### 3.7 What we need to add/fix to make this ICRA/RSS 2027 best paper material

Be specific, prioritize, and tell me which are **blocking** vs. **nice-to-have**:

**BLOCKING ISSUES (must fix before any submission):**
- [ ] The steering doesn't work — legibility is worse with VLM steering than without.
- [ ] No human evaluation — we can't claim "legibility" without human data.
- [ ] VLM calibration: are Gemini's pA/pB actually calibrated probabilities? If not, `max(pA,pB)` is meaningless.
- [ ] No correlation analysis between VLM scores and analytical legibility metrics.
- [ ] Only one task environment (TwoBlockPick) — need at least 2-3 tasks for generalization claims.

**METHODOLOGICAL IMPROVEMENTS:**
- [ ] Better VLM prompting: chain-of-thought, few-shot examples, trajectory overlays.
- [ ] Connect the analytical `legibility_metrics.py` to the steering loop (use as complementary signal).
- [ ] Ablation studies: K values, θ thresholds, look-ahead horizons, VLM models.
- [ ] Compare against proper baselines: (a) random reranking, (b) analytical-metric-guided steering, (c) Bézier arc warping without VLM, (d) no steering at all.

**FOR VLA GENERALIZATION:**
- [ ] Abstract the steering interface: observation → policy → candidates → evaluator → selection. This should work for any policy (diffusion, VLA, BC-transformer).
- [ ] Design the prompt to be task-agnostic: "Given these frames, what is the robot trying to do?" rather than our current binary A/B choice.
- [ ] Consider using VLA self-evaluation: if the VLA can generate actions AND evaluate them, the evaluator is "built in."

**FOR BEST PAPER:**
- [ ] Novel theoretical contribution: formalize VLM-as-Bayesian-observer and prove conditions under which VLM-guided steering improves legibility.
- [ ] Comprehensive human study (minimum N=20 participants, IRB-approved).
- [ ] Real robot experiments on at least a Franka or similar platform.
- [ ] Comparison with SOTA legibility methods: Dragan's functional gradient, Shi's IPF diffusion, Bronars' Legibility Diffuser.
- [ ] Computational analysis: VLM call latency, throughput, cost vs. benefit.

#### 3.8 Edge Cases and Failure Modes

Identify and explain every edge case:
1. What happens when the policy generates ambiguous trajectories that go straight between both blocks? (The VLM correctly says 0.50/0.50 — steering can't help here because the diffusion policy doesn't know how to go around.)
2. What if the VLM hallucinates (says "robot is clearly going left" when it's going right)?
3. What if `enforce_block_direction` conflicts with the diffusion policy's learned dynamics (creates jarring motions)?
4. What if the PyBullet save/restore introduces subtle physics inconsistencies?
5. What if K candidates are all near-identical (diffusion policy has collapsed to a single mode for this observation)?
6. What about the cube jitter=0.0 in experiments — does this make the task unrealistically easy/symmetric?
7. What about temporal consistency — does the VLM's judgment change if you reorder frames?
8. In `prefix_frames` mode with many frames, does the VLM properly attend to early vs. late frames, or does it just look at the last frame?

#### 3.9 Literature Review Integration

Search for and cite these papers and explain their relevance to our work:
1. Dragan, Lee, Srinivasa. "Legibility and Predictability of Robot Motion." HRI 2013.
2. Dragan & Srinivasa. "Generating Legible Motion." RSS 2014.
3. Shi, Grislain, Sigaud, Chetouani. "Controlling Intent Expressiveness in Robot Motion with Diffusion Models." arXiv:2510.12370, 2025.
4. Bronars, Cheng, Xu. "Legibility Diffuser: Offline Imitation for Intent Expressive Motion." RA-L 2024.
5. Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023.
6. Black et al. "Training Diffusion Models with Reinforcement Learning." ICLR 2024.
7. Any work on VLMs as reward models for robotics (2024-2026).
8. Any work on legibility in VLA or foundation model policies.
9. DPPO, DDPO, or other diffusion+RL methods.
10. Preference-based learning for robot motion (DPO applied to robotics).

#### 3.10 Generalization plan to VLA

I want this methodology to eventually work with Vision-Language-Action models (e.g., RT-2, Octo, OpenVLA, π₀). Design the abstraction:
- What would the "steering interface" look like for a VLA?
- Can we use the VLA's own language conditioning to steer toward legibility? (e.g., "pick the left block in a way that makes it obvious you're going left")
- How does the multimodal nature of VLAs change the steering problem?
- What are the unique challenges (latency, sample efficiency, action representation)?

---

### 4. FORMAT YOUR RESPONSE AS

1. **One-paragraph executive summary** of the project's current state and critical path.
2. **Section-by-section answers** to 3.1 through 3.10, each with:
   - Diagnosis of the current state
   - Specific recommended fixes (with code-level detail where applicable)
   - Literature references
   - Priority rating (P0 = blocking, P1 = important, P2 = nice-to-have)
3. **90-day research roadmap** for ICRA/RSS 2027 submission.
4. **Top 5 experiments to run this week** to unblock progress.

---

*This prompt was auto-generated by exhaustive analysis of the full codebase on 2026-03-13. Every file, every metric, every result has been read.*
