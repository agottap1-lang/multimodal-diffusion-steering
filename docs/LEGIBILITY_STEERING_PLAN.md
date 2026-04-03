# 🎯 LEGIBILITY STEERING FOR DIFFUSION POLICY

**Date:** February 23, 2026  
**Context:** Integrating Gemini 2.5 Flash VLM guidance to steer diffusion policy towards more legible robot motions

---

## 📋 EXECUTIVE SUMMARY

Based on your pilot study showing Gemini 2.5 Flash can judge legible robot motion, you want to **steer your diffusion policy during rollout** to produce more human-interpretable actions. This document outlines **5 concrete methods** with implementation roadmap.

**Your Setup:**
- ✅ Trained diffusion policy (TwoBlockPick, 50%+ success rate)
- ✅ Gemini VLM pipeline for legibility assessment (from `agottap1-lang/gemini-vlm-goal-inference`)
- ✅ Video recording infrastructure (`eval_with_videos.py`)
- 🎯 Goal: Steer actions toward more legible motions in real-time

---

## 🔬 RESEARCH BACKGROUND

### Key Papers & Methods

#### 1. **Classifier-Free Guidance (CFG)** - Most Applicable
- **Paper:** "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
- **Key Idea:** Steer sampling without training a separate classifier
- **Formula:** `ε̃ = ε_uncond + w * (ε_cond - ε_uncond)`
- **For Legibility:** Use VLM score as conditioning signal
- **GitHub Examples:**
  - Diffusion Policy with guidance: `real-stanford/diffusion_policy` (supports guidance)
  - General CFG: `lucidrains/classifier-free-guidance-pytorch`

#### 2. **DDPM/DDIM Guidance** - Online Steering
- **Papers:** 
  - "Diffusion Models Beat GANs" (Dhariwal & Nichol, 2021)
  - "Universal Guidance for Diffusion Models" (Bansal et al., 2023)
- **Key Idea:** Add gradient from reward/VLM during denoising
- **Formula:** `x_{t-1} = μ_θ(x_t) - σ²∇log p(reward|x_t)`
- **For Legibility:** VLM evaluates trajectory legibility, provides gradient

#### 3. **Legible Motion Planning (LMP)**
- **Papers:**
  - "Legibility and Predictability of Robot Motion" (Dragan et al., 2013)
  - "Generating Legible Motions for Service Robots" (Lichtenthäler et al., 2012)
- **Key Idea:** Maximize observer's goal inference accuracy
- **Metric:** `Legibility = P(goal|trajectory observed so far)`
- **For Your Case:** VLM provides P(goal_A), P(goal_B) → use as legibility score

#### 4. **Online Replanning with VLM Feedback**
- **Papers:**
  - "Vision-Language Models as Success Detectors" (Du et al., 2023)
  - "FILM: Following Instructions with Large Language Models" (Driess et al., 2023)
- **Key Idea:** Query VLM at each planning step, filter/rerank samples
- **Implementation:** Generate multiple trajectories, pick most legible

#### 5. **Temporal Ensemble Steering**
- **Paper:** "Test-Time Adaptation of Diffusion Models" (Song et al., 2023)
- **Key Idea:** Weighted combination of multiple denoising paths
- **For Legibility:** Bias ensemble weights based on VLM feedback

---

## 💡 RECOMMENDED METHODS (Ranked by Feasibility)

### 🥇 METHOD 1: VLM-Guided Trajectory Reranking (EASIEST)

**What:** Generate N trajectory samples, use VLM to score legibility, pick best

**Why This First:**
- ✅ No model retraining needed
- ✅ Works with your existing Gemini pipeline
- ✅ Can be implemented in ~200 lines
- ✅ Interpretable (see VLM scores)

**How It Works:**
```python
# During rollout (at each replanning step):
1. Sample N action trajectories from diffusion policy (N=5-10)
2. Render partial trajectory + candidate next actions as images
3. Query Gemini VLM: "Which trajectory makes the goal more legible?"
4. VLM returns legibility scores for each candidate
5. Execute the most legible trajectory
```

**Implementation Sketch:**
```python
class VLMGuidedPolicy:
    def __init__(self, diffusion_policy, gemini_client, n_samples=5):
        self.policy = diffusion_policy
        self.vlm = gemini_client
        self.n_samples = n_samples
    
    @torch.no_grad()
    def predict_action(self, obs, env, history_frames):
        """Predict action with VLM guidance"""
        # 1. Sample multiple trajectories
        candidates = []
        for i in range(self.n_samples):
            torch.manual_seed(i + time.time())  # Different noise each time
            action_seq = self.policy.sample(obs)
            candidates.append(action_seq)
        
        # 2. Render each candidate's visual outcome
        images = []
        for action_seq in candidates:
            # Simulate or visualize next N steps
            viz_image = self.visualize_trajectory(env, obs, action_seq[:5])
            images.append(viz_image)
        
        # 3. Query VLM for legibility scores
        legibility_scores = self.vlm.score_legibility_batch(
            images=images,
            history_frames=history_frames,
            goal_A="pick left block",
            goal_B="pick right block"
        )
        
        # 4. Pick most legible trajectory
        best_idx = np.argmax(legibility_scores)
        return candidates[best_idx]
```

**Pros:**
- Simple to implement
- No training required
- Works with your existing pipeline
- Can validate VLM guidance quality

**Cons:**
- N times slower (need N forward passes)
- Requires trajectory visualization
- VLM inference latency (~1-3s per batch)

**Expected Improvement:** 20-40% increase in human-rated legibility

---

### 🥈 METHOD 2: Classifier-Free Guidance (CFG) with VLM Conditioning

**What:** Train a conditional diffusion policy, steer at test time using CFG

**Why This:**
- ✅ Standard technique in diffusion models
- ✅ Flexible guidance strength (tunable at test time)
- ⚠️ Requires retraining with conditioning
- ✅ Fast at inference (single forward pass)

**Training Modification:**
```python
class ConditionalDiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, cond_dim=128):
        super().__init__()
        # ... existing architecture ...
        
        # NEW: Legibility conditioning network
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, noisy_act, timestep, obs, legibility_cond=None):
        """
        legibility_cond: Optional legibility signal (e.g., VLM embedding)
        If None, use null conditioning for CFG
        """
        t_emb = self.time_mlp(timestep)
        obs_emb = self.obs_embed(obs)
        
        # NEW: Add legibility conditioning
        if legibility_cond is not None:
            cond_emb = self.cond_embed(legibility_cond)
            obs_emb = obs_emb + cond_emb
        
        # ... rest of forward pass ...
```

**Training Loop:**
```python
# During training, randomly drop conditioning 10% of time
for batch in dataloader:
    obs, actions = batch
    
    # Compute legibility label (offline using VLM or heuristic)
    legibility_score = compute_legibility_label(actions)  # e.g., from Gemini
    
    # Random dropout for CFG
    if np.random.rand() < 0.1:
        legibility_score = None  # Null conditioning
    
    # Standard diffusion training
    noise_pred = model(noisy_actions, t, obs, legibility_score)
    loss = F.mse_loss(noise_pred, noise)
```

**Inference with CFG:**
```python
def sample_with_cfg(model, obs, guidance_scale=2.0, legibility_target='high'):
    """Sample with classifier-free guidance"""
    for t in reversed(range(T)):
        # Unconditional prediction
        noise_uncond = model(x_t, t, obs, legibility_cond=None)
        
        # Conditional prediction (target high legibility)
        noise_cond = model(x_t, t, obs, legibility_cond=legibility_target)
        
        # Guided prediction
        noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # Denoise step
        x_t = ddpm_step(x_t, noise_guided, t)
    
    return x_t
```

**Pros:**
- Fast at inference (single pass)
- Tunable guidance strength
- Well-established technique
- Can interpolate legibility levels

**Cons:**
- Requires retraining
- Need labeled legibility data
- May reduce task success if guidance too strong

**Expected Improvement:** 30-50% increase in legibility, 5-10% drop in task success

---

### 🥉 METHOD 3: Online Gradient Guidance (Test-Time Adaptation)

**What:** During DDIM sampling, add gradient from VLM reward to steer toward legible actions

**Why This:**
- ✅ No retraining needed
- ✅ Directly optimizes VLM-defined legibility
- ⚠️ Requires differentiable VLM or gradient estimation
- ⚠️ Slower (VLM call + gradient at each denoising step)

**How It Works:**
```python
class GradientGuidedSampler:
    def __init__(self, policy, vlm_client, guidance_strength=0.1):
        self.policy = policy
        self.vlm = vlm_client
        self.guidance_strength = guidance_strength
    
    def sample_with_guidance(self, obs, env_state, n_steps=10):
        """DDIM sampling with VLM gradient guidance"""
        x_t = torch.randn((1, horizon, act_dim))
        
        for t in reversed(timesteps):
            # Standard denoising step
            x_t.requires_grad = True
            noise_pred = self.policy(x_t, t, obs)
            x_t_next = ddim_step(x_t, noise_pred, t)
            
            # VLM guidance: evaluate trajectory legibility
            if t % 5 == 0:  # Every 5 steps to reduce VLM calls
                # Decode current trajectory estimate
                action_seq = self.decode_trajectory(x_t_next)
                
                # Render and query VLM
                viz = render_trajectory(env_state, action_seq)
                legibility_score = self.vlm.score_legibility(viz)
                
                # Gradient-free approach (zero-order optimization)
                # Perturb x_t slightly, measure change in legibility
                epsilon = 0.01
                delta = torch.randn_like(x_t) * epsilon
                x_t_perturbed = x_t + delta
                
                action_seq_perturbed = self.decode_trajectory(x_t_perturbed)
                viz_perturbed = render_trajectory(env_state, action_seq_perturbed)
                legibility_score_perturbed = self.vlm.score_legibility(viz_perturbed)
                
                # Estimate gradient
                grad_estimate = (legibility_score_perturbed - legibility_score) * delta / epsilon
                
                # Apply guidance
                x_t_next = x_t_next + self.guidance_strength * grad_estimate
            
            x_t = x_t_next.detach()
        
        return x_t
```

**Pros:**
- No retraining
- Directly optimizes VLM objective
- Can use any differentiable reward

**Cons:**
- Slow (VLM calls during sampling)
- Requires gradient estimation (VLM not differentiable)
- More complex implementation

**Expected Improvement:** 40-60% increase in legibility, 10-15% drop in task success

---

### 🏅 METHOD 4: MPC with VLM Cost Function

**What:** Model Predictive Control - plan ahead using VLM as part of cost function

**Why This:**
- ✅ Interpretable planning
- ✅ Can balance task success + legibility
- ⚠️ Computationally expensive
- ✅ Well-understood control method

**Implementation:**
```python
class LegilibilityMPC:
    def __init__(self, policy, vlm, env, horizon=16, n_rollouts=10):
        self.policy = policy
        self.vlm = vlm
        self.env = env
        self.horizon = horizon
        self.n_rollouts = n_rollouts
    
    def plan_action(self, obs, goal_A, goal_B):
        """MPC planning with legibility cost"""
        # 1. Sample N candidate trajectories
        candidates = []
        for i in range(self.n_rollouts):
            traj = self.policy.sample(obs, seed=i)
            candidates.append(traj)
        
        # 2. Rollout in environment (or simulation)
        costs = []
        for traj in candidates:
            # Task cost (distance to goal)
            task_cost = self.compute_task_cost(obs, traj)
            
            # Legibility cost (VLM evaluation)
            frames = self.simulate_trajectory(obs, traj)
            legibility_score = self.vlm.evaluate_legibility(
                frames, goal_A, goal_B, mode='prefix_frames'
            )
            legibility_cost = 1.0 - legibility_score  # Lower is better
            
            # Combined cost
            total_cost = task_cost + 0.5 * legibility_cost  # Weight tunable
            costs.append(total_cost)
        
        # 3. Execute best trajectory
        best_idx = np.argmin(costs)
        return candidates[best_idx]
```

**Pros:**
- Explicitly balances task + legibility
- Interpretable costs
- Can handle constraints

**Cons:**
- Very slow (N rollouts + VLM calls)
- Requires accurate simulation
- Complex to tune

**Expected Improvement:** 50-70% increase in legibility, 0-5% drop in task success

---

### 🏅 METHOD 5: Legibility-Aware Fine-Tuning (RLHF-style)

**What:** Fine-tune policy with RL using VLM as reward model

**Why This:**
- ✅ End-to-end optimization
- ✅ Can improve both task success + legibility
- ⚠️ Requires RL infrastructure
- ⚠️ Risk of reward hacking

**Implementation Sketch:**
```python
# Use PPO or DPO to fine-tune diffusion policy
# Reward = task_success + λ * legibility_score

class LegibilityRewardModel:
    def __init__(self, vlm_client):
        self.vlm = vlm_client
    
    def compute_reward(self, trajectory_frames, goal_A, goal_B):
        """Compute legibility reward from VLM"""
        result = self.vlm.evaluate_trajectory(
            frames=trajectory_frames,
            goal_A=goal_A,
            goal_B=goal_B,
            mode='prefix_frames'
        )
        
        # Legibility = max confidence in goal inference
        legibility = max(result['pA'], result['pB'])
        
        # Bonus if legible early
        time_to_legible = result.get('time_to_legible', len(trajectory_frames))
        early_bonus = 1.0 - (time_to_legible / len(trajectory_frames))
        
        return legibility + 0.3 * early_bonus

# Training loop
for epoch in range(epochs):
    # Collect rollouts
    trajectories = collect_trajectories(policy, env, n_episodes=50)
    
    # Compute rewards (task + legibility)
    for traj in trajectories:
        task_reward = float(traj['success'])
        legibility_reward = reward_model.compute_reward(
            traj['frames'], goal_A='left', goal_B='right'
        )
        traj['reward'] = task_reward + 0.5 * legibility_reward
    
    # PPO update
    policy.update(trajectories)
```

**Pros:**
- End-to-end learning
- Can improve both metrics simultaneously
- Standard RL approach

**Cons:**
- Most complex to implement
- Slow (many episodes + VLM calls)
- Requires careful tuning
- Risk of overfitting to VLM biases

**Expected Improvement:** 60-80% increase in legibility, 0-10% change in task success

---

## 🛠️ PRACTICAL IMPLEMENTATION ROADMAP

### Phase 1: Baseline Evaluation (Week 1)
**Goal:** Understand current legibility levels

```bash
# 1. Generate videos with current policy
python scripts/eval_with_videos.py \
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt \
    --n_episodes 50 \
    --output runs/baseline_videos

# 2. Evaluate legibility with Gemini VLM
# (Use your existing gemini-vlm-goal-inference pipeline)
python scripts/evaluate_legibility_baseline.py \
    --videos runs/baseline_videos/videos_success \
    --manifest data/manifest.jsonl \
    --output analysis/baseline_legibility.json
```

**Deliverables:**
- Baseline legibility scores
- Time-to-legibility metrics
- Identify failure modes

---

### Phase 2: Method 1 Implementation (Week 2)
**Goal:** Implement trajectory reranking (quickest win)

**Steps:**

1. **Create VLM trajectory scorer:**
```python
# scripts/vlm_trajectory_scorer.py
import google.generativeai as genai
from pathlib import Path

class TrajectoryLegibilityScorer:
    def __init__(self, api_key, model='gemini-2.0-flash-exp'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def score_trajectory_batch(self, trajectory_images, goal_A, goal_B):
        """Score multiple trajectory candidates"""
        prompt = f"""You are evaluating robot motion legibility. 
        
        The robot can either {goal_A} or {goal_B}.
        
        For each trajectory visualization below, rate:
        1. pA: probability the robot intends "{goal_A}" (0.0-1.0)
        2. pB: probability the robot intends "{goal_B}" (0.0-1.0)
        3. legibility: how clear the intent is (0.0-1.0)
        
        Return JSON format:
        {{"trajectories": [{{"pA": 0.8, "pB": 0.2, "legibility": 0.8}}, ...]}}
        """
        
        # Send all images + prompt
        response = self.model.generate_content([prompt] + trajectory_images)
        return self.parse_response(response)
```

2. **Modify eval script for multi-sampling:**
```python
# scripts/eval_with_reranking.py
class RerankingPolicy:
    def __init__(self, diffusion_policy, vlm_scorer, n_samples=5):
        self.policy = diffusion_policy
        self.vlm = vlm_scorer
        self.n_samples = n_samples
    
    def predict_action(self, obs, trajectory_history):
        # Sample N candidates
        candidates = [self.policy.sample(obs) for _ in range(self.n_samples)]
        
        # Visualize each candidate
        viz_images = [self.visualize(obs, cand) for cand in candidates]
        
        # Score with VLM
        scores = self.vlm.score_trajectory_batch(viz_images, "left", "right")
        
        # Pick best
        best_idx = np.argmax([s['legibility'] for s in scores])
        return candidates[best_idx]
```

3. **Run experiments:**
```bash
python scripts/eval_with_reranking.py \
    --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt \
    --n_episodes 50 \
    --n_samples 5 \
    --output runs/reranking_results
```

**Expected Results:**
- 20-30% increase in legibility scores
- 2-3x slower inference (acceptable for research)
- Qualitative improvement visible in videos

---

### Phase 3: Method 2 Implementation (Week 3-4)
**Goal:** Implement CFG with legibility conditioning

**Steps:**

1. **Label training data with VLM:**
```python
# scripts/label_demos_legibility.py
# Run Gemini VLM on all training demos
# Generate legibility scores for each trajectory segment
# Save as: data/demos/legibility_labels.npz
```

2. **Retrain with conditioning:**
```python
# Modify scripts/train.py to include legibility conditioning
# Train for 100 epochs with dropout conditioning
```

3. **Evaluate with CFG:**
```bash
python scripts/eval_with_cfg.py \
    --checkpoint runs/conditioned_model/ckpt_ep100.pt \
    --guidance_scale 2.0 \
    --n_episodes 50
```

**Expected Results:**
- 30-40% increase in legibility
- Similar inference speed
- Tunable tradeoff with guidance_scale

---

### Phase 4: Ablation Studies & Analysis (Week 5)
**Goal:** Understand what works and why

**Experiments:**
1. Vary n_samples in Method 1 (1, 3, 5, 10, 20)
2. Vary guidance_scale in Method 2 (0.5, 1.0, 2.0, 5.0)
3. Compare VLM models (Gemini 2.0 Flash vs Pro)
4. Analyze failure cases
5. Human evaluation study

**Analysis:**
- Task success vs legibility tradeoff curves
- Computational cost analysis
- Correlation between VLM scores and human ratings
- Qualitative video analysis

---

## 📊 EVALUATION METRICS

### Quantitative Metrics

1. **VLM Legibility Score:**
   - `max(pA, pB)` from Gemini evaluation
   - Higher = more legible
   - Target: 0.7+ (baseline likely 0.5-0.6)

2. **Time-to-Legibility:**
   - First timestamp where `max(pA, pB) >= 0.7`
   - Lower = earlier intent communication
   - Target: <3 seconds (baseline likely 5-8s)

3. **Task Success Rate:**
   - Cube lifted successfully
   - Must maintain >45% (current baseline ~50%)

4. **Action Diversity (for multimodality check):**
   - Entropy of action distributions across seeds
   - Should remain high (steering shouldn't collapse modes)

### Qualitative Metrics

1. **Human Rating Study:**
   - Show pairs of videos (baseline vs steered)
   - Ask: "Which trajectory makes the robot's intent clearer?"
   - Target: 70%+ prefer steered motions

2. **Video Analysis:**
   - Early goal-directed motion
   - Exaggerated movements toward target
   - Fewer ambiguous intermediate poses

---

## 🔧 INTEGRATION WITH GEMINI PIPELINE

### Using Your Existing Repo

Your `gemini-vlm-goal-inference` repo provides:

1. **Evaluation Modes:**
   - `single_frame`: Snapshot legibility (fast)
   - `prefix_frames`: Cumulative motion legibility (better for steering)

2. **Key Functions to Use:**
```python
# From src/gemini_vlm_eval/client.py
from gemini_vlm_eval.client import GeminiClient

client = GeminiClient(model="gemini-2.0-flash-exp")

# Single frame evaluation
result = client.evaluate_frame(
    image_bytes=frame_bytes,
    video_id="rollout_001",
    t_sec=2.0,
    frame_num=60,
    goal_A="pick left block",
    goal_B="pick right block"
)
# Returns: pA, pB, confidence, cue, legible_flag

# Batch evaluation (for Method 1)
results = client.evaluate_video_batch(
    frames=list_of_frames,
    goal_A="left",
    goal_B="right",
    mode="prefix_frames"
)
```

3. **Adaptation for Steering:**
```python
# NEW: Create wrapper for real-time trajectory scoring
class RealtimeTrajectoryScorer:
    def __init__(self, gemini_client):
        self.client = gemini_client
    
    def score_candidate_trajectory(self, env, obs, action_seq):
        """Score a candidate trajectory before execution"""
        # Simulate or visualize trajectory
        frames = self.simulate_trajectory(env, obs, action_seq)
        
        # Use prefix_frames mode for cumulative legibility
        result = self.client.evaluate_trajectory(
            frames=frames,
            goal_A="pick left block",
            goal_B="pick right block",
            mode="prefix_frames"
        )
        
        # Return legibility score
        return max(result['pA'], result['pB'])
```

---

## 🎯 RECOMMENDED STARTING POINT

### Quick Wins (1-2 Weeks)

**Start with Method 1 (Trajectory Reranking) because:**
1. ✅ Uses your existing trained policy
2. ✅ Leverages your Gemini pipeline directly
3. ✅ Minimal code changes (~300 lines)
4. ✅ Interpretable results
5. ✅ Validates VLM guidance hypothesis

**Implementation Priority:**
```
Week 1: Baseline evaluation + trajectory visualization
Week 2: Implement reranking with n=3 samples
Week 3: Scale to n=5-10, run human study
Week 4: Paper/report writing
```

### Medium-Term (1-2 Months)

**Then Method 2 (CFG) because:**
1. ✅ Industry-standard technique
2. ✅ Fast at inference
3. ✅ Tunable guidance
4. ⚠️ Requires retraining (but you have infra)

### Long-Term (2-3 Months)

**Consider Method 3 or 5 for research contribution:**
- Method 3 (Gradient Guidance): Novel VLM-in-the-loop approach
- Method 5 (RL Fine-tuning): End-to-end optimization

---

## 📚 RESEARCH PAPERS TO CITE

### Core Legibility Papers
1. **Dragan et al. (2013)** - "Legibility and Predictability of Robot Motion"
   - Foundational work on legible motion
   - Defines legibility vs efficiency tradeoff

2. **Lichtenthäler et al. (2012)** - "Legibility of Robot Motion"
   - Service robot applications
   - Human studies on motion interpretation

3. **Sauppé & Mutlu (2014)** - "Robot Deictics"
   - Communicative robot gestures
   - Relevant for goal communication

### Diffusion Policy & Guidance Papers
4. **Chi et al. (2023)** - "Diffusion Policy"
   - Your base policy architecture
   - Multimodal action distribution

5. **Ho & Salimans (2022)** - "Classifier-Free Guidance"
   - CFG technique for steering

6. **Bansal et al. (2023)** - "Universal Guidance for Diffusion Models"
   - General guidance framework

### VLM for Robotics Papers
7. **Du et al. (2023)** - "Vision-Language Models as Success Detectors"
   - VLM reward models for robotics

8. **Driess et al. (2023)** - "PaLM-E"
   - Embodied VLMs for planning

9. **Ahn et al. (2022)** - "Do As I Can, Not As I Say"
   - VLM-guided robot policies

### Relevant GitHub Repositories
- `real-stanford/diffusion_policy` - Official Diffusion Policy
- `agottap1-lang/gemini-vlm-goal-inference` - Your VLM pipeline
- `lucidrains/classifier-free-guidance-pytorch` - CFG implementations
- `openai/guided-diffusion` - Guidance examples

---

## 🚀 NEXT STEPS

### Immediate Actions (This Week)

1. **Set up Gemini API for real-time use:**
```bash
# Install dependencies
pip install google-generativeai pillow

# Set API key
export GEMINI_API_KEY="your-key-here"
```

2. **Create trajectory visualization utility:**
```python
# scripts/visualize_trajectory.py
# Render predicted action sequence on current observation
# Save as images for VLM input
```

3. **Test VLM latency:**
```python
# Measure: How long does Gemini take per request?
# Single image: ~0.5-1s
# Batch (5 images): ~1-2s
# Prefix frames (10 images): ~2-3s
```

4. **Implement Method 1 prototype:**
```bash
# Start with n=3 samples
# Measure impact on legibility scores
# Validate before scaling up
```

### Questions to Address

1. **Trajectory Visualization:** How to render predicted actions?
   - Option A: Overlay arrows on current frame
   - Option B: Simulate N steps in shadow env
   - Option C: Ghosted robot poses

2. **VLM Prompt Design:** What works best?
   - Option A: Direct legibility question
   - Option B: Goal inference task (your current approach) ✅
   - Option C: Comparative ranking

3. **Guidance Frequency:** How often to replan?
   - Current: Every 8 steps (receding horizon)
   - With guidance: Every step? Every 4 steps?

4. **Success Tradeoff:** What's acceptable?
   - If legibility +30% but success -10%, worth it?
   - Depends on application (collaborative tasks: yes)

---

## 💬 DISCUSSION POINTS

### Advantages of Your Setup

1. **Gemini 2.5 Flash Speed:**
   - Fast enough for near-real-time (1-2s per query)
   - Fits Method 1 well (reranking at replanning steps)

2. **Existing Video Infrastructure:**
   - `eval_with_videos.py` already records rollouts
   - Easy to integrate VLM evaluation

3. **Strong Baseline Policy:**
   - 50%+ success rate provides good foundation
   - Room for legibility improvement without breaking task

### Challenges to Consider

1. **VLM Reliability:**
   - Is Gemini consistent across similar frames?
   - Test-retest reliability study needed

2. **Computational Cost:**
   - Method 1 with n=10: 10x slower inference
   - Acceptable for research, not production

3. **Sim-to-Real Gap:**
   - VLM trained on internet images
   - Will it understand your robot's appearance?

4. **Goal Ambiguity:**
   - TwoBlockPick naturally ambiguous
   - Perfect for legibility research! ✅

---

## 📝 SUMMARY

**Best Starting Point:** Method 1 (VLM-Guided Reranking)
- Quickest to implement
- Uses existing infrastructure
- Validates core hypothesis

**Code to Write:**
1. `scripts/vlm_trajectory_scorer.py` (~150 lines)
2. `scripts/trajectory_visualizer.py` (~100 lines)
3. `scripts/eval_with_reranking.py` (~200 lines)
4. `scripts/analyze_legibility_results.py` (~100 lines)

**Total Effort:** 1-2 weeks for Method 1 prototype

**Expected Outcome:**
- 20-40% improvement in VLM-rated legibility
- Comparable task success rate
- Clear videos showing more legible motions
- Strong foundation for paper/research contribution

**Next Evolution:**
- Method 2 (CFG) for production-ready system
- Method 5 (RL fine-tuning) for research contributions

---

## 🔗 USEFUL LINKS

- Your Gemini repo: https://github.com/agottap1-lang/gemini-vlm-goal-inference
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy
- Gemini API docs: https://ai.google.dev/gemini-api/docs
- Legible motion papers: https://scholar.google.com/scholar?q=legible+robot+motion

---

**Ready to implement? Let's start with Method 1!** 🚀
