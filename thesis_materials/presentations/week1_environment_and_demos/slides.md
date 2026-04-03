# Week 1: Environment Design & Demonstration Collection

## TwoBlockPick for Legible Robot Motion

**Anudeep Gottapu**
Arizona State University

---

## Slide 1: Problem Motivation

**Why Legible Motion Matters**

- In shared workspaces, robots must communicate intent through motion
- A human watching a robot reach for one of two objects needs to predict the goal **early**
- Ambiguous trajectories → human waits → slower collaboration
- Legible trajectories → early prediction → efficient teamwork

**Research Question:** Can we train a generative policy to produce task-successful AND legible trajectories?

---

## Slide 2: TwoBlockPick Environment

**PyBullet Simulation with Franka Panda**

| Parameter | Value |
|-----------|-------|
| Robot | Franka Panda 7-DOF |
| Physics | PyBullet, dt=1/240s |
| Objects | Two red blocks, 4cm each |
| Left block | (0.50, +0.07, 0.42) |
| Right block | (0.50, -0.07, 0.42) |
| Jitter | ±0.015m per block |
| Success | Lift block above z=0.52m |

**Observation (22-dim):** ee_pos(3) + ee_quat(4) + grip(1) + L_pos(3) + L_quat(4) + R_pos(3) + R_quat(4)

**Action (5-dim):** Δx, Δy, Δz, Δyaw, grip — each ∈ [-1, 1], position scale 0.05m/step

---

## Slide 3: Camera Setup

**Camera Convention (Critical for VLM Evaluation)**

- Camera: yaw=135°, pitch=-30°, distance=0.9m, FOV=60°
- **Image-LEFT = World-RIGHT** (and vice versa)
- Goal A = image-left = world-right block
- Goal B = image-right = world-left block

This mapping was initially incorrect and caused evaluation errors. Getting this right was essential for VLM scoring accuracy.

---

## Slide 4: Demonstration Collection Strategy

**400 Expert Demos via Analytical Bézier Curves**

| Style | Count | Design |
|-------|-------|--------|
| **Legible** (50%) | 200 | Quadratic Bézier, control point toward goal. Lateral commitment by t≈0.3 |
| **Neutral** (25%) | 100 | Quadratic Bézier, control point at y=0. No lateral signal |
| **Deceptive** (25%) | 100 | Cubic Bézier, P1 feints wrong side, P2 commits correct |

- 10 block configurations × 40 demos each
- 20 pick-left + 20 pick-right per configuration
- Each demo: full episode (up to 400 steps) of obs-action pairs

---

## Slide 5: Data Format

**demos_combined.npz**

```
obs:          (400, 400, 22)  float32
actions:      (400, 400, 5)   float32
labels:       (400,)          'left' or 'right'
style_labels: (400,)          0=legible, 1=neutral, 2=deceptive
config_id:    (400,)          0-9
arc_idx:      (400,)          index within config
```

Videos also rendered for each demo → `demo_videos_combined/`

---

## Slide 6: Why Three Styles?

**Training Data Diversity for Evaluation**

- **Legible demos** are the target behavior we want the policy to produce
- **Neutral demos** represent the baseline "shortest path" behavior
- **Deceptive demos** are the worst case — feinting before committing
- Mixed training data lets us evaluate whether the policy learns meaningful style differences
- Also enables VLM evaluation: does the VLM correctly rank styles by legibility?

---

## Slide 7: Design Decisions

**Key Choices Made This Week**

1. **Simulation over real robot** — Enables large-scale data collection (400 demos) and reproducible evaluation
2. **Analytical trajectories** — Bézier curves give precise control over trajectory shape, rather than human teleop which adds noise
3. **Symmetric block placement** — Both blocks equidistant from center, so legibility comes purely from trajectory shape, not proximity
4. **3 trajectory styles** — Provides ground truth for evaluating whether VLMs can distinguish legibility levels

---

## Slide 8: References

- Dragan, A. & Srinivasa, S. (2013). *Generating Legible Motion.* RSS 2013.
- Chi, C., Feng, S., Du, Y., et al. (2023). *Diffusion Policy.* RSS 2023.
- PyBullet documentation: https://pybullet.org
