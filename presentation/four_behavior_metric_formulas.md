# Four-Behavior Pipeline: Slide-Ready Formulas

This note matches the metrics currently used in the codebase for the 4-behavior pipeline.

Source implementations:
- `evaluation/eval_cfg_vlm.py`
- `evaluation/run_fast_cfg_eval.py`
- `evaluation/eval_behaviors_v2.py`

## Notation

- Trajectory: `tau = {x_t}_{t=0}^{T}`, where `x_t` is the end-effector position at step `t`
- Left/right goals: `g_L, g_R`
- True target goal: `g*`
- Obstacle center: `o`
- Waypoint position: `w`
- Number of candidates: `K`

## 1. Best-of-K Selection Formula

For each candidate trajectory `tau^(k)`:

```math
S^{(k)} = s_{\mathrm{VLM}}^{(k)} + b_{\mathrm{feas}}^{(k)}
```

```math
b_{\mathrm{feas}}^{(k)} = \max\left(0,\;0.05\left(1 - \frac{\min(\|x_T^{(k)}-g_L\|_2,\|x_T^{(k)}-g_R\|_2)}{0.2}\right)\right)
```

Selected candidate:

```math
k^* = \arg\max_{k \in \{1,\dots,K\}} S^{(k)}
```

Slide text:
"We generate `K` candidate trajectories, score each with the VLM, add a small feasibility bonus for ending near a block, and execute the highest-scoring candidate."

## 2. Legibility Metric

Important:

- `evaluation/eval_cfg_vlm.py` uses a target-aware version of `L_early`
- `evaluation/eval_behaviors_v2.py` uses a target-free variant that returns the larger of the two mean posteriors

Early window:

```math
T_e = \max(2,\lfloor 0.3T \rfloor)
```

Observer model:

```math
\sigma = \frac{\|g_L-g_R\|_2}{2\sqrt{2\ln 2}}
```

```math
P(g_i \mid x_t) =
\frac{\exp\!\left(-\frac{\|x_t-g_i\|_2^2}{2\sigma^2}\right)}
{\sum_{j \in \{L,R\}} \exp\!\left(-\frac{\|x_t-g_j\|_2^2}{2\sigma^2}\right)}
```

Primary score:

```math
L_{\mathrm{early}}(\tau) = \frac{1}{T_e}\sum_{t=0}^{T_e-1} P(g^* \mid x_t)
```

This is the target-aware form implemented in `evaluation/eval_cfg_vlm.py`.

In `evaluation/eval_behaviors_v2.py`, the code instead computes:

```math
\bar{P}_L = \frac{1}{T_e}\sum_{t=0}^{T_e-1} P(g_L \mid x_t), \qquad
\bar{P}_R = \frac{1}{T_e}\sum_{t=0}^{T_e-1} P(g_R \mid x_t)
```

and returns:

```math
L_{\mathrm{early}}^{v2}(\tau) = \max(\bar{P}_L,\bar{P}_R)
```

with the reported goal label equal to whichever of `left` or `right` has the larger mean posterior.

Slide text:
"Legibility is the average posterior probability of the true goal during the first 30% of the motion. Higher `L_early` means the robot's intent becomes clear earlier."

## 3. Predictability Metric

```math
\mathrm{PathEff}(\tau) =
\frac{\|x_T-x_0\|_2}
{\sum_{t=0}^{T-1}\|x_{t+1}-x_t\|_2}
```

Range:

```math
0 < \mathrm{PathEff} \le 1
```

Slide text:
"Predictability is measured as path efficiency: straight-line distance divided by actual path length. Higher values indicate more direct, less curved motion."

## 4. Safety Metric

Current implementation uses XY-plane clearance from the obstacle center:

```math
C_{\min}(\tau) = \min_{t} \|x_{t,xy} - o_{xy}\|_2
```

Collision flag:

```math
\mathrm{Collision}(\tau) = \mathbb{1}[C_{\min}(\tau) < r_{\mathrm{obs}}]
```

Slide text:
"Safety is measured by minimum obstacle clearance over the trajectory. Higher clearance is safer; a collision is counted if clearance falls below the obstacle radius."

## 5. Grounding Metric

Current implementation measures minimum distance to the waypoint:

```math
D_{\mathrm{wp}}(\tau) = \min_t \|x_t - w\|_2
```

Hover success:

```math
\mathrm{Hovered}(\tau) = \mathbb{1}[D_{\mathrm{wp}}(\tau) < 0.06]
```

Slide text:
"Grounding is measured by the closest approach to the waypoint block. Lower waypoint distance is better; we also report a binary hover success if the end effector comes within 6 cm."

## 6. Success Rate Across Episodes

For `N` evaluation episodes:

```math
\mathrm{SuccessRate} = \frac{1}{N}\sum_{n=1}^{N}\mathbb{1}[\mathrm{success}_n]
```

Mean metric value:

```math
\bar{m} = \frac{1}{N}\sum_{n=1}^{N} m_n
```

Standard deviation:

```math
\sigma_m = \sqrt{\frac{1}{N}\sum_{n=1}^{N}(m_n-\bar{m})^2}
```

Slide text:
"We report success rate and mean +/- standard deviation of the behavior-specific metric over all evaluation episodes."

## 7. Short Version for a Single Slide

If you want one compact thesis slide, use exactly this:

### Behavior-Specific Evaluation Metrics

```math
L_{\mathrm{early}} = \frac{1}{T_e}\sum_{t=0}^{T_e-1} P(g^* \mid x_t)
```

```math
\mathrm{PathEff} = \frac{\|x_T-x_0\|_2}{\sum_{t=0}^{T-1}\|x_{t+1}-x_t\|_2}
```

```math
C_{\min} = \min_t \|x_{t,xy}-o_{xy}\|_2
```

```math
D_{\mathrm{wp}} = \min_t \|x_t-w\|_2
```

Suggested one-line summary:
"We evaluate legibility by early goal posterior, predictability by path efficiency, safety by minimum obstacle clearance, and grounding by minimum waypoint distance."

## 8. Important Thesis Note

The original high-level prompt describes grounding as a sequential waypoint-following behavior. However, the current implemented metric in the evaluation code is:

```math
D_{\mathrm{wp}}(\tau) = \min_t \|x_t-w\|_2
```

So for the thesis presentation, the most accurate wording is:

"Grounding is operationalized as waypoint proximity in the current pipeline."

If you want, this can be extended in future work to an ordering-aware metric such as:

```math
\mathbb{1}[t_{\mathrm{wp}} < t_{\mathrm{goal}} \;\land\; D_{\mathrm{wp}} < \delta]
```

but that is not the metric currently reported by the code.
