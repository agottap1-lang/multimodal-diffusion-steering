# Task-Agnostic Legibility Metrics: Mathematical Foundation

## For ICRA / RSS Submission

> **Module**: `evaluation/legibility_metrics.py`
>
> **Key property**: No task-specific thresholds. The observer model auto-calibrates
> from the goal configuration alone.

---

## 1. Problem Formulation

A robot executes a trajectory $\xi = \{x_0, x_1, \ldots, x_T\}$ in
$\mathbb{R}^d$ toward an intended goal $g^* \in \mathcal{G} = \{g_1,
\ldots, g_K\}$.  A human observer watches the partial trajectory
$\xi_{0:t}$ and maintains a **posterior belief** over goals:

$$P(g \mid \xi_{0:t}) = \frac{L(g;\, \xi_{0:t})\, P(g)}
   {\sum_{g' \in \mathcal{G}} L(g';\, \xi_{0:t})\, P(g')}$$

where $L(g;\, \xi_{0:t})$ is a likelihood model encoding the assumption
that rational agents move efficiently toward their goals, and
$P(g) = 1/K$ (uniform prior—no advance knowledge of the goal).

A trajectory is **legible** if $P(g^* \mid \xi_{0:t})$ concentrates on
$g^*$ **early** in the trajectory.


## 2. Observer Likelihood Models

### 2.1 Spatial Proximity Model (Information Potential Field)

Following Shi et al. (arXiv:2510.12370, 2025), the likelihood of observing
configuration $x$ when the agent aims for goal $g$ is a Gaussian centred
at $g$:

$$P(x \mid g) = \frac{1}{(2\pi\sigma^2)^{d/2}}
   \exp\!\left(-\frac{\|x - g\|^2}{2\sigma^2}\right)$$

The **information potential** is the negative log-posterior:

$$\phi(x \mid g^*) = -\log P(g^* \mid x)$$

Small $\phi$ = position strongly supports $g^*$ (legible).
Large $\phi$ = position is ambiguous.

#### Auto-Calibration of $\sigma$

The scale parameter $\sigma$ is set so that the Gaussian's half-maximum
falls at the Voronoi boundary between the two closest goals:

$$\sigma = \frac{d_{\min}}{2\sqrt{2 \ln 2}}$$

where $d_{\min} = \min_{i \ne j} \|g_i - g_j\|$ is the minimum
pairwise inter-goal distance.

**Derivation**:  We require $P(\text{midpoint} \mid g_i) = \frac{1}{2}
\max_x P(x \mid g_i)$, i.e. the likelihood at the midpoint between the
two closest goals equals half its maximum (which occurs at $x = g_i$).
The midpoint is at distance $d_{\min}/2$ from each goal:

$$\exp\!\left(-\frac{(d_{\min}/2)^2}{2\sigma^2}\right) = \frac{1}{2}$$

$$\frac{d_{\min}^2}{8\sigma^2} = \ln 2$$

$$\sigma = \frac{d_{\min}}{2\sqrt{2 \ln 2}}$$

This is the **theoretically optimal discrimination scale**: any smaller
and the observer is too myopic (can only distinguish goals once very
close); any larger and the observer is too coarse (sees all positions as
equally likely under all goals).


### 2.2 Cost-Ratio Model (Boltzmann Rationality)

Following Dragan, Lee & Srinivasa (HRI 2013), an observer who assumes
the agent acts approximately optimally assigns:

$$P(\xi_{0:t} \mid g) \propto
   \exp\!\left[\beta \left(
      C^*(x_0, g) - C(\xi_{0:t}) - C^*(x_t, g)
   \right)\right]$$

where:
- $C(\xi_{0:t}) = \sum_{s=0}^{t-1} \|x_{s+1} - x_s\|$ is the
  **cumulative path length**
- $C^*(a, b) = \|a - b\|$ is the **optimal** (straight-line) cost
- $\beta > 0$ is the **rationality parameter** (higher = agent assumed
  more rational)

The exponent is the **efficiency surplus**:

$$\Delta(t, g) = C^*(x_0, g) - C(\xi_{0:t}) - C^*(x_t, g)$$

- $\Delta > 0$: trajectory is *more efficient* than the straight-line
  baseline (impossible for Euclidean cost, but occurs transiently when the
  agent has made progress toward $g$ without accumulating much path length)
- $\Delta < 0$: trajectory has *wasted effort* relative to the optimal
  path for $g$
- $\Delta \approx 0$: trajectory is approximately optimal for $g$

The rationality parameter $\beta$ is internally normalised by $\bar{C} =
\text{mean}_k C^*(x_0, g_k)$ to ensure scale-invariance.


### 2.3 Combined Model

When `model="both"`, we compute the **geometric mean** of the two
posteriors and re-normalise:

$$P_{\text{combined}}(g \mid \xi_{0:t}) \propto
   \sqrt{P_{\text{Gauss}}(g \mid \xi_{0:t}) \cdot
         P_{\text{cost}}(g \mid \xi_{0:t})}$$

This combines spatial proximity (instantaneous position) with trajectory
efficiency (cumulative path quality), capturing both *where you are* and
*how you got there*.


## 3. Legibility Scores

All scores are normalised to $[0, 1]$ where higher = more legible.

### 3.1 Posterior Legibility ($L_{\text{posterior}}$)

The original Dragan et al. formulation (Eq. 1 in HRI 2013):

$$L_{\text{posterior}}(\xi) = \frac{\sum_{t=0}^{T} f(t)\, P(g^* \mid \xi_{0:t})}
   {\sum_{t=0}^{T} f(t)}$$

where $f(t) = \exp(-\alpha \cdot t/T)$ is an early-emphasis weighting
function.  With $\alpha = 3$, approximately 95% of the weight falls in
the first third of the trajectory.

**Interpretation**: The time-weighted average confidence in the true goal.
High values mean the observer was confident about $g^*$ throughout, with
emphasis on **early** confidence.


### 3.2 Information Potential Field Score ($L_{\text{IPF}}$)

Following Shi et al. (Eq. 4):

$$L_p(\xi) = -\sum_{t=0}^{T} f(t)\, \phi(x_t \mid g^*)$$

Normalised to $[0, 1]$ by dividing by the worst-case potential
($\phi_{\max} = \ln K$, which occurs under a uniform posterior):

$$L_{\text{IPF}} = 1 + \frac{L_p(\xi)}{\ln K}$$


### 3.3 Entropy Area-Under-Curve ($L_{\text{entropy\_auc}}$)

The Shannon entropy of the goal distribution at timestep $t$:

$$H(t) = -\sum_{k=1}^{K} P(g_k \mid \xi_{0:t}) \ln P(g_k \mid \xi_{0:t})$$

Maximum entropy is $\ln K$ (uniform distribution); minimum is $0$
(certainty about one goal).

$$L_{\text{entropy\_auc}} = 1 - \frac{1}{T} \sum_{t=0}^{T}
   \frac{H(t)}{\ln K}$$

**Interpretation**: 1 minus the normalised mean entropy.  Higher values
mean the entropy was low (goal was unambiguous) for most of the
trajectory.


### 3.4 Intent Information Rate ($L_{\text{IIR}}$)

The rate of information accumulation about the goal:

$$\text{IIR}(t) = D_{\text{KL}}(P_{t} \| U) - D_{\text{KL}}(P_{t-1} \| U)$$

where $D_{\text{KL}}(P_t \| U) = \ln K - H(t)$ is the KL divergence
from the uniform prior.  IIR measures how many "nats" of goal information
each timestep provides.

The score is the early-weighted average of positive IIR values, normalised
by the total KL accumulated:

$$L_{\text{IIR}} = \frac{\sum_t f(t) \cdot \max(\text{IIR}(t), 0)}
   {D_{\text{KL}}(P_T \| U)}$$

**Interpretation**: What fraction of the total goal information was
communicated early?  A score near 1 means almost all disambiguation
happened in the first few timesteps.


### 3.5 Half-Time ($L_{\text{half\_time}}$)

The normalised time at which entropy first drops below half its maximum:

$$t_{\text{half}} = \min\{t : H(t) < \tfrac{1}{2} \ln K\}$$

$$L_{\text{half\_time}} = 1 - \frac{t_{\text{half}}}{T}$$

**Interpretation**: How quickly the observer's uncertainty drops below 50%
of maximum.  A score of 1 means instant disambiguation; 0 means it never
happens.


### 3.6 Early Intent ($L_{\text{early}}$)

The mean posterior of the true goal in the first 30% of the trajectory:

$$L_{\text{early}} = \frac{1}{|T_{0.3}|} \sum_{t \in T_{0.3}}
   P(g^* \mid \xi_{0:t})$$

where $T_{0.3} = \{0, 1, \ldots, \lfloor 0.3T \rfloor\}$.

**Interpretation**: Direct measure of how clearly the intent is
communicated in the critical "first glance" window.  Connected to
psychology research on action observation, where observers form intent
judgments within 200–500 ms (Sebanz & Knoblich, 2009).


### 3.7 Intent Commitment Point ($L_{\text{commitment}}$)

The earliest timestep where the posterior exceeds a strong commitment
threshold ($\theta = 0.80$):

$$t_{\text{ICP}} = \min\{t : P(g^* \mid \xi_{0:t}) > \theta\}$$

$$L_{\text{commitment}} = 1 - \frac{t_{\text{ICP}}}{T}$$

**Interpretation**: When does the observer become "strongly committed" to
$g^*$?  This is the **decision point** — the moment an observer would
feel confident enough to act on their inference.  Earlier commitment =
more legible.


### 3.8 Geometric Legibility ($L_{\text{geometric}}$)

Uses differential geometry (Frenet–Serret analysis) of the trajectory
curve to measure **goal-directed curvature**.

#### Discrete Menger Curvature

For three consecutive points $x_{i-1}, x_i, x_{i+1}$, the Menger
curvature (reciprocal of circumradius) is:

$$\kappa_i = \frac{2\, \|(x_{i+1} - x_i) \times (x_i - x_{i-1})\|}
   {\|x_{i+1} - x_i\| \cdot \|x_i - x_{i-1}\| \cdot \|x_{i+1} - x_{i-1}\|}$$

#### Frenet Normal

The unit normal $\mathbf{N}_i$ is the direction of tangent change:

$$\mathbf{N}_i = \frac{\mathbf{T}_{i+1} - \mathbf{T}_i}
   {\|\mathbf{T}_{i+1} - \mathbf{T}_i\|}$$

where $\mathbf{T}_i = (x_{i+1} - x_i) / \|x_{i+1} - x_i\|$ is the unit
tangent.

#### Goal-Relative Curvature

The **signed goal-relative curvature** projects the curvature normal onto
the direction toward the true goal:

$$\kappa_g(t) = \kappa(t) \cdot \cos \angle\!\left(\mathbf{N}(t),\;
   g^* - x_t\right)$$

- $\kappa_g > 0$: trajectory is **curving toward** $g^*$
- $\kappa_g < 0$: trajectory is **curving away** from $g^*$
- $\kappa_g = 0$: straight or curvature perpendicular to goal direction

The geometric legibility score is the early-weighted integral of the
positive component, mapped through $\tanh$ for normalisation:

$$L_{\text{geometric}} = \tanh\!\left(c \sum_t f(t) \cdot
   \max(\kappa_g(t), 0)\right)$$

**Interpretation**: Captures the physical signature of legible motion —
exaggerated curvature toward the intended goal.  This is the geometric
mechanism by which legible trajectories communicate intent: they curve
more strongly than necessary toward $g^*$, physically "pointing" at it.


### 3.9 Composite Score ($L_{\text{composite}}$)

A weighted combination with default weights:

| Component          | Weight | Rationale                              |
|--------------------|--------|----------------------------------------|
| $L_{\text{posterior}}$  | 0.25   | Core legibility definition (Dragan)     |
| $L_{\text{early}}$     | 0.25   | Emphasises first-glance readability     |
| $L_{\text{commitment}}$| 0.20   | Decision-theoretic intent threshold     |
| $L_{\text{geometric}}$ | 0.15   | Physical curvature signature            |
| $L_{\text{entropy\_auc}}$ | 0.10 | Information-theoretic disambiguation  |
| $L_{\text{IIR}}$       | 0.05   | Temporal information distribution       |

$$L_{\text{composite}} = \sum_i w_i \cdot L_i \bigg/ \sum_i w_i$$


## 4. Geometric Analysis

### 4.1 Path Efficiency

$$\eta = \frac{\|x_T - x_0\|}{\sum_{t=0}^{T-1} \|x_{t+1} - x_t\|}$$

Ratio of straight-line distance to arc length.  $\eta = 1$ for a
perfectly straight path; $\eta < 1$ for curved paths.  Legible
trajectories typically have $\eta < 1$ because they deviate from the
straight line to communicate intent.


### 4.2 Mean Curvature

$$\bar{\kappa} = \frac{1}{T-2} \sum_{t=1}^{T-1} \kappa_t$$

Average unsigned curvature along the trajectory.  Higher values indicate
more curved trajectories.


## 5. Key Properties

### 5.1 Task-Agnosticism

The metric requires **only**:
- End-effector positions $\{x_t\}$ (any dimensionality $d$)
- Goal positions $\{g_k\}$ (same dimensionality)
- Index of the true goal

No task-specific thresholds, no knowledge of the task semantics, no
environment-dependent parameters.  The observer model self-calibrates from
the goal geometry.

### 5.2 Scale Invariance

All scores are normalised to $[0, 1]$ regardless of workspace scale.
The $\sigma$ auto-calibration and $\beta$ normalisation ensure that the
same trajectory geometry produces the same scores whether the workspace
is measured in metres or centimetres.

### 5.3 Goal-Count Generality

The framework handles any $K \geq 2$ goals.  Entropy measures naturally
scale with $\ln K$, and the Bayesian posterior handles arbitrary goal
configurations.

### 5.4 Interpretability

Each score maps to a distinct, intuitive aspect of legibility:
- **Posterior/Early/Commitment**: "How quickly can the observer guess $g^*$?"
- **Entropy/IIR**: "How efficiently is goal information communicated?"
- **Geometric**: "Is the physical motion curvature pointing at $g^*$?"
- **Composite**: Principled combination of all aspects.


## 6. Comparison with Previous Metrics

| Metric                | Task-Specific? | Threshold-Free? | Multi-Goal? | Captures Dynamics? |
|-----------------------|----------------|-----------------|-------------|-------------------|
| Arc magnitude (ours)  | ✗ Yes          | ✗ No            | ✗ 2 only    | ✗ No              |
| VLM score (Gemini)    | ✗ Partially    | ✓ Yes           | ✓ Yes       | ✓ Yes             |
| $L_d$ (Bronars 2024)  | ✗ Partially    | ✓ Yes           | ✗ 2 only    | ✗ No              |
| **Ours (composite)**  | **✓ No**       | **✓ Yes**       | **✓ Yes**   | **✓ Yes**         |


## 7. Usage

```python
from evaluation.legibility_metrics import (
    compute_legibility,
    compute_legibility_from_obs,
    select_most_legible,
)

# Direct API (any task):
result = compute_legibility(
    trajectory=ee_positions,    # (T, d) array
    goals=goal_positions,       # (K, d) array
    true_goal_idx=0,
    model="both",               # "gaussian", "cost_ratio", or "both"
    return_curves=True,
)

print(result.L_composite)       # Overall legibility ∈ [0, 1]
print(result.summary_line())    # Quick console summary

# TwoBlockPickEnv convenience wrapper:
result = compute_legibility_from_obs(obs_trajectory, target_block="LEFT")

# Candidate selection:
best_idx, best_result = select_most_legible(
    trajectories=[traj1, traj2, traj3],
    goals=goals,
    true_goal_idx=0,
)
```


## 8. References

1. Dragan, A. D., Lee, K. C., & Srinivasa, S. S. (2013). "Legibility and
   Predictability of Robot Motion." *8th ACM/IEEE International Conference
   on Human-Robot Interaction (HRI)*, pp. 301–308.

2. Dragan, A. D., & Srinivasa, S. S. (2013). "Generating Legible Motion."
   *Robotics: Science and Systems (RSS)*.

3. Shi, W., Grislain, C., Sigaud, O., & Chetouani, M. (2025). "Controlling
   Intent Expressiveness in Robot Motion with Diffusion Models."
   arXiv:2510.12370.

4. Bronars, M., Cheng, S., & Xu, D. (2024). "Legibility Diffuser: Offline
   Imitation for Intent Expressive Motion." *IEEE Robotics and Automation
   Letters*.

5. Sebanz, N., & Knoblich, G. (2009). "Prediction in Joint Action: What,
   When, and Where." *Topics in Cognitive Science*, 1(2), pp. 353–367.
