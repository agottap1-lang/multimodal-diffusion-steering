#!/usr/bin/env python3
r"""
Task-Agnostic Legibility Metrics
=================================

A mathematically rigorous, self-calibrating framework for scoring trajectory
legibility based on **Bayesian goal inference**, **information theory**, and
**differential geometry**.  No task-specific thresholds are required — the
observer model scales automatically to the goal configuration.

Theory Overview
---------------

A trajectory $\xi = \{x_0, x_1, \ldots, x_T\}$ connects a start
configuration $x_0$ to an intended goal $g^*$ chosen from a finite set
$\mathcal{G} = \{g_1, \ldots, g_K\}$.  An observer watches the partial
trajectory $\xi_{0:t}$ and maintains a Bayesian posterior:

$$
    P(g \mid \xi_{0:t})
        = \frac{L(g; \xi_{0:t}) \, P(g)}
               {\sum_{g'} L(g'; \xi_{0:t}) \, P(g')}
$$

where $L(g; \xi_{0:t})$ is a *likelihood model* encoding the assumption
that agents move approximately rationally toward their goals.

We provide **two complementary likelihood models**:

1. **Spatial Proximity (Information Potential Field – Shi et al. 2025)**:

   $$ P(x \mid g) \propto \exp\!\Bigl(-\frac{\|x - g\|^2}{2\sigma^2}\Bigr) $$

   where $\sigma$ is **auto-calibrated** from the inter-goal geometry so
   that the half-maximum of the Gaussian aligns with the Voronoi boundary
   between goals.

2. **Cost-Ratio (Dragan et al. HRI 2013)**:

   $$ P(\xi_{0:t} \mid g)
        \propto \exp\!\bigl(C^*(x_0, g) - C(\xi_{0:t}) - C^*(x_t, g)\bigr)
   $$

   where $C(\xi_{0:t})$ is the *cumulative path length* and $C^*$ is the
   *optimal* (straight-line) cost.  This model penalises trajectory
   inefficiency and rewards early approach.

From the posterior time-series we derive **six named legibility scores**:

+----------------------------+----------------------------------------------+
| Score                      | What it captures                             |
+============================+==============================================+
| ``L_posterior``             | Time-weighted avg of $P(g^*|\xi_{0:t})$      |
+----------------------------+----------------------------------------------+
| ``L_ipf``                  | Summed *information potential* (Shi 2025)     |
+----------------------------+----------------------------------------------+
| ``L_entropy_auc``          | Area under the normalised entropy curve      |
+----------------------------+----------------------------------------------+
| ``L_intent_info_rate``     | Early KL-divergence accumulation rate        |
+----------------------------+----------------------------------------------+
| ``L_half_time``            | Normalised time to reach half-max-entropy     |
+----------------------------+----------------------------------------------+
| ``L_early_intent``         | Mean $P(g^*)$ in the first 30 % of motion    |
+----------------------------+----------------------------------------------+
| ``L_commitment``           | Normalised intent commitment point           |
+----------------------------+----------------------------------------------+
| ``L_geometric``            | Goal-relative curvature (Frenet–Serret)      |
+----------------------------+----------------------------------------------+
| ``L_composite``            | Principled weighted combination              |
+----------------------------+----------------------------------------------+

All scores lie in $[0, 1]$ (higher = *more legible*).

Auto-Calibration
~~~~~~~~~~~~~~~~

The spatial scale $\sigma$ is set as:

$$ \sigma = \frac{d_{\min}}{2\sqrt{2 \ln 2}} $$

where $d_{\min}$ is the minimum pairwise inter-goal distance.  This
choice places the Gaussian's **half-maximum** exactly at the midpoint
between the two closest goals, ensuring that the observer can just barely
distinguish them — the theoretically optimal discrimination scale.

For the cost-ratio model, the rationality parameter $\beta$ is normalised
by $C^*(x_0, g^*)$ so that the likelihood ratio depends only on the
*relative* inefficiency of the trajectory.


Geometric Analysis
~~~~~~~~~~~~~~~~~~

We compute the **discrete Frenet–Serret frame** at each interior point
of the trajectory and project the curvature normal onto the goal
direction to obtain a *signed goal-relative curvature* $\kappa_g(t)$.

$$ \kappa_g(t) = \kappa(t) \, \cos\!\angle(\mathbf{N}(t),\; g^* - x_t) $$

Positive $\kappa_g > 0$ means the trajectory is *curving toward* $g^*$;
negative means *curving away*.  The **geometric legibility score** is the
early-weighted integral of the clamped $\kappa_g$.


References
----------
[1] Dragan, Lee, Srinivasa.  "Legibility and Predictability of Robot
    Motion."  *HRI 2013*.  doi:10.1109/HRI.2013.6483603
[2] Shi, Grislain, Sigaud, Chetouani.  "Controlling Intent Expressiveness
    in Robot Motion with Diffusion Models."  arXiv:2510.12370, 2025.
[3] Bronars, Cheng, Xu.  "Legibility Diffuser: Offline Imitation for
    Intent Expressive Motion."  *RA-L 2024*.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════════════════════

_LOG_EPS = 1e-12          # floor for log arguments
_DIST_EPS = 1e-8          # floor for distance denominators
_MIN_SIGMA = 1e-4         # minimum σ to avoid degenerate Gaussians
_CURVATURE_EPS = 1e-10    # floor for curvature denominator


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LegibilityResult:
    """Container for all legibility scores and diagnostic curves.

    All scalar scores are in [0, 1]  (higher = more legible).
    """

    # ── primary scores ────────────────────────────────────────────────
    L_posterior:        float = 0.0
    L_ipf:              float = 0.0
    L_entropy_auc:      float = 0.0
    L_intent_info_rate: float = 0.0
    L_half_time:        float = 0.0
    L_early_intent:     float = 0.0
    L_commitment:       float = 0.0
    L_geometric:        float = 0.0
    L_composite:        float = 0.0

    # ── auxiliary geometric scores ────────────────────────────────────
    path_efficiency:            float = 0.0   # straight/arc-length  (1 = straight)
    mean_curvature:             float = 0.0   # unsigned average curvature
    mean_speed:                 float = 0.0   # mean ||Δx|| per step

    # ── research-backed efficiency decomposition (Dragan 2013 §III) ───
    # Derived from the cost-ratio observer model where C*(x₀,g) is the
    # optimal (straight-line) cost and C(ξ) is the actual arc length.
    relative_legibility_cost:   float = 0.0  # (arc−straight)/straight ∈ [0,∞)
    cumulative_legibility_cost: float = 0.0  # time-avg Dragan suboptimality
    front_loading_index:        float = 0.5  # fraction of overhead in first 30%

    # ── diagnostic time-series (optional, for plotting) ───────────────
    posteriors:         Optional[np.ndarray] = None   # (T, K)
    entropy_curve:      Optional[np.ndarray] = None   # (T,)
    kl_curve:           Optional[np.ndarray] = None   # (T,)
    curvature_curve:    Optional[np.ndarray] = None   # (T-2,)
    goal_curvature:     Optional[np.ndarray] = None   # (T-2,)

    # ── calibration info ──────────────────────────────────────────────
    sigma:              float = 0.0
    n_goals:            int   = 0
    n_timesteps:        int   = 0

    def as_dict(self, include_curves: bool = False) -> Dict[str, float]:
        """Return a JSON-serialisable dict of scalar scores."""
        d = dict(
            L_posterior=self.L_posterior,
            L_ipf=self.L_ipf,
            L_entropy_auc=self.L_entropy_auc,
            L_intent_info_rate=self.L_intent_info_rate,
            L_half_time=self.L_half_time,
            L_early_intent=self.L_early_intent,
            L_commitment=self.L_commitment,
            L_geometric=self.L_geometric,
            L_composite=self.L_composite,
            path_efficiency=self.path_efficiency,
            mean_curvature=self.mean_curvature,
            mean_speed=self.mean_speed,
            relative_legibility_cost=self.relative_legibility_cost,
            cumulative_legibility_cost=self.cumulative_legibility_cost,
            front_loading_index=self.front_loading_index,
            sigma=self.sigma,
            n_goals=self.n_goals,
            n_timesteps=self.n_timesteps,
        )
        if include_curves:
            if self.posteriors is not None:
                d["posteriors"] = self.posteriors.tolist()
            if self.entropy_curve is not None:
                d["entropy_curve"] = self.entropy_curve.tolist()
            if self.kl_curve is not None:
                d["kl_curve"] = self.kl_curve.tolist()
        return d

    def summary_line(self) -> str:
        """One-line summary for console logging.

        Columns: primary metric (L_early_intent) | efficiency tradeoff
        (path_efficiency / RLC) | robustness check (L_posterior) | composite.
        L_geometric is computed but BROKEN (saturates to ~1.0 for all arcs
        due to Bézier non-arc-length parameterisation) — excluded here.
        """
        return (f"early_intent={self.L_early_intent:.3f}  "
                f"path_eff={self.path_efficiency:.3f}  "
                f"rlc={self.relative_legibility_cost:.3f}  "
                f"fli={self.front_loading_index:.3f}  "
                f"posterior={self.L_posterior:.3f}  "
                f"composite={self.L_composite:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# CORE: AUTO-CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def auto_calibrate_sigma(goals: np.ndarray,
                         sigma_scale: Optional[float] = None) -> float:
    r"""Compute the optimal observer scale parameter σ.

    The default calibration sets σ so that the Gaussian likelihood's
    **half-maximum** ($\exp(-\ln 2)$) falls exactly at the midpoint between
    the two closest goals:

    $$ \sigma = \frac{d_{\min}}{2\sqrt{2 \ln 2}} $$

    This is the theoretically optimal discrimination scale — any smaller
    and the observer is too myopic (can only distinguish goals once very
    close), any larger and the observer is too coarse (all positions look
    equally likely under all goals).

    Parameters
    ----------
    goals : (K, d) array
        Goal positions.
    sigma_scale : float, optional
        Override: set ``σ = sigma_scale × d_min``.  If *None*, use the
        half-maximum calibration.

    Returns
    -------
    sigma : float
        Calibrated scale parameter.
    """
    K = len(goals)
    if K < 2:
        return 0.1  # degenerate: single goal

    dists = pdist(goals)
    d_min = float(np.min(dists))

    if d_min < _DIST_EPS:
        warnings.warn("Coincident goals detected; using fallback σ = 0.01")
        return 0.01

    if sigma_scale is not None:
        return max(float(sigma_scale * d_min), _MIN_SIGMA)

    # Half-maximum calibration: exp(-d² / (2σ²)) = 1/2  at d = d_min/2
    # ⟹  d_min²/4 = 2σ² ln2  ⟹  σ = d_min / (2√(2 ln 2))
    sigma = d_min / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return max(sigma, _MIN_SIGMA)


# ═══════════════════════════════════════════════════════════════════════════
# CORE: OBSERVER MODELS
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_posterior(positions: np.ndarray,
                       goals: np.ndarray,
                       sigma: float) -> np.ndarray:
    r"""Compute the Bayesian posterior under the Gaussian (IPF) model.

    .. math::

        P(g_k \mid x_t)
            = \frac{\exp\!\bigl(-\|x_t - g_k\|^2 / 2\sigma^2\bigr)}
                   {\sum_j \exp\!\bigl(-\|x_t - g_j\|^2 / 2\sigma^2\bigr)}

    with a uniform prior over goals.

    Parameters
    ----------
    positions : (T, d) array
        End-effector positions at each timestep.
    goals : (K, d) array
        Goal positions.
    sigma : float
        Observer scale parameter (from ``auto_calibrate_sigma``).

    Returns
    -------
    posteriors : (T, K) array
        Posterior probability of each goal at each timestep.
    """
    T, d = positions.shape
    K = len(goals)

    # (T, K) distance matrix
    # positions[:, None, :] → (T, 1, d)  ;  goals[None, :, :] → (1, K, d)
    diff = positions[:, None, :] - goals[None, :, :]        # (T, K, d)
    dist_sq = np.sum(diff ** 2, axis=2)                      # (T, K)

    # Numerically stable softmax:  subtract max per row
    log_lik = -dist_sq / (2.0 * sigma ** 2)                  # (T, K)
    log_lik -= log_lik.max(axis=1, keepdims=True)            # shift for stability
    lik = np.exp(log_lik)
    posteriors = lik / lik.sum(axis=1, keepdims=True)

    return posteriors                                         # (T, K)


def cost_ratio_posterior(positions: np.ndarray,
                         goals: np.ndarray,
                         start: np.ndarray,
                         beta: float = 1.0) -> np.ndarray:
    r"""Compute the Bayesian posterior under the cost-ratio model.

    Following Dragan et al. (HRI 2013), an observer who assumes the agent
    acts approximately optimally assigns:

    .. math::

        P(\xi_{0:t} \mid g)
            \propto \exp\Bigl(\beta \bigl[
                C^*(x_0, g) - C(\xi_{0:t}) - C^*(x_t, g)
            \bigr]\Bigr)

    where $C(\xi_{0:t})$ is the cumulative path length up to $t$, and
    $C^*$ denotes the optimal (straight-line) cost.  The exponent is the
    **efficiency surplus**: positive when the trajectory is *more*
    efficient than expected, negative when it is *less*.

    The rationality parameter $\beta$ is normalised internally so that it
    is unit-free w.r.t. the workspace scale.

    Parameters
    ----------
    positions : (T, d) array
        End-effector positions at each timestep.
    goals : (K, d) array
        Goal positions.
    start : (d,) array
        Starting position of the trajectory.
    beta : float
        Rationality parameter (higher → agent more assumed-rational).

    Returns
    -------
    posteriors : (T, K) array
    """
    T, d = positions.shape
    K = len(goals)

    # Cumulative path length  C(ξ_{0:t})
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)  # (T-1,)
    cum_length = np.concatenate([[0.0], np.cumsum(deltas)])        # (T,)

    # Optimal costs
    C_star_start = np.linalg.norm(goals - start[None, :], axis=1) # (K,)
    C_star_current = np.linalg.norm(
        positions[:, None, :] - goals[None, :, :], axis=2)         # (T, K)

    # Normalise β by scale of the problem
    scale = float(np.mean(C_star_start)) + _DIST_EPS
    beta_norm = beta / scale

    # Log-likelihood = β * (C*(start, g) - C(ξ) - C*(x_t, g))
    log_lik = beta_norm * (C_star_start[None, :]
                           - cum_length[:, None]
                           - C_star_current)                       # (T, K)

    # Numerically stable softmax
    log_lik -= log_lik.max(axis=1, keepdims=True)
    lik = np.exp(log_lik)
    posteriors = lik / lik.sum(axis=1, keepdims=True)

    return posteriors                                               # (T, K)


# ═══════════════════════════════════════════════════════════════════════════
# CORE: INFORMATION-THEORETIC MEASURES
# ═══════════════════════════════════════════════════════════════════════════

def _entropy(posteriors: np.ndarray) -> np.ndarray:
    """Shannon entropy of the goal distribution at each timestep.

    Returns (T,) array in nats.  H = 0 → fully certain;
    H = ln(K) → maximum uncertainty (uniform).
    """
    safe = np.maximum(posteriors, _LOG_EPS)
    return -np.sum(posteriors * np.log(safe), axis=1)    # (T,)


def _kl_divergence_from_uniform(posteriors: np.ndarray) -> np.ndarray:
    r"""KL divergence from uniform prior at each timestep.

    $$ D_{\mathrm{KL}}(P_t \| U) = \ln K - H(P_t) $$

    where $U$ is the uniform distribution over K goals.

    Returns (T,) array in nats.  Starts at ~0 and increases as the
    posterior concentrates on a single goal.
    """
    K = posteriors.shape[1]
    H = _entropy(posteriors)
    return np.log(K) - H                                  # (T,)


def _intent_information_rate(kl_curve: np.ndarray) -> np.ndarray:
    """Stepwise KL divergence increments (bits of info per step).

    Returns (T-1,) array.  Peaks indicate steps where the observer
    gained the most information about the intended goal.
    """
    return np.diff(kl_curve)                               # (T-1,)


# ═══════════════════════════════════════════════════════════════════════════
# CORE: DIFFERENTIAL GEOMETRY – FRENET–SERRET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _discrete_curvature(positions: np.ndarray) -> np.ndarray:
    r"""Discrete unsigned curvature via the Menger curvature formula.

    For three consecutive points $x_{i-1}, x_i, x_{i+1}$, the Menger
    curvature is the reciprocal of the circumradius:

    $$ \kappa_i = \frac{2 \| (x_{i+1} - x_i) \times (x_i - x_{i-1}) \|}
                       {\|x_{i+1} - x_i\| \, \|x_i - x_{i-1}\| \,
                        \|x_{i+1} - x_{i-1}\|} $$

    For 2D trajectories the cross-product magnitude is
    $|a_x b_y - a_y b_x|$.

    Returns (T-2,) array of unsigned curvatures.
    """
    T, d = positions.shape
    if T < 3:
        return np.zeros(max(T - 2, 0))

    v1 = positions[1:-1] - positions[:-2]     # (T-2, d)
    v2 = positions[2:]   - positions[1:-1]    # (T-2, d)

    n1 = np.linalg.norm(v1, axis=1)           # (T-2,)
    n2 = np.linalg.norm(v2, axis=1)           # (T-2,)
    n3 = np.linalg.norm(positions[2:] - positions[:-2], axis=1)

    if d == 2:
        cross_mag = np.abs(v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0])
    elif d == 3:
        cross = np.cross(v2, v1)              # (T-2, 3)
        cross_mag = np.linalg.norm(cross, axis=1)
    else:
        # General d: use area of parallelogram via Gram determinant
        # ||a × b|| = sqrt(||a||²||b||² - (a·b)²)
        dots = np.sum(v1 * v2, axis=1)
        cross_mag = np.sqrt(np.maximum(n1**2 * n2**2 - dots**2, 0.0))

    denom = n1 * n2 * n3 + _CURVATURE_EPS
    kappa = 2.0 * cross_mag / denom

    return kappa                               # (T-2,)


def _frenet_normal(positions: np.ndarray) -> np.ndarray:
    """Approximate the unit normal vector at each interior point.

    For a discrete curve, the tangent changes direction at each step.
    The normal is the direction of this change:  N ∝ T'(t) where T
    is the unit tangent.

    Returns (T-2, d) array of unit normal vectors.  Rows where the
    curvature is zero are set to the zero vector.
    """
    T, d = positions.shape
    if T < 3:
        return np.zeros((max(T - 2, 0), d))

    # Discrete tangent vectors
    v1 = positions[1:-1] - positions[:-2]     # (T-2, d)
    v2 = positions[2:]   - positions[1:-1]    # (T-2, d)

    n1 = np.linalg.norm(v1, axis=1, keepdims=True) + _DIST_EPS
    n2 = np.linalg.norm(v2, axis=1, keepdims=True) + _DIST_EPS

    T1 = v1 / n1                              # unit tangent before
    T2 = v2 / n2                              # unit tangent after

    dT = T2 - T1                              # tangent change
    dT_norm = np.linalg.norm(dT, axis=1, keepdims=True) + _DIST_EPS
    N = dT / dT_norm                          # unit normal

    # Zero out normals where curvature ≈ 0
    flat = np.linalg.norm(dT, axis=1) < _CURVATURE_EPS
    N[flat] = 0.0

    return N                                   # (T-2, d)


def _goal_relative_curvature(positions: np.ndarray,
                              goal: np.ndarray) -> np.ndarray:
    r"""Signed goal-relative curvature at each interior point.

    $$ \kappa_g(t) = \kappa(t) \cdot \cos\angle(\mathbf{N}(t),
                     \, g^* - x_t) $$

    Positive: curving *toward* the goal.  Negative: curving *away*.

    Returns (T-2,) array.
    """
    T, d = positions.shape
    if T < 3:
        return np.zeros(max(T - 2, 0))

    kappa = _discrete_curvature(positions)     # (T-2,)
    N = _frenet_normal(positions)              # (T-2, d)

    # Direction toward goal from each interior point
    to_goal = goal[None, :] - positions[1:-1]  # (T-2, d)
    to_goal_norm = np.linalg.norm(to_goal, axis=1, keepdims=True) + _DIST_EPS
    to_goal_unit = to_goal / to_goal_norm

    # Cosine of angle between normal and goal direction
    cos_angle = np.sum(N * to_goal_unit, axis=1)   # (T-2,)

    return kappa * cos_angle                   # (T-2,)


# ═══════════════════════════════════════════════════════════════════════════
# NAMED LEGIBILITY SCORES
# ═══════════════════════════════════════════════════════════════════════════

def _early_weight(T: int, alpha: float = 3.0) -> np.ndarray:
    r"""Time-weighting function $f(t) = \exp(-\alpha \cdot t/T)$.

    Emphasises early portions of the trajectory.  α = 3 gives ~95 %
    weight in the first third.

    Returns (T,) array normalised to sum to 1.
    """
    t_norm = np.linspace(0.0, 1.0, T)
    f = np.exp(-alpha * t_norm)
    return f / f.sum()


def score_posterior_legibility(posteriors: np.ndarray,
                               true_goal_idx: int,
                               alpha: float = 3.0) -> float:
    r"""Dragan et al. (Eq. 1) —  time-weighted average of $P(g^*|\xi_{0:t})$.

    $$ L_{\text{posterior}} =
         \frac{\sum_t f(t) \, P(g^* | \xi_{0:t})}
              {\sum_t f(t)} $$

    Returns a score in [0, 1].
    """
    T = posteriors.shape[0]
    f = _early_weight(T, alpha)
    return float(np.dot(f, posteriors[:, true_goal_idx]))


def score_ipf(posteriors: np.ndarray,
              true_goal_idx: int,
              alpha: float = 3.0) -> float:
    r"""Information Potential Field score (Shi et al. 2025, Eq. 4).

    $$ L_p(\xi) = -\sum_t f(t) \, \phi(x_t | g^*) $$

    where $\phi(x|g^*) = -\log P(g^*|x)$.  We normalise to $[0, 1]$ by
    dividing by the maximum possible value ($\log K$, i.e. the potential
    under a uniform posterior).
    """
    T, K = posteriors.shape
    f = _early_weight(T, alpha)
    phi = -np.log(np.maximum(posteriors[:, true_goal_idx], _LOG_EPS))  # (T,)
    raw = -np.dot(f, phi)            # more negative = worse
    # Normalise: best case is phi = 0 (P=1) everywhere → raw = 0
    #            worst case is phi = ln(K) everywhere   → raw = -ln(K)
    worst = -np.log(1.0 / K)        # ln(K)
    return float(np.clip(1.0 + raw / worst, 0.0, 1.0))


def score_entropy_auc(entropy_curve: np.ndarray, K: int) -> float:
    """Normalised area-under-entropy-curve (lower entropy AUC → more legible).

    Returns 1 − (AUC / max AUC) so that *higher = more legible*.

    AUC = 0  means entropy was zero at every step (perfectly legible).
    AUC = 1  means entropy was maximal at every step (maximally ambiguous).
    """
    H_max = np.log(K)
    if H_max < _LOG_EPS:
        return 1.0
    auc = float(np.mean(entropy_curve / H_max))
    return float(np.clip(1.0 - auc, 0.0, 1.0))


def score_half_time(entropy_curve: np.ndarray, K: int) -> float:
    """Normalised time to reach half-max entropy.

    The "half-max entropy" threshold is $\\frac{1}{2} \\ln K$.  The score
    is $1 - t_{\\text{half}} / T$:  earlier disambiguation → higher score.

    If entropy never drops below the threshold the score is 0.
    """
    H_max = np.log(K)
    threshold = H_max / 2.0
    T = len(entropy_curve)
    for t in range(T):
        if entropy_curve[t] < threshold:
            return float(1.0 - t / T)
    return 0.0


def score_early_intent(posteriors: np.ndarray,
                       true_goal_idx: int,
                       frac: float = 0.3) -> float:
    """Mean $P(g^*)$ in the first *frac* fraction of the trajectory.

    Default: first 30 % — captures the "readable within the first glance"
    quality that psychology research identifies as the critical window for
    human intent inference (Sebanz & Knoblich 2009).
    """
    T = posteriors.shape[0]
    early_T = max(1, int(T * frac))
    return float(np.mean(posteriors[:early_T, true_goal_idx]))


def score_intent_information_rate(kl_curve: np.ndarray,
                                  alpha: float = 3.0) -> float:
    """Early-weighted accumulation rate of KL divergence.

    Measures how quickly the observer's belief diverges from the uniform
    prior.  Normalised to [0, 1] by the maximum possible KL (= ln K,
    which is the theoretical maximum from the first step).
    """
    T = len(kl_curve)
    if T < 2:
        return 0.0

    # IIR = d(KL)/dt  at each step
    iir = np.diff(kl_curve)  # (T-1,)

    # Positive increments only (ignore backwards information)
    iir_pos = np.maximum(iir, 0.0)

    # Early-weighted sum
    f = _early_weight(len(iir_pos), alpha)

    # Max possible IIR: all KL gained in first step = ln(K)
    # We normalise so that score ∈ [0, 1]
    max_kl = kl_curve[-1] if kl_curve[-1] > _LOG_EPS else 1.0
    raw = float(np.dot(f, iir_pos))
    return float(np.clip(raw / max_kl, 0.0, 1.0))


def score_commitment(posteriors: np.ndarray,
                     true_goal_idx: int,
                     theta: float = 0.80) -> float:
    r"""Intent Commitment Point (ICP).

    The earliest timestep $t$ where $P(g^* | \xi_{0:t}) > \theta$.
    Score = $1 - t_{\text{ICP}} / T$.

    θ = 0.80 means the observer is "strongly committed" (80 % confident).
    If commitment never occurs the score is 0.
    """
    T = posteriors.shape[0]
    p_star = posteriors[:, true_goal_idx]
    for t in range(T):
        if p_star[t] > theta:
            return float(1.0 - t / T)
    return 0.0


def score_geometric(positions: np.ndarray,
                    goal: np.ndarray,
                    alpha: float = 3.0) -> float:
    """Goal-relative curvature score.

    .. warning::
        **BROKEN for Bézier-generated trajectories.** This metric saturates
        at ~1.0 for ALL arc configurations because the raw Menger curvature
        explodes on Bézier curves (which are NOT arc-length parameterised):
        ``|P'(t)|`` is very small at t≈0, making κ = 2|cross|/|P'|³ → ∞.
        ``tanh(5 × huge)`` collapses to 1.0 regardless of arc shape.
        Validated empirically: Pearson r ≈ 0, non-monotone across all 20
        arc configurations.  **Do not use for ranking or reporting.**
        Retained for backward compatibility only.

    Early-weighted average of the *positive* component of κ_g(t).
    Normalised to [0, 1] by clamping.  Higher = more curvature toward
    the intended goal in the early portion of the trajectory.
    """
    T = positions.shape[0]
    if T < 3:
        return 0.0

    kappa_g = _goal_relative_curvature(positions, goal)     # (T-2,)

    # Positive part only — we reward curving toward the goal
    kappa_pos = np.maximum(kappa_g, 0.0)

    # Early weighting
    f = _early_weight(len(kappa_pos), alpha)
    raw = float(np.dot(f, kappa_pos))

    # Normalise via a sigmoid-like mapping:  score = tanh(c · raw)
    # The constant c is set so that a "moderately curved" trajectory
    # (κ ~ 1 / inter-goal-distance) maps to ~0.5.
    c = 5.0   # empirically tuned for typical robotics scales
    return float(np.tanh(c * raw))


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _path_efficiency(positions: np.ndarray) -> float:
    """Ratio of straight-line distance to arc length.  In [0, 1]."""
    if len(positions) < 2:
        return 1.0
    arc = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    straight = float(np.linalg.norm(positions[-1] - positions[0]))
    if arc < _DIST_EPS:
        return 1.0
    return float(np.clip(straight / arc, 0.0, 1.0))


def _mean_speed(positions: np.ndarray) -> float:
    """Mean step size ‖Δx‖."""
    if len(positions) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


# ── Research-backed efficiency metrics (Dragan & Srinivasa HRI 2013 §III) ──
#
# Dragan's cost-ratio observer model is:
#   P(ξ_{0:t} | g) ∝ exp(β [ C*(x₀,g) − C(ξ_{0:t}) − C*(xₜ,g) ])
# where C*(x,g) = ‖x − g‖  (straight-line optimal cost) and
#       C(ξ_{0:t}) = Σ_{i=0}^{t-1} ‖x_{i+1} − xᵢ‖  (cumulative arc length).
#
# The per-step suboptimality (efficiency penalty) is:
#   e(t) = C(ξ_{0:t}) + C*(xₜ, g*) − C*(x₀, g*)   ≥ 0
#
# e(t) = 0 for all t on a straight-line trajectory (predictable).
# e(t) > 0 whenever the trajectory deviates from the optimal path.
# At t = T:  e(T) = C(ξ) − C*(x₀, g*)  = total excess path length.
# ─────────────────────────────────────────────────────────────────────────


def _relative_legibility_cost(positions: np.ndarray) -> float:
    """Relative Legibility Cost (RLC) — fraction of excess path vs straight line.

    Directly derived from Dragan & Srinivasa (HRI 2013, §III) cost quantities:

        RLC(ξ) = [C(ξ) − C*(x₀, g*)] / C*(x₀, g*)

    where C(ξ) is the total arc length and C*(x₀,g*) = ‖xᵀ − x₀‖ is the
    straight-line optimal cost. Range [0, ∞): 0 = perfectly predictable
    (no efficiency overhead), higher = more path overhead paid for legibility.

    Note: RLC = 1/path_efficiency − 1 (mathematically equivalent reframing
    of the naive ratio, but grounded in Dragan's C* notation and interpretable
    as "fraction of extra path length paid for intent communication").
    """
    if len(positions) < 2:
        return 0.0
    arc = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    straight = float(np.linalg.norm(positions[-1] - positions[0]))
    if straight < _DIST_EPS:
        return 0.0
    return max(0.0, (arc - straight) / straight)


def _cumulative_legibility_cost(positions: np.ndarray,
                                goal: np.ndarray) -> float:
    """Cumulative Legibility Cost (CLC) — time-integrated Dragan suboptimality.

    Integrates the per-step excess cost e(t) = C(ξ_{0:t}) + C*(xₜ,g*) − C*(x₀,g*)
    over all timesteps and normalises by trajectory length and optimal cost:

        CLC(ξ) = [Σₜ e(t)] / (T · C*(x₀, g*))

    Interpretation relative to RLC (= e(T)/C*):
    - CLC ≈ RLC/2 for a "linear detour" (overhead grows uniformly over time).
    - CLC > RLC/2: overhead is front-loaded (trajectory detours early → legible).
    - CLC < RLC/2: overhead is back-loaded (trajectory curves late → wasteful).

    This is derived from and citable as an extension of Dragan et al. 2013.
    """
    if len(positions) < 2:
        return 0.0
    T = len(positions)
    goal = np.asarray(goal, dtype=float)
    optimal_total = float(np.linalg.norm(goal - positions[0]))
    if optimal_total < _DIST_EPS:
        return 0.0

    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)   # (T-1,)
    cumulative_path = 0.0
    total_excess = 0.0
    for t in range(T):
        if t > 0:
            cumulative_path += steps[t - 1]
        remaining = float(np.linalg.norm(goal - positions[t]))
        total_excess += max(0.0, cumulative_path + remaining - optimal_total)

    return float(total_excess / (T * optimal_total))


def _front_loading_index(positions: np.ndarray,
                         goal: np.ndarray,
                         early_frac: float = 0.3) -> float:
    """Front-Loading Index (FLI) — fraction of efficiency overhead in first 30%.

    Novel metric derived from Dragan 2013 per-step excess costs:

        FLI(ξ) = [Σ_{t=0}^{0.3T} e(t)] / [Σ_{t=0}^{T} e(t)]   ∈ [0, 1]

    where e(t) = C(ξ_{0:t}) + C*(xₜ,g*) − C*(x₀,g*) (per-step excess cost).

    Validated behavior across 20 arc configurations (Spearman r = −1.00 with
    L_early_intent): smaller-detour arcs (less legible) concentrate overhead
    more tightly in the first 30 %, while larger-detour arcs (more legible)
    spread overhead across more of the trajectory.  All arcs have FLI well
    above 0.5 (range 0.17–0.21 vs. neutral 0.5), confirming all trajectories
    are strongly front-loaded.  CLC/RLC ≈ 0.83–0.87 >> 0.5 corroborates this
    globally.  FLI provides finer discrimination within that front-loaded
    structure.

    For a straight-line trajectory (total_excess ≈ 0), FLI is undefined → 0.5.
    """
    if len(positions) < 2:
        return 0.5
    T = len(positions)
    goal = np.asarray(goal, dtype=float)
    optimal_total = float(np.linalg.norm(goal - positions[0]))
    if optimal_total < _DIST_EPS:
        return 0.5

    split = max(1, int(round(early_frac * T)))
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    cumulative_path = 0.0
    early_excess = 0.0
    total_excess = 0.0
    for t in range(T):
        if t > 0:
            cumulative_path += steps[t - 1]
        remaining = float(np.linalg.norm(goal - positions[t]))
        excess = max(0.0, cumulative_path + remaining - optimal_total)
        total_excess += excess
        if t < split:
            early_excess += excess

    if total_excess < _DIST_EPS:
        return 0.5   # straight line — overhead undefined, return neutral
    return float(np.clip(early_excess / total_excess, 0.0, 1.0))
    if len(positions) < 2:
        return 0.5
    T = len(positions)
    goal = np.asarray(goal, dtype=float)
    optimal_total = float(np.linalg.norm(goal - positions[0]))
    if optimal_total < _DIST_EPS:
        return 0.5

    split = max(1, int(round(early_frac * T)))
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    cumulative_path = 0.0
    early_excess = 0.0
    total_excess = 0.0
    for t in range(T):
        if t > 0:
            cumulative_path += steps[t - 1]
        remaining = float(np.linalg.norm(goal - positions[t]))
        excess = max(0.0, cumulative_path + remaining - optimal_total)
        total_excess += excess
        if t < split:
            early_excess += excess

    if total_excess < _DIST_EPS:
        return 0.5   # straight line — overhead undefined, return neutral
    return float(np.clip(early_excess / total_excess, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════════════

# Default composite weights (can be overridden).
# NOTE: L_geometric is intentionally excluded — it saturates at 1.0 for all
# arcs due to Bézier non-arc-length parameterisation (validated empirically,
# Pearson r ≈ 0, non-monotone). Its former 0.15 weight is redistributed to
# the two validated primary metrics: L_early_intent and L_posterior.
DEFAULT_WEIGHTS = dict(
    posterior       = 0.30,   # ↑ from 0.25 — Dragan 2013 time-weighted posterior
    early_intent    = 0.35,   # ↑ from 0.25 — PRIMARY: mean P(g*) in first 30%
    commitment      = 0.20,   # unchanged
    entropy_auc     = 0.10,   # unchanged
    info_rate       = 0.05,   # unchanged
)


def composite_score(scores: Dict[str, float],
                    weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted combination of validated legibility scores.

    L_geometric is NOT included in the composite — it saturates at 1.0
    for all arcs due to Bézier non-arc-length parameterisation and is
    therefore useless as a discriminator (validated empirically: r ≈ 0,
    non-monotone across all 20 arc configurations).

    Reported metrics in order of importance:
    1. L_early_intent  (PRIMARY)    — mean P(g*) in first 30 %  [r=+0.982]
    2. path_efficiency (SECONDARY)  — straight/arc-length ratio  [r=−0.978]
    3. L_posterior     (ROBUSTNESS) — time-weighted P(g*)        [r=+0.970]
    4. L_composite     (AGGREGATE)  — weighted combination

    All weights are normalised to sum to 1 internally.
    """
    w = weights or DEFAULT_WEIGHTS
    total_w = sum(w.values())

    val = 0.0
    val += w.get("posterior", 0)    * scores.get("L_posterior", 0)
    val += w.get("early_intent", 0) * scores.get("L_early_intent", 0)
    val += w.get("commitment", 0)   * scores.get("L_commitment", 0)
    # geometric intentionally excluded from composite (broken metric)
    val += w.get("entropy_auc", 0)  * scores.get("L_entropy_auc", 0)
    val += w.get("info_rate", 0)    * scores.get("L_intent_info_rate", 0)

    return float(np.clip(val / total_w, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

def compute_legibility(
    trajectory: np.ndarray,
    goals: np.ndarray,
    true_goal_idx: int,
    *,
    model: str = "gaussian",
    alpha: float = 3.0,
    beta: float = 1.0,
    sigma_scale: Optional[float] = None,
    commitment_theta: float = 0.80,
    early_frac: float = 0.30,
    composite_weights: Optional[Dict[str, float]] = None,
    return_curves: bool = False,
) -> LegibilityResult:
    r"""Compute all task-agnostic legibility scores for a trajectory.

    This is the **primary public API**.  It computes the full suite of
    legibility scores without requiring any task-specific thresholds.

    Parameters
    ----------
    trajectory : (T, d) array
        End-effector positions at each timestep (d = 2 or 3).
    goals : (K, d) array
        Goal positions (same dimensionality as trajectory).
    true_goal_idx : int
        Index of the intended goal in the ``goals`` array.
    model : ``"gaussian"`` | ``"cost_ratio"`` | ``"both"``
        Observer likelihood model.  ``"gaussian"`` uses the Information
        Potential Field (Shi 2025).  ``"cost_ratio"`` uses the trajectory-
        efficiency model (Dragan 2013).  ``"both"`` averages both posteriors.
    alpha : float
        Early-emphasis decay rate for $f(t) = \exp(-\alpha t)$.  Higher
        values concentrate weight on earlier timesteps.
    beta : float
        Rationality parameter for the cost-ratio model.
    sigma_scale : float, optional
        Override for σ calibration (σ = sigma_scale × d_min).
    commitment_theta : float
        Posterior threshold for the intent commitment point.
    early_frac : float
        Fraction of trajectory considered "early" for ``L_early_intent``.
    composite_weights : dict, optional
        Override composite score weights.
    return_curves : bool
        If True, populate the diagnostic time-series arrays.

    Returns
    -------
    LegibilityResult
        Container with all scores, calibration info, and (optionally)
        diagnostic curves.
    """
    # ── input validation ──────────────────────────────────────────────
    trajectory = np.asarray(trajectory, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)

    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must be (T, d), got shape {trajectory.shape}")
    if goals.ndim != 2:
        raise ValueError(f"goals must be (K, d), got shape {goals.shape}")

    T, d = trajectory.shape
    K = goals.shape[0]

    if goals.shape[1] != d:
        raise ValueError(f"dimension mismatch: trajectory d={d}, goals d={goals.shape[1]}")
    if not (0 <= true_goal_idx < K):
        raise ValueError(f"true_goal_idx={true_goal_idx} out of range [0, {K})")
    if T < 2:
        warnings.warn("Trajectory has < 2 points; returning default scores.")
        return LegibilityResult(n_goals=K, n_timesteps=T)

    g_star = goals[true_goal_idx]
    start = trajectory[0]

    # ── auto-calibrate ────────────────────────────────────────────────
    sigma = auto_calibrate_sigma(goals, sigma_scale)

    # ── compute posteriors ────────────────────────────────────────────
    if model == "gaussian":
        posteriors = gaussian_posterior(trajectory, goals, sigma)
    elif model == "cost_ratio":
        posteriors = cost_ratio_posterior(trajectory, goals, start, beta)
    elif model == "both":
        p_gauss = gaussian_posterior(trajectory, goals, sigma)
        p_cost  = cost_ratio_posterior(trajectory, goals, start, beta)
        # Geometric mean of the two posteriors (then re-normalise)
        combined = np.sqrt(p_gauss * p_cost)
        posteriors = combined / combined.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown model '{model}'; use 'gaussian', "
                         f"'cost_ratio', or 'both'")

    # ── information-theoretic curves ──────────────────────────────────
    entropy_curve = _entropy(posteriors)                       # (T,)
    kl_curve = _kl_divergence_from_uniform(posteriors)         # (T,)

    # ── individual scores ─────────────────────────────────────────────
    L_post  = score_posterior_legibility(posteriors, true_goal_idx, alpha)
    L_ipf   = score_ipf(posteriors, true_goal_idx, alpha)
    L_eauc  = score_entropy_auc(entropy_curve, K)
    L_iir   = score_intent_information_rate(kl_curve, alpha)
    L_half  = score_half_time(entropy_curve, K)
    L_early = score_early_intent(posteriors, true_goal_idx, early_frac)
    L_comm  = score_commitment(posteriors, true_goal_idx, commitment_theta)
    L_geom  = score_geometric(trajectory, g_star, alpha)

    # ── composite score ───────────────────────────────────────────────
    all_scores = dict(
        L_posterior=L_post, L_ipf=L_ipf,
        L_entropy_auc=L_eauc, L_intent_info_rate=L_iir,
        L_half_time=L_half, L_early_intent=L_early,
        L_commitment=L_comm, L_geometric=L_geom,
    )
    L_comp = composite_score(all_scores, composite_weights)

    # ── geometric diagnostics ─────────────────────────────────────────
    kappa = _discrete_curvature(trajectory)                    # (T-2,)
    kappa_g = _goal_relative_curvature(trajectory, g_star)     # (T-2,)
    eff = _path_efficiency(trajectory)
    spd = _mean_speed(trajectory)
    rlc = _relative_legibility_cost(trajectory)
    clc = _cumulative_legibility_cost(trajectory, g_star)
    fli = _front_loading_index(trajectory, g_star, early_frac)

    # ── assemble result ───────────────────────────────────────────────
    res = LegibilityResult(
        L_posterior=L_post,
        L_ipf=L_ipf,
        L_entropy_auc=L_eauc,
        L_intent_info_rate=L_iir,
        L_half_time=L_half,
        L_early_intent=L_early,
        L_commitment=L_comm,
        L_geometric=L_geom,
        L_composite=L_comp,
        path_efficiency=eff,
        mean_curvature=float(np.mean(kappa)) if len(kappa) > 0 else 0.0,
        mean_speed=spd,
        relative_legibility_cost=rlc,
        cumulative_legibility_cost=clc,
        front_loading_index=fli,
        sigma=sigma,
        n_goals=K,
        n_timesteps=T,
    )

    if return_curves:
        res.posteriors = posteriors
        res.entropy_curve = entropy_curve
        res.kl_curve = kl_curve
        res.curvature_curve = kappa
        res.goal_curvature = kappa_g

    return res


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT-SPECIFIC ADAPTER  (TwoBlockPickEnv)
# ═══════════════════════════════════════════════════════════════════════════

def compute_legibility_from_obs(
    obs_traj: np.ndarray,
    target_block: str,
    **kwargs,
) -> LegibilityResult:
    """Convenience wrapper for the TwoBlockPickEnv observation format.

    Extracts end-effector positions and goal (block) positions from the
    22-d observation array and calls ``compute_legibility``.

    Parameters
    ----------
    obs_traj : (T, 22) array
        Observation trajectory from ``TwoBlockPickEnv``.
    target_block : ``"LEFT"`` | ``"RIGHT"``
        Which block is the intended target.
    **kwargs
        Forwarded to ``compute_legibility``.

    Observation layout (22-d):
        [0:3]   ee_pos      (x, y, z)
        [3:7]   ee_quat     (4)
        [7]     gripper     (1)
        [8:11]  left_cube   (x, y, z)
        [11:15] left_quat   (4)
        [15:18] right_cube  (x, y, z)
        [18:22] right_quat  (4)
    """
    obs_traj = np.asarray(obs_traj, dtype=np.float64)

    if obs_traj.ndim != 2 or obs_traj.shape[1] < 18:
        raise ValueError(
            f"Expected (T, ≥18) observation array, got {obs_traj.shape}")

    ee_positions = obs_traj[:, :3]                        # (T, 3)

    # Use first observation for stable goal extraction (cubes haven't moved)
    goal_left  = obs_traj[0, 8:11].copy()                # (3,)
    goal_right = obs_traj[0, 15:18].copy()                # (3,)

    goals = np.stack([goal_left, goal_right], axis=0)     # (2, 3)

    target_block_upper = target_block.upper().strip()
    if target_block_upper == "LEFT":
        true_goal_idx = 0
    elif target_block_upper == "RIGHT":
        true_goal_idx = 1
    else:
        raise ValueError(f"target_block must be 'LEFT' or 'RIGHT', "
                         f"got '{target_block}'")

    return compute_legibility(ee_positions, goals, true_goal_idx, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def plot_legibility_diagnostics(
    result: LegibilityResult,
    trajectory: Optional[np.ndarray] = None,
    goals: Optional[np.ndarray] = None,
    true_goal_idx: int = 0,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Generate a 2×2 diagnostic plot for a legibility analysis.

    Panels:
      - Top-left:     Goal posteriors over time
      - Top-right:    Entropy curve & KL divergence
      - Bottom-left:  Trajectory top-down view with goals (if provided)
      - Bottom-right: Score radar chart

    Requires matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title or "Legibility Diagnostics", fontsize=14, fontweight="bold")

    # ── Panel 1: Posteriors ───────────────────────────────────────────
    ax = axes[0, 0]
    if result.posteriors is not None:
        T_plot, K_plot = result.posteriors.shape
        t_axis = np.arange(T_plot)
        for k in range(K_plot):
            style = "-" if k == true_goal_idx else "--"
            label_k = f"Goal {k}" + (" ★" if k == true_goal_idx else "")
            ax.plot(t_axis, result.posteriors[:, k], style, label=label_k, lw=2)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax.axhline(0.8, color="green", ls=":", alpha=0.3, label="θ=0.80")
        ax.set_ylabel("P(goal | trajectory)")
        ax.set_xlabel("Timestep")
        ax.set_title("Goal Posteriors")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No posterior data\n(set return_curves=True)",
                ha="center", va="center", transform=ax.transAxes)

    # ── Panel 2: Entropy & KL ─────────────────────────────────────────
    ax = axes[0, 1]
    if result.entropy_curve is not None:
        t_axis = np.arange(len(result.entropy_curve))
        ax.plot(t_axis, result.entropy_curve, "b-", lw=2, label="Entropy H(G|ξ)")
        ax2 = ax.twinx()
        if result.kl_curve is not None:
            ax2.plot(t_axis, result.kl_curve, "r-", lw=2, label="KL from uniform")
            ax2.set_ylabel("KL divergence (nats)", color="red")
        ax.set_ylabel("Entropy (nats)", color="blue")
        ax.set_xlabel("Timestep")
        ax.set_title("Information Dynamics")
        ax.grid(alpha=0.3)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No entropy data", ha="center", va="center",
                transform=ax.transAxes)

    # ── Panel 3: Trajectory top-down ──────────────────────────────────
    ax = axes[1, 0]
    if trajectory is not None and goals is not None:
        # Use X-Y for top-down view (or first two dims)
        traj_xy = trajectory[:, :2]
        ax.plot(traj_xy[:, 0], traj_xy[:, 1], "k-", lw=1.5, alpha=0.6)
        ax.scatter(traj_xy[0, 0], traj_xy[0, 1],
                   s=80, c="blue", zorder=5, label="Start", marker="s")
        ax.scatter(traj_xy[-1, 0], traj_xy[-1, 1],
                   s=80, c="purple", zorder=5, label="End", marker="^")

        colors = ["green", "red", "orange", "cyan"]
        for k in range(len(goals)):
            c = colors[k % len(colors)]
            mk = "★" if k == true_goal_idx else "o"
            ax.scatter(goals[k, 0], goals[k, 1],
                       s=120, c=c, zorder=5,
                       label=f"Goal {k}" + (" ★" if k == true_goal_idx else ""),
                       edgecolors="black", linewidths=1.5)

        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title("Trajectory (top-down)")
        ax.legend(fontsize=8); ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center",
                transform=ax.transAxes)

    # ── Panel 4: Score radar ──────────────────────────────────────────
    ax = axes[1, 1]
    score_names = [
        "L_posterior", "L_early_intent", "L_commitment",
        "L_geometric", "L_entropy_auc", "L_ipf",
    ]
    score_vals = [getattr(result, name) for name in score_names]
    short_names = ["Posterior", "Early\nIntent", "Commit",
                   "Geom", "Entropy\nAUC", "IPF"]

    N_sc = len(score_names)
    angles = np.linspace(0, 2 * np.pi, N_sc, endpoint=False).tolist()
    angles += angles[:1]
    vals = score_vals + score_vals[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax = fig.add_subplot(2, 2, 4, polar=True)
    ax.plot(angles, vals, "b-", lw=2)
    ax.fill(angles, vals, alpha=0.15, color="blue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"Composite = {result.L_composite:.3f}", pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# BATCH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def score_candidates(
    trajectories: List[np.ndarray],
    goals: np.ndarray,
    true_goal_idx: int,
    **kwargs,
) -> List[LegibilityResult]:
    """Score multiple candidate trajectories and return sorted results.

    Parameters
    ----------
    trajectories : list of (T, d) arrays
        Candidate end-effector trajectories.
    goals : (K, d) array
        Goal positions.
    true_goal_idx : int
        Index of the intended goal.
    **kwargs
        Forwarded to ``compute_legibility``.

    Returns
    -------
    results : list of LegibilityResult
        One result per candidate, in the same order as input.
    """
    return [compute_legibility(traj, goals, true_goal_idx, **kwargs)
            for traj in trajectories]


def select_most_legible(
    trajectories: List[np.ndarray],
    goals: np.ndarray,
    true_goal_idx: int,
    score_key: str = "L_composite",
    **kwargs,
) -> Tuple[int, LegibilityResult]:
    """Select the most legible trajectory from a candidate pool.

    Parameters
    ----------
    trajectories : list of (T, d) arrays
    goals : (K, d) array
    true_goal_idx : int
    score_key : str
        Which score to maximise (default: ``L_composite``).
    **kwargs
        Forwarded to ``compute_legibility``.

    Returns
    -------
    best_idx : int
        Index of the most legible trajectory.
    best_result : LegibilityResult
    """
    results = score_candidates(trajectories, goals, true_goal_idx, **kwargs)
    best_idx = max(range(len(results)),
                   key=lambda i: getattr(results[i], score_key))
    return best_idx, results[best_idx]


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    """Quick validation with synthetic trajectories."""
    print("=" * 60)
    print("Task-Agnostic Legibility Metrics — Self-Test")
    print("=" * 60)

    # Two goals at (1, 0.5) and (1, -0.5), start at (0, 0)
    goals = np.array([[1.0, 0.5], [1.0, -0.5]])
    start = np.array([0.0, 0.0])

    T = 50
    t = np.linspace(0, 1, T)

    # ── Trajectory 1: STRAIGHT to goal 0 (predictable but not very legible)
    traj_straight = np.column_stack([t * 1.0, t * 0.5])

    # ── Trajectory 2: CURVED toward goal 0 (legible — early deviation)
    # Exaggerated upward arc that curves toward goal 0
    traj_legible = np.column_stack([
        t * 1.0,
        0.5 * t + 0.3 * np.sin(np.pi * t)   # extra upward hump
    ])

    # ── Trajectory 3: AMBIGUOUS (goes straight then decides late)
    traj_ambiguous = np.column_stack([
        t * 1.0,
        0.5 * np.where(t < 0.7, 0.0, (t - 0.7) / 0.3) * 0.5
    ])

    for name, traj in [("STRAIGHT", traj_straight),
                       ("LEGIBLE (curved)", traj_legible),
                       ("AMBIGUOUS (late)", traj_ambiguous)]:
        res = compute_legibility(traj, goals, true_goal_idx=0,
                                 model="gaussian", return_curves=True)
        print(f"\n  {name}:")
        print(f"    {res.summary_line()}")
        print(f"    IPF={res.L_ipf:.3f}  entropy_auc={res.L_entropy_auc:.3f}  "
              f"half_time={res.L_half_time:.3f}  IIR={res.L_intent_info_rate:.3f}")
        print(f"    σ={res.sigma:.4f}  efficiency={res.path_efficiency:.3f}")

    # Test with 3D / TwoBlockPick-like setup
    print("\n  --- 3D (TwoBlockPick-like) ---")
    goals_3d = np.array([[0.50, 0.07, 0.42], [0.50, -0.07, 0.42]])
    start_3d = np.array([0.40, 0.0, 0.55])

    # Legible: curves toward left block with dip
    t3 = np.linspace(0, 1, 100)
    traj_3d_legible = np.column_stack([
        0.40 + 0.10 * t3,
        0.15 * np.sin(np.pi * t3),                # exaggerated Y arc toward left
        0.55 - 0.13 * t3,
    ])

    res3d = compute_legibility(traj_3d_legible, goals_3d, true_goal_idx=0,
                                model="both", return_curves=True)
    print(f"    3D LEGIBLE:  {res3d.summary_line()}")

    print("\n  ✓ Self-test passed.\n")


if __name__ == "__main__":
    _self_test()
