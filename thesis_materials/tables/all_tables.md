# Thesis Tables — Ready for LaTeX or Markdown

All tables below are derived from experimental results. Copy directly into thesis.

---

## Table 5.1: Base Policy Evaluation (50 Episodes)

| Metric | Value |
|--------|-------|
| Success Rate | 84% (42/50) |
| Mean Episode Steps | 344 |
| Left-Block Picks (of successes) | 31/42 (73.8%) |

---

## Table 5.2: Demonstration Dataset Summary

| Component | Specification |
|-----------|--------------|
| Total demos | 400 |
| Legible | 200 (50%) |
| Neutral | 100 (25%) |
| Deceptive | 100 (25%) |
| Configurations | 10 (block placements) |
| Demos per config | 40 |
| Episode length | 400 steps |
| obs_dim / act_dim | 22 / 5 |
| Training chunks | 6,920 (horizon=32, stride=2) |

---

## Table 5.3: VLM Expert Demo Evaluation (Gemini 2.5 Flash)

| Style | N | Goal Accuracy | Mean VLO | Std VLO |
|-------|---|:------------:|:--------:|:-------:|
| Legible | — | — | 2.93 | — |
| Neutral | — | — | 3.00 | — |
| Deceptive | — | — | 3.71 | — |
| **Overall** | **40** | **94.7%** | — | — |

VLO ordering: legible (2.93) < neutral (3.00) < deceptive (3.71) ✓

VLM v2 full accuracy: 97.5% (up from 45% in v1)

---

## Table 5.4: Baseline Policy VLM Legibility Assessment (42 Successful Episodes)

| Metric | Value |
|--------|-------|
| Mean VLO | 4.57 / 6 |
| Median VLO | 6.0 |
| Episodes never identified | 26/42 (61.9%) |
| k (prefix windows) | 6 |

---

## Table 5.5: Classifier Guidance Scale Sweep

| Scale (w) | Success Rate | L_early | Δ vs w=0 |
|:---------:|:-----------:|:-------:|:---------:|
| 0 | 95% | 0.906 | — |
| 5 | 90% | 0.946 | +0.040 |
| **10** | **100%** | **0.952** | **+0.046** |
| 20 | 100% | 0.948 | +0.042 |

Optimal: w=10 (highest L_early at 100% success)

---

## Table 5.6: Best-of-N Legibility Scaling

| N | L_early | Δ vs N=1 | Improvement |
|---|:-------:|:--------:|:----------:|
| 1 | 0.732 | — | — |
| 4 | 0.779 | +0.047 | +6.4% |
| 8 | 0.797 | +0.065 | +8.8% |
| 16 | 0.806 | +0.074 | +10.1% |

---

## Table 5.7: Full Pipeline — Staged Results (20 Paired Episodes)

| Stage | Method | Success | L_early | Δ L_early |
|:-----:|--------|:-------:|:-------:|:---------:|
| 0 | Baseline diffusion policy | 80% | 0.898 | — |
| 1 | + Classifier guidance (w=10) | 100% | 0.937 | +0.039 |
| 2 | + VLM reranking (K=5) | **100%** | **0.972** | **+0.074** |

**Statistical significance:** Stage 2 vs Stage 0: p = 0.00042 (paired Wilcoxon)

---

## Table 5.8: All Guidance Methods Compared

| Method | Training-Free? | Success | L_early | Notes |
|--------|:--------------:|:-------:|:-------:|-------|
| Base policy | — | 84% | 0.732 | Mostly illegible |
| Best-of-16 | ✓ | — | 0.806 | Simple, diminishing returns |
| LegDiff (CFG, w=3) | ✗ | 100% | 0.935 | Requires conditional training |
| Classifier guidance (w=10) | ✓ | **100%** | 0.952 | LLM-generated scoring function |
| VLM reranking (K=5) | ✓ | **100%** | **0.972** | Best overall, higher latency |

---

## Table 5.9: True DPS Verification

| Method | Success Rate | Notes |
|--------|:-----------:|-------|
| Classifier guidance (w=10) | 100% | Gradient through scoring function |
| True DPS (Song et al., 2023) | **0%** | Actions explode, denoising fails |

Confirms: method is classifier guidance (Dhariwal & Nichol, 2021), NOT DPS.

---

## Table 5.10: Model Architecture Summary

| Component | Details |
|-----------|---------|
| Parameters | ~5.5M |
| Time embedding | Sinusoidal(128) → MLP(256) |
| Obs embedding | MLP(22→256→256) |
| Encoder channels | [256, 512, 1024] |
| Bottleneck | 1024→1024 |
| Decoder channels | [1024→512, 512→256] (skip) |
| Output | MLP(256→256→5) |
| Noise schedule | β: 0.0001→0.1 (linear) |
| Training steps | 100 (DDPM) |
| Inference steps | 10 (DDIM, η=0.3) |
| EMA decay | 0.999 |
| Horizon / Execute | 32 / 8 |
| Batch size / LR | 64 / 2e-4 |

---

## Table 5.11: Honest Assessment Summary

| Issue | Severity | Impact |
|-------|----------|--------|
| Terminology: classifier guidance, not DPS | High | Must correct in all text |
| Text-only VLM scoring (no visual input) | Medium | Any LLM could produce similar |
| Single-shot, not iterative EUREKA | Medium | No environment feedback loop |
| No human study | High | VLM scores not validated by humans |
| No reverse-steering (w<0) | Medium | Cannot prove causal effect |
| Same train/test block configs | Medium | Generalization undemonstrated |

---

## LaTeX-Ready Table Template

```latex
\begin{table}[t]
\centering
\caption{Full Pipeline Results (20 Paired Episodes)}
\label{tab:pipeline}
\begin{tabular}{clcc}
\toprule
Stage & Method & Success & $L_{\text{early}}$ \\
\midrule
0 & Baseline diffusion policy & 80\% & 0.898 \\
1 & + Classifier guidance ($w{=}10$) & 100\% & 0.937 \\
2 & + VLM reranking ($K{=}5$) & \textbf{100\%} & \textbf{0.972} \\
\bottomrule
\end{tabular}
\end{table}
```
