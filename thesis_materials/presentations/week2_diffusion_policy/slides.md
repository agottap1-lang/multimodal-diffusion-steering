# Week 2: Diffusion Policy Architecture & Training

## DDPM-based Action Chunk Prediction

**Anudeep Gottapu**
Arizona State University

---

## Slide 1: Why Diffusion Policy?

**Advantages over Traditional Policy Classes**

1. **Multimodal distribution modeling** — Captures multiple trajectory styles (legible/neutral/deceptive) in one model
2. **Training-free controllability** — The iterative denoising process allows gradient-based steering without retraining
3. **Chunk-based prediction** — 32-step action chunks provide temporal coherence for smooth trajectories
4. **State-of-the-art performance** — Chi et al. (2023) showed DDPM-based policies outperform explicit policy classes on contact-rich tasks

**Key insight:** By learning a *distribution* over trajectories, we can later steer that distribution toward legibility.

---

## Slide 2: DDPM Training Formulation

**Forward Process (Add Noise)**

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

**Training Objective (Predict Noise)**

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**Reverse Process (Generate Actions)**

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

- Linear noise schedule: β_start=0.0001, β_end=0.1, T=100 steps
- ε-prediction (no tanh on output — predicts unbounded noise)

---

## Slide 3: U-Net Architecture

**Architecture Diagram**

```
Input: noisy_actions (B, 32, 5)
                ↓
        input_proj: Linear(5→256)
                ↓
    +— time_embed: Sinusoidal(128)→MLP(128→256→256) —→ [additive conditioning]
    +— obs_embed:  MLP(22→256→256) ——————————————————→ [additive conditioning]
                ↓
        Encoder Block 1: UNetBlock(256→512)    ——skip₁——→
                ↓
        Encoder Block 2: UNetBlock(512→1024)   ——skip₂——→
                ↓
        Bottleneck: UNetBlock(1024→1024)
                ↓
        Decoder Block 1: UNetBlock(2048→512)   ←—cat(skip₂)—
                ↓
        Decoder Block 2: UNetBlock(1024→256)   ←—cat(skip₁)—
                ↓
        output_proj: MLP(256→256→5)
                ↓
Output: predicted_noise (B, 32, 5)
```

Each UNetBlock: Linear + GroupNorm(8) + Mish + time_proj(additive) + residual shortcut

---

## Slide 4: Training Configuration

| Parameter | Combined Model | Original Model |
|-----------|:-------------:|:--------------:|
| Demo data | 400 (combined) | 200 (left/right) |
| Hidden dim | 256 | 256 |
| Encoder blocks | 3 | 6 |
| Batch size | 64 | 256 |
| Learning rate | 2e-4 | 1e-4 |
| Weight decay | 1e-5 | — |
| EMA decay | 0.999 | 0.999 |
| Epochs | 100 | 500 |
| Mirror augment | No | Yes |

**Key change:** Fewer blocks (3 vs 6) and smaller batch size — sufficient for 400 mixed demos

---

## Slide 5: DDIM Inference

**10-Step Fast Sampling**

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon$$

- 100-step training → 10-step DDIM inference (10× speedup)
- η = 0.3 (partial stochasticity for diversity)
- Temporal ensemble: weighted blend of overlapping predictions

**Execution:** Predict 32-step chunk, execute first 8 steps, re-predict. This provides both planning horizon (32) and responsiveness (replan every 8 steps).

---

## Slide 6: Training Results

**Loss Curve (100 Epochs)**

- Epoch 0: loss ≈ 0.154
- Epoch 50: loss ≈ 0.065
- Epoch 100: loss ≈ 0.045

**Chunking Statistics:**
- 6,920 chunks extracted from 400 demos
- Each chunk: (obs, action_sequence) of shape (22,) and (32, 5)
- Valid chunks sampled from within episodes at random offsets

---

## Slide 7: Base Policy Evaluation

**50 Episodes, Epoch 100 Checkpoint**

| Metric | Value |
|--------|-------|
| **Success Rate** | **84%** (42/50) |
| Mean Episode Length | 344 steps |
| Picked Left | 31/42 (73.8%) |
| Picked Right | 11/42 (26.2%) |

**Observations:**
- Left-side bias reflects demo distribution (legible demos favor clear left commitment)
- 16% failure rate — some episodes fail to lift block above threshold
- Policy successfully learns multi-style distribution but doesn't distinguish styles at inference

---

## Slide 8: Earlier Training Challenges

**Debugging Journey**

1. **Initial 13% success rate** — Root cause: checkpoint bug saving normalized stats (mean=0, std=1) instead of real demo statistics
2. **Fix:** Correct checkpoint saving → 20-25% → architecture refinements → 84%
3. **87% failure in early eval** — 100 rollouts across 10 seeds showed modal collapse in some seeds
4. **Key lesson:** Normalization statistics must match inference conditions exactly

---

## Slide 9: Key Takeaways

1. **DDPM successfully learns multi-style trajectory distribution** from mixed demos
2. **84% task success** demonstrates viable base policy
3. **But base policy is largely illegible** (VLO assessment in Week 3)
4. **Training-free steering** is the key advantage — can modify behavior without retraining
5. **Architecture is simple** — 1D U-Net with ResBlocks, not transformer-based

---

## Slide 10: References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models.* ICLR.
- Chi, C., Feng, S., Du, Y., et al. (2023). *Diffusion Policy.* RSS.
