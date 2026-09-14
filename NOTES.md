# Technical Notes & Architectural Decisions

## Reward Scoring in PE-RL: Logit Difference vs. Softmax Probabilities

### 1. Context & Observation
Reward models in this repository are sequence classifiers trained with binary cross-entropy:
* **Label 0**: Hallucinated / Non-factual (`"Yes"`)
* **Label 1**: Faithful / Factual (`"No"`)

During hyperparameter search on synthetic/augmented data:
1. Reward models achieve high ROC-AUC ($>0.95$) and low evaluation loss.
2. However, the optimal classification threshold $\tau$ (maximizing Youden's $J = \text{TPR} - \text{FPR}$) frequently shifts to extreme values close to 1 (e.g. $\tau \approx 0.99952$).

---

### 2. Why Saturated Probabilities Fail in RL (PE-RL / RLOO)
Passing raw softmax probabilities $R(x, y) = P(\text{Label 1}) = \sigma(z_1 - z_0)$ into RLOO creates three major issues:

1. **Advantage & Gradient Vanishing (Score Compression)**:
   * Confident models compress almost all outputs into $[0.9990, 0.9999]$.
   * The reward delta between rollouts becomes microscopic ($\Delta R \approx 10^{-4}$).
   * In RLOO, policy gradient updates scale with $(R(y_i) - b)$, causing policy updates to stall unless unreliably large learning rates are used.
2. **Sigmoid Flat Region**:
   * The gradient of the sigmoid $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ approaches zero when $\sigma(z) \to 1.0$.
   * A great completion ($z_1 - z_0 = 12$) and an acceptable completion ($z_1 - z_0 = 7$) both map to $P \approx 0.9999$, discarding fine-grained preference distinction.
3. **Threshold Centering Pitfall ($\frac{P - \tau}{1 - \tau}$)**:
   * Scaling by $\frac{1}{1 - \tau}$ when $\tau = 0.99952$ creates a $\approx 2083\times$ amplification above the threshold vs. $1\times$ below it, causing massive gradient asymmetry and training instability.

---

### 3. The Adopted Solution: Logit Difference ($z_1 - z_0$)
In `src/pipelines.py` (`PERLPipeline.setup_trainer`), rewards are computed directly as the unconstrained logit difference (log-odds):

$$R(x, y) = z_1 - z_0 = \text{logits}[\text{Label 1}] - \text{logits}[\text{Label 0}]$$

```python
# In PERLPipeline.setup_trainer:
logits = reward_model(**inputs).logits
rewards = (logits[:, 1] - logits[:, 0]).cpu().tolist()
return rewards
```

---

### 4. Key Advantages & Justification

* **Exact Bradley-Terry Alignment**:
  Under the Bradley-Terry preference model, $P(y_1 \succ y_0) = \sigma(r(y_1) - r(y_0))$. For binary cross-entropy:
  $$\sigma(z_1 - z_0) = P \implies z_1 - z_0 = \log\left(\frac{P}{1 - P}\right) = r(x, y)$$
  $z_1 - z_0$ is mathematically the true implicit scalar reward.
* **100% Backward-Compatible (Zero Retraining)**:
  Existing trained reward model checkpoints are used as-is. No architectural changes or fine-tuning required.
* **Linear Dynamic Range**:
  Separations between poor ($-5$), decent ($+3$), and exceptional ($+10$) completions remain linear and unconstrained.
* **Natural Fit with RLOO Dynamic Baselines**:
  RLOO automatically subtracts the leave-one-out baseline $\frac{1}{K-1}\sum_{j \neq i} R_j$ per prompt. Logit differences provide clean, uncompressed relative advantage values $A_i$.
* **Robust to Distribution Drift**:
  Unlike a static empirical threshold $\tau$, logit differences do not degrade when the generation policy drifts out-of-distribution during RL training.

---

## Gradient Accumulation & Signal Sparsity in PE-RL (RLOO)

### 1. Problem: Sparse Signals in Rollout Batches
Because the reward model measures rare events (hallucinations), rollouts at low batch sizes ($B_{\text{eff}} = 16$ prompts across 2 GPUs) frequently yield batches where zero prompts exhibit hallucination variance. Under RLOO's leave-one-out baseline ($A_i = r_i - \frac{1}{K-1}\sum_{j \neq i} r_j$), homogeneous rollouts produce zero advantage ($A_i \approx 0$), resulting in uninformative policy updates dominated by KL drift.

### 2. Solution: Gradient Accumulation ($G = 4$)
* **Effective Batch Size**: With `per_device_train_batch_size = 8` on 2 GPUs and $G = 4$, the effective batch size scales to $8 \times 2 \times 4 = 64$ prompts (256 rollouts per update).
* **Signal Coverage**: At a ~5% event probability, the chance that an update contains at least one informative prompt increases from ~56% ($G=1$) to >96% ($G=4$).
* **Rollout Divisibility**: Neatly divides `steps_per_generation = 16`, producing exactly 4 optimizer updates per rollout phase.
* **HBM Neutral**: Accumulates gradients in-place in `param.grad` and discards activations after each micro-batch backward pass, preventing OOM on 80GB HBM.
* **DeepSpeed Integration**: Configured `gradient_accumulation_steps: auto` in `scripts/deepspeed_config.yaml` to dynamically synchronize with CLI and Accelerate arguments.

