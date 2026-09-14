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

### 2. Solution: High-Exploration Rollouts ($K = 8$) with Gradient Accumulation ($G = 8$)
* **Intra-Prompt Exploration ($K = 8$)**: Sampling 8 completions per prompt nearly doubles the probability of encountering rare hallucinations within each prompt compared to $K=4$ ($~33.7\%$ vs $~18.5\%$ at $p=0.05$), unlocking non-zero RLOO leave-one-out baselines.
* **Invariant Peak HBM Memory ($B = 4, K = 8$)**: Halving `per_device_train_batch_size` to 4 while doubling $K$ to 8 keeps simultaneous parallel sequences per device at exactly $4 \times 8 = 32$ (identical to the previous $8 \times 4 = 32$). Peak KV-cache and backprop activation memory remain completely unchanged.
* **Effective Batch Size ($G = 8$)**: With $B=4$ per device across 2 GPUs (8 prompts / micro-batch), accumulating over $G=8$ steps maintains the canonical effective batch size of $4 \times 2 \times 8 = 64$ prompts (512 rollouts per optimizer step).
* **Rollout Divisibility**: $16 \div 8 = 2$ optimizer updates per 16-step generation phase ($16 \% 8 == 0$).
* **DeepSpeed Integration**: Configured `gradient_accumulation_steps: auto` in `scripts/deepspeed_config.yaml` to dynamically synchronize with CLI and Accelerate arguments.


