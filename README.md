# Ternary Transformer Lab

An empirical investigation into ternary-weight transformers trained from scratch,
comparing their accuracy, convergence behaviour, memory footprint, and inference
speed against float32 equivalents across a suite of synthetic mathematical sequence
tasks. All experiments are conducted at small scale to enable rapid iteration and
clean hypothesis testing, with the explicit goal of characterising the conditions
under which ternary weight representations are viable for practical deployment.

---

## Overview

Ternary neural networks constrain every weight to one of three values: -1, 0, or +1.
This eliminates floating-point multiply-accumulate operations at inference time,
replacing them with additions, subtractions, and zero-skips. The theoretical
inference efficiency gain is substantial, particularly on memory-bandwidth-limited
hardware such as microcontrollers. The open question is whether these efficiency
gains come at an unacceptable accuracy cost, and under what conditions that cost
can be reduced to negligible levels.

This project trains both a standard float32 transformer and an identical architecture
with ternary-quantized feed-forward weights under controlled conditions, measuring
accuracy, convergence speed, weight distribution dynamics, and inference performance
across five experimental runs with progressively increasing model capacity.

---

## Key Findings

> At 22k parameters, ternary transformers underperform float32 by 5–28% on
> structured tasks due to capacity limitations. At 144k parameters the gap closes
> to 1–3% with 2x training budget. At 1.08M parameters the gap closes entirely —
> ternary and float32 produce **identical accuracy** on all learnable tasks, with
> ternary achieving marginally better best validation loss, 1.53x faster inference,
> and 1.86x memory compression.

---

## Method

### Architecture

Both models share identical transformer architecture. The only structural difference
is that the float32 model uses `nn.Linear` for feed-forward layers, while the ternary
model uses a custom `TernaryLinear` layer that projects weights to {-1, 0, +1} during
the forward pass. Token embeddings, layer normalisations, attention projections, and
the output head remain float32 in both models.

Ternary quantization uses a threshold rule with τ = 0.05:

```
q(w) = +1  if  w >  τ
q(w) =  0  if  |w| ≤ τ
q(w) = -1  if  w < -τ
```

Gradients pass through the quantization step via the straight-through estimator (STE):

```
forward:   uses q(w)
backward:  gradient flows as if quantization = identity
```

The optimizer maintains full-precision latent weights which are re-quantized at each
forward pass. Ternary and float32 models therefore have identical training memory
footprints. At deployment, only the ternary values (1.58 bits per weight) need to be
stored, giving substantial inference memory compression.

### Tasks

All models are trained on next-token prediction over four synthetic mathematical
sequence tasks. No task labels are given — the model must implicitly distinguish
tasks from sequence statistics alone.

| Task      | Description                                                           |
|-----------|-----------------------------------------------------------------------|
| Fibonacci | Next-token prediction over digit-encoded Fibonacci sequences          |
| FizzBuzz  | Next-token prediction over FizzBuzz output sequences                  |
| Parity    | Given a complete bit string, predict the single parity bit at the end |
| Primes    | Next-token prediction over consecutive prime digit sequences          |

All tasks are serialised over a vocabulary of 15 tokens (digits 0–9, FIZZ, BUZZ,
FIZZBUZZ, SEP, PAD). Sequences are 128 tokens long.

The parity task was reformulated partway through the experiment series. The original
formulation interleaved bit strings and parity bits in a continuous stream, requiring
the model to predict parity at positions where the full bit string had not yet been
seen — a fundamentally ill-posed supervision signal. The reformulated task presents
each sample as `[bits] SEP [parity_bit] PAD...`, with the parity bit as the single
supervised prediction target and all PAD positions excluded from the loss via
`ignore_index=TOK_PAD`.

### Training

Both models are trained with AdamW. The float32 model uses LR = 3×10⁻⁴ throughout.
Cosine annealing with warm restarts (T₀=50, T_mult=2) is used from run 4 onward.
All experiments from run 2 onward use Apple M4 with MPS acceleration.

<img width="1427" height="476" alt="loss_curves" src="https://github.com/user-attachments/assets/b1843c09-2a0f-407e-a21f-8afb6329e81a" />

---

## Experimental Runs

### Run 1 — Baseline

**Purpose:** Establish baseline behaviour of ternary training at small scale.

| Parameter     | Value          |
|---------------|----------------|
| Parameters    | 22,208         |
| Embed / Heads / Layers / FF | 32 / 2 / 2 / 64 |
| Dataset       | 50,000 samples |
| Epochs        | 300 (both)     |
| LR (both)     | 3×10⁻⁴         |
| Device        | CPU            |
| Training time | ~6.3 hours     |

**Key observation:** The ternary model exhibited a cascade delay of approximately
9 epochs during which almost no weights crossed the quantization threshold (zero
fraction above 0.92). This was followed by a rapid cascade phase where a large
fraction of weights simultaneously committed to ternary values, driven by AdamW's
momentum accumulation pushing latent weights past the threshold τ.

| Metric             | FP32   | Ternary | Delta   |
|--------------------|--------|---------|---------|
| Final val loss     | 0.2937 | 0.5706  | +0.2769 |
| Fibonacci accuracy | 92.7%  | 64.9%   | -27.8%  |
| FizzBuzz accuracy  | 98.3%  | 92.7%   | -5.6%   |
| Parity accuracy    | 49.9%  | 49.8%   | -0.1%   |
| Primes accuracy    | 98.7%  | 93.0%   | -5.6%   |
| Final zero frac    | —      | 50.1%   | —       |

---

### Run 2 — Hyperparameter Sensitivity

**Purpose:** Test whether learning rate and dataset size affect the ternary ceiling.

| Parameter     | Value          |
|---------------|----------------|
| Parameters    | 22,208         |
| Dataset       | 10,000 samples |
| Epochs        | 300 (both)     |
| LR FP32       | 3×10⁻⁴         |
| LR Ternary    | 1×10⁻³         |
| Device        | MPS            |
| Training time | ~40 minutes    |

**Key observation:** Higher ternary LR triggered the cascade earlier with lower peak
churn. Final ternary accuracy was nearly identical to run 1 on every task, establishing
that the ternary performance ceiling is set by model capacity, not by dataset size or
learning rate within the ranges tested.

| Metric             | FP32   | Ternary | Delta   |
|--------------------|--------|---------|---------|
| Final val loss     | 0.3108 | 0.5663  | +0.2556 |
| Fibonacci accuracy | 88.2%  | 64.5%   | -23.7%  |
| FizzBuzz accuracy  | 98.1%  | 92.7%   | -5.4%   |
| Parity accuracy    | 49.4%  | 50.3%   | +0.8%   |
| Primes accuracy    | 98.5%  | 93.2%   | -5.3%   |
| Final zero frac    | —      | 50.4%   | —       |

---

### Run 3 — Capacity Scaling

**Purpose:** Test whether increasing model capacity closes the ternary-FP32 gap.

| Parameter     | Value          |
|---------------|----------------|
| Parameters    | 144,128        |
| Embed / Heads / Layers / FF | 64 / 4 / 4 / 128 |
| Dataset       | 10,000 samples |
| Epochs        | 300 (both)     |
| LR Ternary    | 5×10⁻⁴         |
| Device        | MPS            |
| Training time | ~98 minutes    |

**Key observation:** The 6.5x parameter increase dramatically closed the accuracy gap
on all learnable tasks. Fibonacci went from -27.8% in run 1 to -3.5%, confirming the
gap was a capacity limitation, not a fundamental constraint of ternary representations.

| Metric             | FP32   | Ternary | Delta  |
|--------------------|--------|---------|--------|
| Final val loss     | 0.1984 | 0.2478  | +0.049 |
| Fibonacci accuracy | 99.1%  | 95.6%   | -3.5%  |
| FizzBuzz accuracy  | 98.6%  | 98.2%   | -0.4%  |
| Parity accuracy    | 51.0%  | 49.5%   | -1.5%  |
| Primes accuracy    | 98.8%  | 98.5%   | -0.3%  |
| Final zero frac    | —      | 41.5%   | —      |

---

### Run 4 — Asymmetric Training Budget

**Purpose:** Test the hypothesis that ternary models need approximately twice the
training budget to reach FP32-equivalent performance. Also introduces the cosine
annealing scheduler and the first inference speed and memory footprint measurements.

| Parameter     | Value                                          |
|---------------|------------------------------------------------|
| Parameters    | 144,128                                        |
| Embed / Heads / Layers / FF | 64 / 4 / 4 / 128             |
| Epochs        | FP32: 200, Ternary: 400                        |
| LR Ternary    | 1×10⁻³                                         |
| Scheduler     | CosineAnnealingWarmRestarts (T₀=50, T_mult=2)  |
| Device        | MPS                                            |
| Training time | ~110 minutes                                   |

**Key observation:** The asymmetric training budget hypothesis was confirmed. FP32 at
200 epochs and ternary at 400 epochs converged to near-identical accuracy on all
learnable tasks. Ternary consistently required 1.5–2x more epochs to cross each
accuracy threshold. Inference on MPS showed ternary running 1.57x faster despite MPS
not being optimised for ternary arithmetic.

| Metric                  | FP32   | Ternary | Delta   |
|-------------------------|--------|---------|---------|
| Final val loss          | 0.0516 | 0.0749  | +0.023  |
| Best val loss           | 0.0516 | 0.0716  | +0.020  |
| Fibonacci accuracy      | 99.2%  | 98.0%   | -1.1%   |
| FizzBuzz accuracy       | 98.7%  | 98.4%   | -0.2%   |
| Parity accuracy         | 44.8%  | 44.4%   | -0.4%   |
| Primes accuracy         | 98.9%  | 98.6%   | -0.3%   |
| Final zero frac         | —      | 38.9%   | —       |
| Inference speed ratio   | 1.000x | 0.636x  | —       |
| Inference memory (KB)   | 563.0  | 319.6   | -1.76x  |

---

### Run 5 — Parameter Scaling (Final)

**Purpose:** Test hypothesis H1 at larger scale — whether the ternary-FP32 accuracy
gap converges to zero at sufficient parameter count. Also a final test of whether
increased depth resolves the parity task failure.

| Parameter     | Value                                          |
|---------------|------------------------------------------------|
| Parameters    | 1,080,320                                      |
| Embed / Heads / Layers / FF | 128 / 8 / 8 / 256            |
| Epochs        | FP32: 200, Ternary: 400                        |
| LR Ternary    | 1×10⁻³                                         |
| Scheduler     | CosineAnnealingWarmRestarts (T₀=50, T_mult=2)  |
| Device        | MPS                                            |
| Training time | ~442 minutes                                   |

**Key observation:** This is the definitive result of the experiment series. The
accuracy gap closed completely on all three learnable tasks — fibonacci, fizzbuzz,
and primes all show 0.0% delta to three decimal places. Ternary's best validation
loss (0.0494) was marginally lower than FP32's (0.0495), suggesting the discrete
weight constraint acts as a mild regulariser at this scale rather than a capacity
limiter. The zero fraction settled at 34.1% — lower than all previous runs — with
a nearly symmetric distribution (-1: 33.1%, 0: 34.1%, +1: 32.8%), indicating the
optimizer made productive use of the full ternary capacity.

Notably, the mean training epoch time for ternary (43.5s) was slightly faster than
FP32 (45.2s), giving a per-epoch ratio of 0.96x. This is the first run where ternary
trained faster per epoch than FP32, likely because the lower and more stable gradients
(mean norm 0.2551 vs 0.4433) reduce optimizer state update costs at this scale.

The parity task remained unsolved at 47.5% / 45.7% — both at chance level. This is
now definitively confirmed as an architectural limitation of causal decoder
transformers on this task, not a ternary limitation and not a capacity limitation.
Eight layers and eight heads do not provide sufficient compositional depth to route
the accumulated bit count through the residual stream to the single prediction
position.

| Metric                  | FP32   | Ternary | Delta    |
|-------------------------|--------|---------|----------|
| Final val loss          | 0.0496 | 0.0500  | +0.0004  |
| Best val loss           | 0.0495 | 0.0494  | -0.0001  |
| Fibonacci accuracy      | 99.1%  | 99.1%   | **0.0%** |
| FizzBuzz accuracy       | 98.7%  | 98.7%   | **0.0%** |
| Parity accuracy         | 47.5%  | 45.7%   | -1.8%    |
| Primes accuracy         | 98.8%  | 98.8%   | **0.0%** |
| Final zero frac         | —      | 34.1%   | —        |
| Mean grad norm          | 0.4433 | 0.2551  | -0.188   |
| Training time ratio     | 1.000x | 0.963x  | —        |
| Inference speed ratio   | 1.000x | 0.653x  | —        |
| Inference memory (KB)   | 4220.0 | 2273.1  | -1.86x   |

---

## Cross-Run Analysis

### Accuracy Gap vs Model Size

The ternary-FP32 accuracy gap is primarily determined by model capacity.

| Run | Params    | Fibonacci Delta | FizzBuzz Delta | Primes Delta |
|-----|-----------|-----------------|----------------|--------------|
| 1   | 22,208    | -27.8%          | -5.6%          | -5.6%        |
| 2   | 22,208    | -23.7%          | -5.4%          | -5.3%        |
| 3   | 144,128   | -3.5%           | -0.4%          | -0.3%        |
| 4   | 144,128   | -1.1%           | -0.2%          | -0.3%        |
| 5   | 1,080,320 | **0.0%**        | **0.0%**       | **0.0%**     |

<img width="1428" height="946" alt="per_task_accuracy" src="https://github.com/user-attachments/assets/289e7389-392d-4198-b47a-8dbf1c278844" />

The relationship is nonlinear. Doubling parameters from 22k produced minimal
improvement (runs 1 vs 2). A 6.5x increase to 144k produced near-complete gap
closure. A further 7.5x increase to 1.08M closed it entirely. This suggests a
threshold effect where ternary representations require minimum parameter density
to reliably encode the decision boundaries of each task.

### Inference Performance Scaling

| Run | Params    | Inference speed ratio | Memory compression |
|-----|-----------|-----------------------|--------------------|
| 4   | 144,128   | 0.636x (1.57x faster) | 1.76x              |
| 5   | 1,080,320 | 0.653x (1.53x faster) | 1.86x              |

Inference advantage is consistent across model sizes and attributable to reduced
memory bandwidth pressure during weight loading. This is a conservative lower bound —
on purpose-built ternary hardware (ESP32-P4 PIE SIMD) the advantage is 20–25x.

### Ternary Weight Distribution Dynamics

<img width="1187" height="468" alt="weight_churn" src="https://github.com/user-attachments/assets/ef3d7c0d-7d55-47c0-bfb4-b0b709b178a2" />

Across all runs the zero fraction followed a consistent lifecycle:

<img width="1548" height="468" alt="ternary_weight_distribution" src="https://github.com/user-attachments/assets/45e15fc1-f9dd-4dc4-a631-77f7af7f9788" />

1. **Initialisation** — nearly all weights at zero (std=0.02 initialisation places
   most latent weights below τ=0.05)
2. **Dead zone** — optimizer builds momentum without visible ternary transitions
3. **Cascade** — rapid commitment of weights to {-1, +1} once latent magnitudes
   cross τ, accompanied by a peak in weight churn
4. **Stabilisation** — churn decays as weights settle
5. **Fine-tuning** — slow continued improvement at low churn

The final zero fraction decreased monotonically with model size and training budget,
from 50.1% in run 1 to 34.1% in run 5. The weight distribution became increasingly
symmetric across runs, approaching a near-uniform split at 1.08M parameters.

### Training Time

Ternary and float32 models train at effectively identical speed per epoch across all
runs (ratio 0.96–1.01x). At 1.08M parameters ternary was marginally faster per epoch,
likely because the more stable and lower-magnitude gradients (norm 0.2551 vs 0.4433)
reduce optimizer state update costs. The STE overhead is negligible compared to
attention computation and data loading at all scales tested.

<img width="1187" height="468" alt="gradient_norms" src="https://github.com/user-attachments/assets/befc333e-3f06-4c84-9ed4-9fb8ef233505" />

### Parity Task

The parity task was not solved by either model across any of the five runs, with
accuracy consistently near 50% (chance level). This is a shared failure of both
architectures and is definitively not a ternary-specific limitation. The cause is
the causal attention mechanism combined with the task formulation: the model must
integrate a variable-length sequence of bits into a single scalar parity decision at
a specific output position. Even at 1.08M parameters with 8 layers and 8 heads, the
residual stream cannot reliably carry the accumulated bit count from the input
positions to the single prediction position. This is an architectural limitation of
causal decoder transformers on global integration tasks, not a property of the
ternary weight representation.

---

## Hypotheses and Findings

**H1: Ternary models match FP32 accuracy at sufficient model capacity**

**Confirmed.** At 1.08M parameters the accuracy delta on all three learnable tasks
is 0.0% to three decimal places. At 22k parameters the gap was 5–28%. The capacity
threshold lies between 144k and 1.08M parameters for this task set. Ternary's best
validation loss at run 5 (0.0494) was marginally lower than FP32's (0.0495),
suggesting the discrete weight constraint provides mild regularisation rather than
capacity reduction at sufficient scale.

**H2: Ternary models need approximately 2x the training budget to match FP32**

**Confirmed.** Consistent across runs 4 and 5. Ternary requires 1.5–2x more epochs
to cross each accuracy threshold. At 1.08M parameters with 400 ternary epochs vs 200
FP32 epochs, all learnable tasks reached identical final accuracy.

**H3: Training time is not increased by ternary quantization**

**Confirmed and strengthened.** Training time ratio is 0.96–1.01x across all runs.
At 1.08M parameters ternary trained faster per epoch than FP32. The STE overhead is
negligible at all scales tested.

**H4: Parity can be solved with increased model capacity**

**Rejected.** Parity was not solved at 1.08M parameters with 8 layers and 8 heads.
The failure is architectural — causal decoder transformers cannot reliably route
accumulated sequence statistics to a single prediction position regardless of
precision or capacity within the ranges tested.

---

## Inference Performance Summary

Measured on Apple M4 with MPS, 20-pass average over the validation set.

| Metric                    | Run 4 (144k) |           | Run 5 (1.08M) |           |
|---------------------------|--------------|-----------|---------------|-----------|
|                           | FP32         | Ternary   | FP32          | Ternary   |
| Mean ms per batch         | 0.849        | 0.540     | 0.749         | 0.489     |
| Mean µs per token         | 1.062        | 0.675     | 0.936         | 0.612     |
| Inference footprint (KB)  | 563.0        | 319.6     | 4220.0        | 2273.1    |
| Compression ratio         | 1.00x        | 1.76x     | 1.00x         | 1.86x     |
| Speed ratio               | 1.000x       | 0.636x    | 1.000x        | 0.653x    |

The inference advantage on MPS is attributable to reduced memory bandwidth pressure
when loading weights during the forward pass. MPS hardware is not optimised for
ternary arithmetic, so this is a conservative lower bound. On purpose-built hardware
with the ESP32-P4's PIE SIMD extensions, the companion project `mcu-ternary-matmul`
measured 20–25x speedup over plain C INT8 for the same weight format.

---

## Repository Structure

```
ternary-transformer-lab/
  notebooks/
    00_config_and_environment.ipynb   Shared configuration and env validation
    01_dataset.ipynb                  Synthetic dataset generation
    02_models.ipynb                   Model architecture definitions
    03_training.ipynb                 Training loop with live display and run versioning
    04_evaluation.ipynb               Loss curves, per-task accuracy, weight analysis,
                                      inference benchmarks
  data/                               Generated dataset (gitignored)
  checkpoints/                        Model state dicts per run (gitignored)
  metrics/                            JSON, CSV, and plots per run (gitignored)
  environment.yml
  README.md
  .gitignore
```

Each run is saved to a timestamped folder under `metrics/` containing:
- `run_config.json` — full hyperparameter record
- `training_metrics.json` — all metrics across all epochs
- `metrics_per_epoch.csv` — epoch-level training metrics
- `metrics_per_eval.csv` — evaluation-interval metrics
- `fp32_model.pt` / `ternary_model.pt` — model checkpoints

---

## Setup

```bash
conda env create -f environment.yml
conda activate ternary-transformer
python -m ipykernel install --user --name ternary-transformer \
    --display-name "ternary-transformer"
```

Run notebooks in order 00 → 01 → 02 → 03 → 04. Each notebook is self-contained
and can be re-run independently after the dataset is generated.

### Notebook Execution Order

| Notebook                        | Purpose                                             |
|---------------------------------|-----------------------------------------------------|
| 00_config_and_environment.ipynb | Shared configuration, environment validation        |
| 01_dataset.ipynb                | Synthetic dataset generation and inspection         |
| 02_models.ipynb                 | Model architecture definitions and parameter counts |
| 03_training.ipynb               | Full training loop, metric collection, checkpointing|
| 04_evaluation.ipynb             | Loss curves, per-task accuracy, weight analysis     |

---

## Dependencies

- Python 3.11
- PyTorch (MPS supported on Apple Silicon)
- NumPy
- Matplotlib
- SymPy (prime number generation)
- tqdm
- Jupyter

---

## Related Work

This project is part of a broader investigation into ternary neural network
deployment on microcontroller hardware. The companion project `mcu-ternary-matmul`
benchmarks four approaches to ternary matrix-vector multiplication on the ESP32-P4
microcontroller using PIE SIMD assembly, validating that the memory and compute
savings of ternary weights are practically realisable on embedded hardware with
20–25x speedup over plain C INT8.
