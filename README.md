---
type: Repository Guide
title: Ternary Transformer Lab
description: An empirical investigation into ternary weight transformers trained from scratch.
status: stable
tags: [ternary, research]
generated:
  by: human:ciprian-florin_ifrim
  at: 2026-03-17T00:00:00Z
---

# Ternary Transformer Lab

This project is an empirical investigation into ternary weight transformers. We train
every model from scratch. We compare the accuracy, the convergence, the memory
footprint and the inference speed against a float32 equivalent. The tasks are a suite
of synthetic mathematical sequences. All experiments run at small scale, so we can
iterate fast and test one hypothesis at a time. The goal is to find the conditions
where a ternary weight representation is practical for deployment.

---

## Overview

A ternary neural network holds every weight at one of three values: -1, 0, or +1. The
network therefore does no float32 multiply-accumulate operation at inference time. It
uses an addition, a subtraction, or a zero-skip in place of each multiplication. The
theoretical gain in inference efficiency is large, and it is largest on hardware that
memory bandwidth limits, such as a microcontroller. One question stays open. Does the
gain cost too much accuracy, and can we make that cost negligible?

This project trains a standard float32 transformer and the same architecture with
ternary-quantized feed-forward weights. Both models train under controlled conditions.
We measure the accuracy, the convergence speed, the weight distribution dynamics and
the inference performance. There are six experimental runs, and the model capacity
increases at each step.

---

## Key Findings

> At 22k parameters a ternary transformer is 5% to 28% below float32 on the structured
> tasks, because it has too little capacity. At 144k parameters the gap closes to
> 1% to 3% with a 2x training budget. At 550k parameters the gap closes completely:
> ternary and float32 give **identical accuracy** on all learnable tasks. Run 5
> confirms the result again at 1.08M parameters. The practical minimum viable
> configuration is **550k parameters at 100/200 epochs**. There ternary gives 1.53x
> faster inference and 1.86x memory compression.

---

## Method

### Architecture

Both models have the same transformer architecture. One structural difference exists.
The float32 model uses `nn.Linear` for the feed-forward layers. The ternary model uses
a custom `TernaryLinear` layer, which projects the weights to {-1, 0, +1} in the
forward pass. In both models the token embeddings, the layer normalizations, the
attention projections and the output head stay float32.

The ternary quantization uses a threshold rule with tau = 0.05:

```
q(w) = +1  if  w >  tau
q(w) =  0  if  |w| <= tau
q(w) = -1  if  w < -tau
```

Gradients pass through the quantization step with the straight-through estimator (STE):

```
forward:   uses q(w)
backward:  gradient flows as if quantization = identity
```

The optimizer keeps full-precision latent weights. It re-quantizes them at each
forward pass. The ternary model and the float32 model therefore use the same amount of
memory for training. At deployment you store only the ternary values, at 1.58 bits per
weight. This gives a large compression of the inference memory.

### Tasks

We train all models on next-token prediction over four synthetic mathematical sequence
tasks. We give no task label. The model must separate the tasks from the sequence
statistics alone.

| Task      | Description                                                           |
|-----------|-----------------------------------------------------------------------|
| Fibonacci | Next-token prediction over digit-encoded Fibonacci sequences          |
| FizzBuzz  | Next-token prediction over FizzBuzz output sequences                  |
| Parity    | Given a complete bit string, predict the single parity bit at the end |
| Primes    | Next-token prediction over consecutive prime digit sequences          |

Each task serializes over a vocabulary of 15 tokens: the digits 0 to 9, FIZZ, BUZZ,
FIZZBUZZ, SEP and PAD. Each sequence is 128 tokens long.

We changed the parity task in the middle of the experiment series. The first form
interleaved the bit strings and the parity bits in one continuous stream. The model had
to predict the parity at a position where it had not yet seen the full bit string. That
supervision signal was ill-posed. The new form gives each sample as
`[bits] SEP [parity_bit] PAD...`. The parity bit is the one supervised target, and
`ignore_index=TOK_PAD` removes every PAD position from the loss.

### Training

We train both models with AdamW. The float32 model uses LR = 3x10^-4 throughout. We
add the `CosineAnnealingWarmRestarts` scheduler from run 4 onward, with T0=50 and
T_mult=2. All experiments from run 2 onward use Apple Silicon with MPS acceleration.

<img width="1427" height="476" alt="loss_curves" src="https://github.com/user-attachments/assets/b1843c09-2a0f-407e-a21f-8afb6329e81a" />

---

## Experimental Runs

### Run 1: Baseline

**Purpose:** Establish the baseline behavior of ternary training at small scale.

| Parameter     | Value          |
|---------------|----------------|
| Parameters    | 22,208         |
| Embed / Heads / Layers / FF | 32 / 2 / 2 / 64 |
| Dataset       | 50,000 samples |
| Epochs        | 300 (both)     |
| LR (both)     | 3x10^-4        |
| Device        | CPU            |

**Key observation:** The ternary model showed a cascade delay of about 9 epochs. During
that delay almost no weight crossed the quantization threshold, and the zero fraction
stayed above 0.92. A rapid cascade phase came next. A large fraction of the weights
committed to ternary values at the same time. The momentum accumulation of AdamW caused
this, because it pushed the latent weights past the threshold tau.

| Metric             | FP32   | Ternary | Delta   |
|--------------------|--------|---------|---------|
| Final val loss     | 0.2937 | 0.5706  | +0.2769 |
| Fibonacci accuracy | 92.7%  | 64.9%   | -27.8%  |
| FizzBuzz accuracy  | 98.3%  | 92.7%   | -5.6%   |
| Parity accuracy    | 49.9%  | 49.8%   | -0.1%   |
| Primes accuracy    | 98.7%  | 93.0%   | -5.6%   |
| Final zero frac    | n/a    | 50.1%   | n/a     |

---

### Run 2: Hyperparameter Sensitivity

**Purpose:** Test whether the learning rate and the dataset size move the ternary
performance limit.

| Parameter     | Value          |
|---------------|----------------|
| Parameters    | 22,208         |
| Dataset       | 10,000 samples |
| Epochs        | 300 (both)     |
| LR FP32       | 3x10^-4        |
| LR Ternary    | 1x10^-3        |
| Device        | MPS            |

**Key observation:** The higher ternary LR triggered the cascade earlier, and the peak
churn was lower. The final ternary accuracy was almost identical to run 1 on every
task. This result shows that the model capacity sets the ternary performance limit.
The dataset size and the learning rate do not set it, inside the ranges that we tested.

| Metric             | FP32   | Ternary | Delta   |
|--------------------|--------|---------|---------|
| Final val loss     | 0.3108 | 0.5663  | +0.2556 |
| Fibonacci accuracy | 88.2%  | 64.5%   | -23.7%  |
| FizzBuzz accuracy  | 98.1%  | 92.7%   | -5.4%   |
| Parity accuracy    | 49.4%  | 50.3%   | +0.8%   |
| Primes accuracy    | 98.5%  | 93.2%   | -5.3%   |
| Final zero frac    | n/a    | 50.4%   | n/a     |

---

### Run 3: Capacity Scaling

**Purpose:** Test whether more model capacity closes the gap between ternary and FP32.

| Parameter     | Value          |
|---------------|----------------|
| Parameters    | 144,128        |
| Embed / Heads / Layers / FF | 64 / 4 / 4 / 128 |
| Dataset       | 10,000 samples |
| Epochs        | 300 (both)     |
| LR Ternary    | 5x10^-4        |
| Device        | MPS            |

**Key observation:** The 6.5x increase in parameters closed most of the accuracy gap on
all learnable tasks. Fibonacci moved from -27.8% in run 1 to -3.5%. This confirms that
the gap was a capacity limit, and not a fundamental constraint of the ternary
representation.

| Metric             | FP32   | Ternary | Delta  |
|--------------------|--------|---------|--------|
| Final val loss     | 0.1984 | 0.2478  | +0.049 |
| Fibonacci accuracy | 99.1%  | 95.6%   | -3.5%  |
| FizzBuzz accuracy  | 98.6%  | 98.2%   | -0.4%  |
| Parity accuracy    | 51.0%  | 49.5%   | -1.5%  |
| Primes accuracy    | 98.8%  | 98.5%   | -0.3%  |
| Final zero frac    | n/a    | 41.5%   | n/a    |

---

### Run 4: Asymmetric Training Budget

**Purpose:** Test the hypothesis that a ternary model needs about twice the training
budget of FP32. This run also adds the `CosineAnnealingWarmRestarts` scheduler. It
gives the first measurements of inference speed and memory footprint.

| Parameter     | Value                                          |
|---------------|------------------------------------------------|
| Parameters    | 144,128                                        |
| Embed / Heads / Layers / FF | 64 / 4 / 4 / 128             |
| Epochs        | FP32: 200, Ternary: 400                        |
| LR Ternary    | 1x10^-3                                        |
| Scheduler     | CosineAnnealingWarmRestarts (T0=50, T_mult=2)  |
| Device        | MPS                                            |

**Key observation:** This run confirms the hypothesis about the asymmetric training
budget. FP32 at 200 epochs and ternary at 400 epochs reached almost identical accuracy
on all learnable tasks. Ternary always needed 1.5x to 2x more epochs to cross each
accuracy threshold. On MPS the ternary model ran 1.57x faster, although MPS has no
optimization for ternary arithmetic.

| Metric                  | FP32   | Ternary | Delta   |
|-------------------------|--------|---------|---------|
| Final val loss          | 0.0516 | 0.0749  | +0.023  |
| Best val loss           | 0.0516 | 0.0716  | +0.020  |
| Fibonacci accuracy      | 99.2%  | 98.0%   | -1.1%   |
| FizzBuzz accuracy       | 98.7%  | 98.4%   | -0.2%   |
| Parity accuracy         | 44.8%  | 44.4%   | -0.4%   |
| Primes accuracy         | 98.9%  | 98.6%   | -0.3%   |
| Final zero frac         | n/a    | 38.9%   | n/a     |
| Inference speed ratio   | 1.000x | 0.636x  | n/a     |
| Inference memory (KB)   | 563.0  | 319.6   | -1.76x  |

---

### Run 5: Parameter Scaling

**Purpose:** Test hypothesis H1 at a larger scale. Does the accuracy gap between
ternary and FP32 go to zero when the parameter count is large enough? This run also
tests whether more depth solves the parity task failure.

| Parameter     | Value                                          |
|---------------|------------------------------------------------|
| Parameters    | 1,080,320                                      |
| Embed / Heads / Layers / FF | 128 / 8 / 8 / 256            |
| Epochs        | FP32: 200, Ternary: 400                        |
| LR Ternary    | 1x10^-3                                        |
| Scheduler     | CosineAnnealingWarmRestarts (T0=50, T_mult=2)  |
| Device        | MPS                                            |

**Key observation:** The accuracy gap closed completely on all three learnable tasks.
Fibonacci, fizzbuzz and primes all show a 0.0% delta to three decimal places. The best
validation loss of ternary (0.0494) was a little below the FP32 value (0.0495). At this
scale the discrete weight constraint therefore acts as a mild regularizer and not as a
capacity limit. The zero fraction settled at 34.1%. The distribution is almost
symmetric: -1 at 33.1%, 0 at 34.1%, and +1 at 32.8%.

The parity task stayed unsolved at 47.5% / 45.7%. This is an architectural limit of a
causal decoder transformer, and not a limit of ternary weights or of capacity.

| Metric                  | FP32   | Ternary | Delta    |
|-------------------------|--------|---------|----------|
| Final val loss          | 0.0496 | 0.0500  | +0.0004  |
| Best val loss           | 0.0495 | 0.0494  | -0.0001  |
| Fibonacci accuracy      | 99.1%  | 99.1%   | **0.0%** |
| FizzBuzz accuracy       | 98.7%  | 98.7%   | **0.0%** |
| Parity accuracy         | 47.5%  | 45.7%   | -1.8%    |
| Primes accuracy         | 98.8%  | 98.8%   | **0.0%** |
| Final zero frac         | n/a    | 34.1%   | n/a      |
| Mean grad norm          | 0.4433 | 0.2551  | -0.188   |
| Training time ratio     | 1.000x | 0.963x  | n/a      |
| Inference speed ratio   | 1.000x | 0.653x  | n/a      |
| Inference memory (KB)   | 4220.0 | 2273.1  | -1.86x   |

---

### Run 6: Minimum Viable Configuration

**Purpose:** Establish the lowest parameter count and training budget where ternary
matches FP32 accuracy. This run uses a smaller architecture and a shorter training
schedule, to find the most efficient point of operation.

| Parameter     | Value                                          |
|---------------|------------------------------------------------|
| Parameters    | 550,400                                        |
| Embed / Heads / Layers / FF | 128 / 4 / 4 / 256            |
| Epochs        | FP32: 100, Ternary: 200                        |
| LR Ternary    | 1x10^-3                                        |
| Scheduler     | CosineAnnealingWarmRestarts (T0=50, T_mult=2)  |
| Device        | MPS                                            |

**Key observation:** This run gives the same accuracy as run 5 on all three learnable
tasks. It uses about half the parameters and one fifth of the total epoch count. The
best ternary val loss (0.0509) sits a little above the FP32 value (0.0501). The ternary
model was therefore a little under-trained, and not short of capacity. The accuracy
numbers are already identical at 200 epochs, and more training would close the loss
gap. The training time ratio per epoch stays at 1.00x, as in every earlier run.

This run defines the practical minimum viable configuration: 550k parameters, 100 FP32
epochs and 200 ternary epochs.

| Metric                  | FP32   | Ternary | Delta    |
|-------------------------|--------|---------|----------|
| Final val loss          | 0.0501 | 0.0520  | +0.0019  |
| Best val loss           | 0.0501 | 0.0509  | +0.0008  |
| Fibonacci accuracy      | 99.1%  | 99.1%   | **0.0%** |
| FizzBuzz accuracy       | 98.7%  | 98.7%   | **0.0%** |
| Parity accuracy         | 46.0%  | 46.2%   | +0.2%    |
| Primes accuracy         | 98.8%  | 98.8%   | **0.0%** |
| Final zero frac         | n/a    | 37.0%   | n/a      |
| Mean grad norm          | 0.7937 | 0.5444  | -0.249   |
| Training time ratio     | 1.000x | 1.003x  | n/a      |

---

## Cross-Run Analysis

### Accuracy Gap vs Model Size

The model capacity mostly determines the accuracy gap between ternary and FP32.

| Run | Params    | Epochs (FP32/Tern) | Fibonacci Delta | FizzBuzz Delta | Primes Delta |
|-----|-----------|-------------------|-----------------|----------------|--------------|
| 1   | 22,208    | 300 / 300         | -27.8%          | -5.6%          | -5.6%        |
| 2   | 22,208    | 300 / 300         | -23.7%          | -5.4%          | -5.3%        |
| 3   | 144,128   | 300 / 300         | -3.5%           | -0.4%          | -0.3%        |
| 4   | 144,128   | 200 / 400         | -1.1%           | -0.2%          | -0.3%        |
| 5   | 1,080,320 | 200 / 400         | **0.0%**        | **0.0%**       | **0.0%**     |
| 6   | 550,400   | 100 / 200         | **0.0%**        | **0.0%**       | **0.0%**     |

<img width="1428" height="946" alt="per_task_accuracy" src="https://github.com/user-attachments/assets/289e7389-392d-4198-b47a-8dbf1c278844" />

The relation is nonlinear. Twice the parameters from 22k gave a minimal improvement.
Compare run 1 and run 2. A 6.5x increase to 144k closed almost all of the gap. Run 5 at
1.08M confirmed full closure. Run 6 at 550k also reached a 0.0% delta on all learnable
tasks, with a much shorter training budget. The estimate of the capacity threshold is
therefore tighter: it is between 144k and 550k parameters for this task set. The
practical minimum viable configuration for zero-gap ternary inference is 550k
parameters at 100/200 epochs.

### Inference Performance Scaling

| Run | Params    | Inference speed ratio | Memory compression |
|-----|-----------|-----------------------|--------------------|
| 4   | 144,128   | 0.636x (1.57x faster) | 1.76x              |
| 5   | 1,080,320 | 0.653x (1.53x faster) | 1.86x              |

The inference advantage is consistent across the model sizes. The cause is less
pressure on the memory bandwidth when the model reads the weights. This figure is a
conservative lower bound. On hardware that is built for ternary arithmetic, the
ESP32-P4 with PIE SIMD, the advantage is 20x to 25x.

### Ternary Weight Distribution Dynamics

<img width="1187" height="468" alt="weight_churn" src="https://github.com/user-attachments/assets/ef3d7c0d-7d55-47c0-bfb4-b0b709b178a2" />

In all runs the zero fraction followed the same lifecycle:

<img width="1548" height="468" alt="ternary_weight_distribution" src="https://github.com/user-attachments/assets/45e15fc1-f9dd-4dc4-a631-77f7af7f9788" />

1. **Initialization**: almost all the weights sit at zero. The std=0.02 initialization
   puts most latent weights below tau=0.05.
2. **Dead zone**: the optimizer builds momentum, and no ternary transition is visible.
3. **Cascade**: the weights commit fast to {-1, +1} after the latent magnitudes cross
   tau. The weight churn peaks here.
4. **Stabilization**: the churn decays as the weights settle.
5. **Fine-tuning**: slow continued improvement at low churn.

The final zero fraction decreased monotonically with the model size and the training
budget, from 50.1% in run 1 to 34.1% in run 5. Run 6 at 550k settled at 37.0%. This
agrees with the relation between the capacity and the number of active weights. The
weight distribution became more symmetric with each run.

### Training Time

The ternary and float32 models train at almost the same speed per epoch in all runs,
with a ratio of 0.96x to 1.00x. At 1.08M parameters ternary was a little faster per
epoch. The gradients are more stable and smaller in magnitude, so they probably reduce
the cost to update the optimizer state. At all scales that we tested, the STE overhead
is negligible against the cost of attention and of the data load.

<img width="1187" height="468" alt="gradient_norms" src="https://github.com/user-attachments/assets/befc333e-3f06-4c84-9ed4-9fb8ef233505" />

### Parity Task

Neither model solved the parity task in any of the six runs. The accuracy stayed near
50%, which is chance level. Both models fail in the same way, so this is not a limit of
ternary weights. The cause is the causal attention mechanism together with the task
formulation. The model must reduce a bit sequence of variable length to one scalar
parity decision at one output position.

Even at 1.08M parameters, with 8 layers and 8 heads, the residual stream cannot carry
the accumulated bit count reliably. It must carry that count from the input positions
to the one prediction position. This is an architectural limit of a causal decoder
transformer on a global integration task. It is not a property of the ternary weight
representation.

---

## Hypotheses and Findings

**H1: Ternary models match FP32 accuracy at a large enough model capacity**

**Confirmed.** The delta reaches 0.0% on all three learnable tasks. This happens at 550k
parameters (run 6, 100/200 epochs) and at 1.08M parameters (run 5, 200/400 epochs). At
22k parameters the gap was 5% to 28%. The capacity threshold is between 144k and 550k
parameters for this task set. In run 5 the best ternary validation loss (0.0494) was a
little below the FP32 value (0.0495). At a large enough scale the discrete weight
constraint therefore gives mild regularization. Run 6 shows that the zero-gap result
survives a lower parameter count and a shorter training budget.

**H2: Ternary models need about 2x the training budget to match FP32**

**Confirmed.** Runs 4, 5 and 6 agree. Ternary needs 1.5x to 2x more epochs to cross
each accuracy threshold. The relation holds at 100/200 epochs (run 6) and at 200/400
epochs (runs 4 and 5).

**H3: Ternary quantization does not increase the training time**

**Confirmed and strengthened.** The training time ratio is 0.96x to 1.00x in all runs.
At 1.08M parameters ternary trained faster per epoch than FP32. The STE overhead is
negligible at all scales that we tested.

**H4: More model capacity solves parity**

**Rejected.** No run of the six solved parity. The failure is architectural. A causal
decoder transformer cannot route the accumulated sequence statistics reliably to one
prediction position. The precision and the capacity do not change this, inside the
ranges that we tested.

---

## Inference Performance Summary

We measured on Apple Silicon with MPS. Each figure is a 20-pass average over the
validation set.

| Metric                    | Run 4 (144k) |           | Run 5 (1.08M) |           |
|---------------------------|--------------|-----------|---------------|-----------|
|                           | FP32         | Ternary   | FP32          | Ternary   |
| Mean ms per batch         | 0.849        | 0.540     | 0.749         | 0.489     |
| Mean microseconds/token   | 1.062        | 0.675     | 0.936         | 0.612     |
| Inference footprint (KB)  | 563.0        | 319.6     | 4220.0        | 2273.1    |
| Compression ratio         | 1.00x        | 1.76x     | 1.00x         | 1.86x     |
| Speed ratio               | 1.000x       | 0.636x    | 1.000x        | 0.653x    |

The inference advantage on MPS comes from less pressure on the memory bandwidth when
the model reads the weights in the forward pass. MPS hardware has no optimization for
ternary arithmetic, so this figure is a conservative lower bound. The companion project
`mcu-ternary-matmul` measured a 20x to 25x speedup over plain C INT8 for the same
weight format. It ran on hardware that is built for the job, with the PIE SIMD
extensions of the ESP32-P4.

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

Each run writes a timestamped folder under `metrics/`. The folder holds:
- `run_config.json`: the full hyperparameter record
- `training_metrics.json`: all metrics across all epochs
- `metrics_per_epoch.csv`: the training metrics per epoch
- `metrics_per_eval.csv`: the metrics per evaluation interval
- `fp32_model.pt` / `ternary_model.pt`: the model checkpoints

---

## Setup

```bash
conda env create -f environment.yml
conda activate ternary-transformer
python -m ipykernel install --user --name ternary-transformer \
    --display-name "ternary-transformer"
```

Run the notebooks in the order 00 - 01 - 02 - 03 - 04. Each notebook is self-contained.
You can run one notebook again on its own after you generate the dataset.

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

- Python 3.11.
- PyTorch, with MPS support on Apple Silicon.
- NumPy.
- Matplotlib.
- SymPy, for prime number generation.
- tqdm.
- Jupyter.

---

## Related Work

This project is one part of a larger investigation into ternary neural networks on
microcontroller hardware. The companion project `mcu-ternary-matmul` benchmarks four
approaches to ternary matrix-vector multiplication. It runs on the ESP32-P4
microcontroller with PIE SIMD assembly. It confirms that embedded hardware can realize
the memory savings and the compute savings of ternary weights. The speedup is 20x to
25x over plain C INT8.
