# DeltaTSF

**Learning Temporal Dynamics with Derivative Representations for Time Series Forecasting**

## Abstract

Time series forecasting aims to predict the future evolution of complex systems by learning their underlying dynamics from past observations. Despite recent advances in deep learning, Transformers in particular still struggle to capture the temporal evolution of system dynamics. Existing approaches have been largely driven by architectural innovation, with comparatively little attention paid to training strategies that explicitly exploit the temporal structure governing system dynamics. In this work, we introduce DeltaTSF, a dynamics-aware forecasting framework based on discrete derivative representations of time series. Specifically, DeltaTSF jointly learns the series and their discrete temporal derivatives across both input and forecast horizons within a unified, model-agnostic framework. By coupling these representations over time, the model is encouraged to maintain temporal consistency, a property we show to induce a form of discrete Taylor consistency during training. This formulation enables the model to better reflect the underlying system dynamics, resulting in more stable and physically coherent forecasts. Notably, DeltaTSF does not require explicit derivative computation and introduces no additional computational overhead at inference time. Extensive experiments across diverse benchmarks and backbone architectures demonstrate that DeltaTSF consistently improves forecasting performance and yields more dynamically coherent predictions than strong SOTA methods. These results highlight the effectiveness of enforcing dynamic coherence through derivative representations as a model-agnostic paradigm that operates alongside and complements architectural design. Source code will be made publicly available upon acceptance.

## Overview

![DeltaTSF overview](output/deltatsf_overview.png)

**Figure 1.** Overview of our DeltaTSF, a model-agnostic training paradigm for existing forecasting backbones without modifying their architectures. DeltaTSF jointly models series and their temporal derivatives within a shared architecture, coupling their predictions across time to enforce temporally structural alignment.

## What Is In This Repository?

This repository uses **PatchTST** as the main codebase for DeltaTSF. The goal is
to make the main implementation easy to run and easy to inspect: the PatchTST
backbone is kept unchanged, while DeltaTSF is added through the dataloader,
experiment loop, and loss construction.

If you are new to this project, start from the PatchTST implementation in this
repository. The additional integrations for iTransformer, TimeMixer, and
TimesNet are provided separately under `integrations/` and are explained near the
end of this README.

## 1. Environment Setup

Create a fresh Python environment first. For example, using conda:

```bash
conda create -n deltatsf python=3.8 -y
conda activate deltatsf
```

Install PyTorch according to your CUDA version from the official PyTorch
website, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The reference server environment used PyTorch 2.0.0 with CUDA 11.7, but the code
only requires a compatible PyTorch installation. If you are running on CPU or a
different CUDA version, install the matching PyTorch build for your machine.

## 2. Prepare Datasets

The experiment scripts expect standard long-term forecasting CSV files such as:

```text
ETTh1.csv
ETTh2.csv
ETTm1.csv
ETTm2.csv
weather.csv
electricity.csv
traffic.csv
```

By default, the scripts currently use the server path:

```bash
root_path_name=/media/D1/temp22/dataset/
```

Before running experiments on your own machine, edit the `root_path_name` value
inside the scripts under:

```text
scripts/PatchTST/rr/
```

For example, if your datasets are stored in `/home/user/datasets/`, change:

```bash
root_path_name=/media/D1/temp22/dataset/
```

to:

```bash
root_path_name=/home/user/datasets/
```

## 3. Run DeltaTSF With PatchTST

The main DeltaTSF + CES implementation is selected by:

```bash
--rand_replace 1
```

This routes training to:

```text
exp/exp_rr.py
```

### Run The Best Paper Settings

To run the best-setting PatchTST DeltaTSF scripts:

```bash
bash run_best.sh
```

`run_best.sh` launches the CES scripts in:

```text
scripts/PatchTST/rr/
```

The current best-setting launcher includes:

```text
ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic
```

### Run A Single Dataset

For example, to run ETTh1 with the paper setting `w_diff1=5`:

```bash
bash scripts/PatchTST/rr/etth1.sh 5
```

Each dataset script runs four forecasting horizons:

```text
96, 192, 336, 720
```

Logs are written under:

```text
logs/<dataset>/<w_diff1>/
```

### Run A Grid Search

To run the grid-search launcher:

```bash
bash scripts/PatchTST/run_all.sh
```

The searched weights are defined inside `scripts/PatchTST/run_all.sh`:

```bash
w_diff_params="1e-3 0.01 0.1 1.0 5.0"
```

## 4. GPU Scheduling

Both `run_best.sh` and `scripts/PatchTST/run_all.sh` use the shared launcher:

```text
scripts/PatchTST/launcher.sh
```

By default, it schedules jobs over GPU IDs `0` to `7`. You can override this
from the command line:

```bash
START_GPU=0 END_GPU=3 MAX_PARALLEL=4 bash run_best.sh
```

For a single GPU machine, use:

```bash
START_GPU=0 END_GPU=0 MAX_PARALLEL=1 bash run_best.sh
```

## 5. Core DeltaTSF Files

The most important files are:

```text
utils/deltatsf.py
```

Reusable DeltaTSF utilities: first-order temporal difference construction,
DeltaTSF loss, CES sampling helpers, and raw-channel selection.

```text
data_provider/data_loader.py
```

Loads raw time-series values and appends first-order temporal difference
channels. This is why the experiment scripts use doubled `enc_in` values, e.g.
ETTh1 changes from `7` raw channels to `14` raw+delta channels.

```text
exp/exp_rr.py
```

The main paper implementation. It uses Compute-Equivalent Sampling (CES): for
each raw/delta channel pair, one member is sampled during training, so the
backbone still receives the original number of channels.

```text
exp/exp_main.py
```

A full raw+delta training path kept for ablation. It is useful for comparing CES
against a direct concatenation strategy.

```text
run_longExp.py
```

Adds DeltaTSF-specific arguments such as:

```bash
--w_diff1
--rand_replace
```

## 6. Other Backbone Integrations

DeltaTSF is model-agnostic. Besides the full PatchTST implementation in this
repository, we also provide key replacement files for:

```text
integrations/itransformer/
integrations/timemixer/
integrations/timesnet/
```

These are **not** complete forks of the original repositories. To use them,
start from each backbone's official codebase and original environment, then copy
only the files from the corresponding integration folder into the matching paths
of that upstream repository.

In other words:

- PatchTST: use this repository directly.
- iTransformer / TimeMixer / TimesNet: use their original repositories and
  environments, then replace the provided core files.

Each integration folder contains its own README with the exact files to copy.

## Results

The rendered paper result tables are stored under:

```text
output/
```

- `output/table9_long_term_results.png`
- `output/table10_baseline_comparison.png`

### Table 9. Long-term multivariate forecasting results

![Table 9: long-term multivariate forecasting results](output/table9_long_term_results.png)

### Table 10. Long-term multivariate forecasting results on six datasets

![Table 10: baseline comparison results](output/table10_baseline_comparison.png)

## Notes

The repository intentionally excludes checkpoints, logs, generated results, and
large experiment artifacts. The core PatchTST architecture files are kept close
to the original backbone so the DeltaTSF modifications are easy to inspect.
