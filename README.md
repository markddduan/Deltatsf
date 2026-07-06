# DeltaTSF

**Learning Temporal Dynamics with Derivative Representations for Time Series Forecasting**

## Abstract

Time series forecasting aims to predict the future evolution of complex systems by learning their underlying dynamics from past observations. Despite recent advances in deep learning, Transformers in particular still struggle to capture the temporal evolution of system dynamics. Existing approaches have been largely driven by architectural innovation, with comparatively little attention paid to training strategies that explicitly exploit the temporal structure governing system dynamics. In this work, we introduce DeltaTSF, a dynamics-aware forecasting framework based on discrete derivative representations of time series. Specifically, DeltaTSF jointly learns the series and their discrete temporal derivatives across both input and forecast horizons within a unified, model-agnostic framework. By coupling these representations over time, the model is encouraged to maintain temporal consistency, a property we show to induce a form of discrete Taylor consistency during training. This formulation enables the model to better reflect the underlying system dynamics, resulting in more stable and physically coherent forecasts. Notably, DeltaTSF does not require explicit derivative computation and introduces no additional computational overhead at inference time. Extensive experiments across diverse benchmarks and backbone architectures demonstrate that DeltaTSF consistently improves forecasting performance and yields more dynamically coherent predictions than strong SOTA methods. These results highlight the effectiveness of enforcing dynamic coherence through derivative representations as a model-agnostic paradigm that operates alongside and complements architectural design. Source code will be made publicly available upon acceptance.

## Overview

![DeltaTSF overview](output/deltatsf_overview.png)

**Figure 1.** Overview of our DeltaTSF, a model-agnostic training paradigm for existing forecasting backbones without modifying their architectures. DeltaTSF jointly models series and their temporal derivatives within a shared architecture, coupling their predictions across time to enforce temporally structural alignment.

## Code Structure

This repository contains the PatchTST-based implementation of **DeltaTSF**, a
training framework for time-series forecasting that jointly models raw series
values and first-order temporal differences.

The PatchTST backbone is kept unchanged. The DeltaTSF additions are concentrated
in a small number of files:

- `utils/deltatsf.py`: DeltaTSF feature construction, losses, and CES helpers
- `data_provider/data_loader.py`: raw + first-order difference channels
- `exp/exp_rr.py`: main DeltaTSF + Compute-Equivalent Sampling path
- `exp/exp_main.py`: full raw + difference training path for the no-CES ablation
- `run_longExp.py`: DeltaTSF arguments such as `--w_diff1` and `--rand_replace`

## Main Entry Points

`exp/exp_rr.py` is the main DeltaTSF + CES experiment path. It is selected by
passing `--rand_replace 1` to `run_longExp.py`.

`exp/exp_main.py` keeps the full raw + difference channel training path and is
useful for the no-CES ablation.

## Reproduction

Prepare the datasets under the path used by the scripts, then run:

```bash
bash run_best.sh
```

The CES scripts in `scripts/PatchTST/rr/` correspond to the paper's DeltaTSF
structure and are used by `run_best.sh` and `scripts/PatchTST/run_all.sh`.

```text
scripts/PatchTST/rr/
```

For example:

```bash
bash scripts/PatchTST/rr/etth1.sh 5
```

The full raw + difference channel scripts without CES are kept for ablation in:

```text
scripts/PatchTST/main/
```

Both `run_best.sh` and `scripts/PatchTST/run_all.sh` use the shared GPU
launcher in `scripts/PatchTST/launcher.sh`. You can control scheduling with:

```bash
START_GPU=0 END_GPU=7 MAX_PARALLEL=8 bash run_best.sh
```

## Backbone Integrations

Additional DeltaTSF + CES integration files for iTransformer, TimeMixer, and
TimesNet are provided in:

```text
integrations/
```

These folders contain only the key files that should be copied into the
corresponding upstream backbone repositories. They are not full forks.

## Results

The original LaTeX result tables and Markdown renderings are stored under:

```text
output/
```

- `output/table9_mixup_vs_delta.tex`
- `output/table9_mixup_vs_delta.md`
- `output/table10_long_term_results.tex`
- `output/table10_long_term_results.md`

<details open>
<summary>Table 9: MixUp vs DeltaTSF</summary>

### Table 9. Comparative analysis of PatchTST with +MixUp and +Delta

Results are reported in terms of MSE / MAE (lower is better).

| Dataset | Horizon | PatchTST MSE | PatchTST MAE | +MixUp MSE | +MixUp MAE | +Delta MSE | +Delta MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 96 | 0.377 | 0.408 | 0.375 | 0.407 | **0.362** | **0.395** |
| ETTh1 | 192 | 0.413 | 0.431 | 0.414 | 0.432 | **0.396** | **0.416** |
| ETTh1 | 336 | 0.436 | 0.446 | 0.426 | 0.441 | **0.405** | **0.424** |
| ETTh1 | 720 | 0.455 | 0.475 | 0.448 | 0.469 | **0.420** | **0.448** |
| ETTh1 | AVG | 0.420 | 0.440 | 0.416 | 0.437 | **0.396** | **0.421** |
| ETTh2 | 96 | 0.276 | 0.339 | 0.274 | 0.337 | **0.269** | **0.333** |
| ETTh2 | 192 | 0.342 | 0.385 | 0.339 | 0.379 | **0.330** | **0.375** |
| ETTh2 | 336 | 0.364 | 0.405 | 0.330 | 0.384 | **0.316** | **0.377** |
| ETTh2 | 720 | 0.395 | 0.434 | 0.383 | 0.426 | **0.368** | **0.417** |
| ETTh2 | AVG | 0.344 | 0.391 | 0.332 | 0.382 | **0.321** | **0.376** |
| ETTm1 | 96 | 0.298 | 0.352 | 0.295 | 0.347 | **0.299** | **0.352** |
| ETTm1 | 192 | 0.335 | 0.373 | 0.333 | 0.370 | **0.327** | **0.370** |
| ETTm1 | 336 | 0.366 | 0.394 | 0.364 | 0.390 | **0.354** | **0.390** |
| ETTm1 | 720 | 0.420 | 0.421 | 0.414 | 0.418 | **0.401** | **0.418** |
| ETTm1 | AVG | 0.355 | 0.385 | 0.352 | 0.381 | **0.345** | **0.383** |

</details>

<details>
<summary>Table 10: Long-term multivariate forecasting results</summary>

### Table 10. Long-term multivariate forecasting results

The table reports MSE and MAE for different forecasting lengths T in {96, 192, 336, 720}. The input sequence length is fixed to 720 for all methods.

| Dataset | PatchTST ori MSE | PatchTST ori MAE | PatchTST +Delta MSE | PatchTST +Delta MAE | iTransformer ori MSE | iTransformer ori MAE | iTransformer +Delta MSE | iTransformer +Delta MAE | TimeMixer ori MSE | TimeMixer ori MAE | TimeMixer +Delta MSE | TimeMixer +Delta MAE | TimesNet ori MSE | TimesNet ori MAE | TimesNet +Delta MSE | TimesNet +Delta MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1-96 | 0.377 | 0.408 | **0.362** | **0.395** | 0.389 | 0.421 | **0.373** | **0.406** | 0.410 | 0.441 | **0.391** | **0.420** | 0.437 | 0.454 | **0.401** | **0.433** |
| ETTh1-192 | 0.413 | 0.431 | **0.396** | **0.416** | 0.424 | 0.446 | **0.409** | **0.431** | 0.448 | 0.465 | **0.415** | **0.431** | 0.454 | 0.469 | **0.425** | **0.453** |
| ETTh1-336 | 0.436 | 0.446 | **0.405** | **0.424** | 0.456 | 0.469 | **0.423** | **0.440** | 0.482 | 0.490 | **0.426** | **0.440** | 0.494 | 0.494 | **0.466** | **0.487** |
| ETTh1-720 | 0.455 | 0.475 | **0.420** | **0.448** | 0.545 | 0.532 | **0.486** | **0.505** | 0.475 | 0.500 | **0.450** | **0.468** | 0.632 | 0.578 | **0.485** | **0.502** |
| ETTh1-AVG | 0.420 | 0.440 | **0.396** | **0.421** | 0.454 | 0.467 | **0.423** | **0.446** | 0.454 | 0.474 | **0.421** | **0.440** | 0.504 | 0.499 | **0.444** | **0.469** |
| ETTh2-96 | 0.276 | 0.339 | **0.269** | **0.333** | 0.305 | 0.361 | **0.285** | **0.345** | 0.315 | 0.380 | **0.297** | **0.356** | 0.349 | 0.403 | **0.315** | **0.358** |
| ETTh2-192 | 0.342 | 0.385 | **0.330** | **0.375** | 0.405 | 0.421 | **0.389** | **0.411** | 0.383 | 0.415 | **0.360** | **0.395** | 0.500 | 0.488 | **0.388** | **0.400** |
| ETTh2-336 | 0.364 | 0.405 | **0.316** | **0.377** | 0.411 | 0.436 | **0.407** | **0.432** | 0.385 | 0.438 | **0.340** | **0.394** | 0.445 | 0.465 | **0.442** | **0.446** |
| ETTh2-720 | 0.395 | 0.434 | **0.368** | **0.417** | 0.448 | 0.470 | **0.410** | **0.448** | 0.432 | 0.471 | **0.397** | **0.436** | 0.438 | 0.465 | **0.439** | **0.452** |
| ETTh2-AVG | 0.344 | 0.391 | **0.321** | **0.376** | 0.392 | 0.422 | **0.373** | **0.409** | 0.379 | 0.426 | **0.349** | **0.395** | 0.433 | 0.455 | **0.396** | **0.414** |
| ETTm1-96 | 0.298 | 0.352 | 0.299 | 0.352 | 0.315 | 0.369 | **0.310** | **0.363** | 0.332 | 0.384 | **0.306** | **0.358** | 0.359 | 0.391 | **0.313** | **0.362** |
| ETTm1-192 | 0.335 | 0.373 | **0.327** | **0.370** | 0.349 | 0.388 | **0.349** | **0.389** | 0.355 | 0.398 | **0.340** | **0.374** | 0.368 | 0.398 | **0.346** | **0.383** |
| ETTm1-336 | 0.366 | 0.394 | **0.354** | **0.390** | 0.381 | 0.409 | **0.378** | **0.406** | 0.386 | 0.416 | **0.366** | **0.389** | 0.429 | 0.438 | **0.378** | **0.400** |
| ETTm1-720 | 0.420 | 0.421 | **0.401** | **0.418** | 0.437 | 0.439 | **0.431** | **0.434** | 0.452 | 0.457 | **0.426** | **0.425** | 0.477 | 0.474 | **0.449** | **0.440** |
| ETTm1-AVG | 0.355 | 0.385 | **0.345** | **0.383** | 0.371 | 0.401 | **0.367** | **0.398** | 0.381 | 0.414 | **0.360** | **0.387** | 0.408 | 0.425 | **0.372** | **0.396** |
| ETTm2-96 | 0.165 | 0.260 | **0.163** | **0.253** | 0.179 | 0.274 | **0.174** | **0.265** | 0.192 | 0.285 | **0.173** | **0.256** | 0.200 | 0.288 | **0.181** | **0.274** |
| ETTm2-192 | 0.219 | 0.298 | **0.221** | **0.297** | 0.239 | 0.314 | **0.240** | **0.308** | 0.253 | 0.329 | **0.237** | **0.297** | 0.274 | 0.337 | **0.250** | **0.316** |
| ETTm2-336 | 0.268 | 0.333 | **0.269** | **0.327** | 0.309 | 0.356 | **0.293** | **0.347** | 0.307 | 0.362 | **0.294** | **0.335** | 0.340 | 0.382 | **0.289** | **0.340** |
| ETTm2-720 | 0.352 | 0.386 | **0.347** | **0.378** | 0.387 | 0.407 | **0.364** | **0.391** | 0.380 | 0.412 | **0.390** | **0.393** | 0.384 | 0.407 | **0.380** | **0.399** |
| ETTm2-AVG | 0.251 | 0.319 | **0.250** | **0.314** | 0.279 | 0.338 | **0.268** | **0.328** | 0.283 | 0.347 | **0.274** | **0.320** | 0.300 | 0.354 | **0.275** | **0.332** |
| Weather-96 | 0.149 | 0.199 | **0.146** | **0.197** | 0.159 | 0.212 | 0.166 | 0.221 | 0.163 | 0.223 | **0.153** | **0.208** | 0.176 | 0.234 | **0.176** | **0.233** |
| Weather-192 | 0.193 | 0.243 | **0.191** | **0.242** | 0.203 | 0.252 | 0.227 | 0.256 | 0.201 | 0.254 | **0.199** | **0.249** | 0.219 | 0.270 | 0.220 | 0.271 |
| Weather-336 | 0.240 | 0.281 | **0.241** | **0.280** | 0.253 | 0.291 | 0.254 | 0.292 | 0.258 | 0.300 | **0.253** | **0.293** | 0.277 | 0.311 | **0.260** | **0.295** |
| Weather-720 | 0.312 | 0.334 | **0.305** | **0.328** | 0.317 | 0.337 | 0.322 | 0.344 | 0.329 | 0.348 | **0.330** | **0.348** | 0.344 | 0.356 | **0.319** | **0.341** |
| Weather-AVG | 0.224 | 0.264 | **0.221** | **0.262** | 0.233 | 0.273 | 0.242 | 0.278 | 0.238 | 0.281 | **0.234** | **0.275** | 0.254 | 0.293 | **0.244** | **0.285** |
| Electricity-96 | 0.141 | 0.240 | **0.130** | **0.225** | 0.135 | 0.233 | **0.131** | **0.228** | 0.139 | 0.239 | **0.131** | **0.226** | 0.202 | 0.308 | **0.180** | **0.283** |
| Electricity-192 | 0.156 | 0.256 | **0.147** | **0.241** | 0.155 | 0.253 | **0.153** | **0.250** | 0.155 | 0.250 | **0.149** | **0.243** | 0.218 | 0.322 | **0.187** | **0.292** |
| Electricity-336 | 0.172 | 0.267 | **0.161** | **0.256** | 0.169 | 0.267 | **0.168** | **0.265** | 0.171 | 0.265 | **0.164** | **0.258** | 0.232 | 0.332 | **0.211** | **0.311** |
| Electricity-720 | 0.208 | 0.299 | **0.195** | **0.288** | 0.204 | 0.301 | **0.189** | **0.284** | 0.208 | 0.300 | **0.200** | **0.293** | 0.299 | 0.375 | **0.244** | **0.335** |
| Electricity-AVG | 0.169 | 0.266 | **0.158** | **0.253** | 0.166 | 0.264 | **0.160** | **0.257** | 0.168 | 0.264 | **0.161** | **0.255** | 0.238 | 0.334 | **0.206** | **0.305** |

</details>

## Notes

The repository intentionally excludes checkpoints, logs, generated results, and
large experiment artifacts. The core PatchTST architecture files are kept close
to the original backbone so the DeltaTSF modifications are easy to inspect.
