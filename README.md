# DeltaTSF

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

## Notes

The repository intentionally excludes checkpoints, logs, generated results, and
large experiment artifacts. The core PatchTST architecture files are kept close
to the original backbone so the DeltaTSF modifications are easy to inspect.
