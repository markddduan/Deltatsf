# DeltaTSF for TimeMixer

Copy these files into the official TimeMixer repository:

```text
data_provider/data_loader.py
exp/exp_long_term_forecasting.py
run.py
scripts/
```

The dataloader appends first-order delta channels after the raw channels. The
experiment file builds TimeMixer with the original channel count and performs
CES sampling before the TimeMixer forward pass, so the multi-scale mixing path
receives a standard `C`-channel input.

The CES logic is located in:

```text
exp/exp_long_term_forecasting.py
```

Look for:

```text
_gpu_rand_coupling
_ces_loss
```

The included scripts are copied from the TimeMixer DeltaTSF runs and keep their
original hyperparameter settings.
