# DeltaTSF for TimesNet

Copy these files into the official Time-Series-Library repository when running
TimesNet:

```text
data_provider/data_loader.py
exp/exp_long_term_forecasting.py
run.py
scripts/
```

The dataloader appends first-order delta channels after the raw channels. The
experiment file builds TimesNet with the original channel count and applies CES
sampling before the model forward pass. This is important for TimesNet because
its period selection uses frequency-domain statistics; CES is applied before
the frequency block, not by truncating FFT bins or frequency-domain features.

The CES logic is located in:

```text
exp/exp_long_term_forecasting.py
```

Look for:

```text
_ces_pair_inputs
_ces_loss
```

For scripts that did not contain a recorded best `w_diff1` value in the source
workspace, this integration uses `--w_diff1 1.0` as the default runnable value.
Tune it if you want to reproduce a dataset-specific best setting.
