# DeltaTSF for iTransformer

Source used for this integration:

```text
/media/D1/temp22/itransformer_copy
```

Copy these files into the official iTransformer repository:

```text
data_provider/data_loader.py
experiments/exp_long_term_forecasting.py
run.py
scripts/difftst_best/
```

This integration keeps the official iTransformer model unchanged. The
dataloader returns `raw + delta` channels, while
`experiments/exp_long_term_forecasting.py` builds the backbone with the
original channel count and performs CES sampling before the forward pass.

The `scripts/difftst_best/` scripts are the input-length-720 scripts used for
the DeltaTSF iTransformer integration. The `enc_in`, `dec_in`, and `c_out`
values are doubled in the scripts because the dataloader exposes raw and delta
channels. The experiment file restores the backbone to the original channel
count internally.

Run from the root of the iTransformer repository, for example:

```bash
bash scripts/difftst_best/etth1.sh
```

