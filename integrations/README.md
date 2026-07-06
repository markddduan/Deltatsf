# DeltaTSF Backbone Integrations

This directory contains minimal integration files for applying DeltaTSF to
additional forecasting backbones. The main repository remains the complete
PatchTST implementation. For the backbones below, copy the files from each
integration folder into the matching paths of the corresponding upstream
repository.

Each integration follows the same high-level pattern:

1. The dataloader appends first-order temporal differences after the raw
   channels.
2. The experiment file keeps the backbone at the original channel count.
3. CES randomly selects either the raw or delta member of each channel pair
   before the backbone forward pass.
4. The CES loss applies `w_diff1` only to selected delta targets.

The backbone-specific folders are intentionally small. They are not full forks
of the upstream projects.

| Folder | Backbone | Notes |
| --- | --- | --- |
| `itransformer/` | iTransformer | Uses the input-length-720 scripts from `itransformer_copy`. |
| `timemixer/` | TimeMixer | Applies CES before the multi-scale mixing blocks. |
| `timesnet/` | TimesNet | Applies CES before TimesNet's frequency-domain period selection. |

