import numpy as np
import torch


def build_first_order_delta_features(raw_values):
    """
    Build DeltaTSF input/output channels by concatenating raw values and
    first-order temporal differences.

    Input:
        raw_values: array with shape [T, C]

    Output:
        combined_values: array with shape [T, 2C], where the first C channels
        are raw values and the last C channels are first-order differences.
    """
    raw_values = np.asarray(raw_values)

    if raw_values.ndim == 1:
        raw_values = raw_values.reshape(-1, 1)

    diff1_values = raw_values[1:] - raw_values[:-1]

    # Keep the sequence length unchanged by padding the first difference.
    diff1_padding = diff1_values[:1]
    diff1_values = np.concatenate([diff1_padding, diff1_values], axis=0)

    return np.concatenate([raw_values, diff1_values], axis=1)


def deltatsf_loss(outputs, targets, orig_c, criterion, w_diff1):
    """
    Compute DeltaTSF training loss.

    The first orig_c channels correspond to raw values, while the remaining
    orig_c channels correspond to first-order temporal differences.
    """
    pred_raw = outputs[:, :, :orig_c]
    true_raw = targets[:, :, :orig_c]

    pred_diff = outputs[:, :, orig_c:]
    true_diff = targets[:, :, orig_c:]

    loss_raw = criterion(pred_raw, true_raw)
    loss_diff = criterion(pred_diff, true_diff)

    return loss_raw + w_diff1 * loss_diff


def compute_equivalent_sampling(batch_x, batch_y, orig_c, device):
    """
    Compute-Equivalent Sampling (CES) for DeltaTSF.

    For each variable in each sample, randomly choose either the raw channel
    or its first-order difference channel. This keeps the number of channels
    equal to the original backbone input size while still exposing the model
    to derivative representations during training.
    """
    B, T, _ = batch_x.shape
    M = orig_c

    # Keep the same random-number consumption pattern as the original code.
    # The sampled alpha is not used by CES, but preserving it avoids changing
    # the stochastic training trajectory when using the same random seed.
    _ = torch.rand((B, 1, M), device=device)

    offsets = torch.randint(0, 2, (B, 1, M), device=device) * M
    base_indices = torch.arange(M, device=device).view(1, 1, M)
    rand_indices = base_indices + offsets

    batch_x_sampled = torch.gather(batch_x, 2, rand_indices.expand(B, T, M))
    batch_y_sampled = torch.gather(
        batch_y,
        2,
        rand_indices.expand(B, batch_y.shape[1], M),
    )

    return batch_x_sampled, batch_y_sampled, offsets.squeeze(1)


def select_raw_channels(tensor, orig_c):
    """
    Select raw-value channels for validation and testing.

    DeltaTSF trains with raw and derivative channels, but forecasting metrics
    are computed only on the original raw time-series variables.
    """
    return tensor[:, :, :orig_c]

def ces_deltatsf_loss(outputs, targets, rand_offsets, pred_len, criterion, w_diff1):
    """
    Compute DeltaTSF loss under Compute-Equivalent Sampling.

    After CES, each variable is either a raw-value channel or a first-order
    difference channel. rand_offsets records which variables were sampled from
    difference channels.
    """
    is_diff = (rand_offsets > 0).unsqueeze(1).repeat(1, pred_len, 1)

    loss_raw = criterion(outputs[~is_diff], targets[~is_diff]) if (~is_diff).any() else 0
    loss_diff = criterion(outputs[is_diff], targets[is_diff]) if is_diff.any() else 0

    return loss_raw + w_diff1 * loss_diff