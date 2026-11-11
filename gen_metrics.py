"""Distributional fidelity metrics.

Revision 1: Wasserstein distances are computed on MinMax-scaled features (unit-free),
so that logs with large physical ranges (e.g. GR in API, DT in us/ft) do not dominate
the mean; the first submission averaged W1 across heterogeneous physical units.
"""
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp


def fidelity(real_scaled_2d, synth_scaled_2d):
    A = real_scaled_2d.shape[1]
    wd = [wasserstein_distance(real_scaled_2d[:, a], synth_scaled_2d[:, a]) for a in range(A)]
    ks = [ks_2samp(real_scaled_2d[:, a], synth_scaled_2d[:, a]).statistic for a in range(A)]
    cr = np.nan_to_num(np.corrcoef(real_scaled_2d, rowvar=False))
    cs = np.nan_to_num(np.corrcoef(synth_scaled_2d, rowvar=False))
    return {"W1_mean": float(np.mean(wd)), "KS_mean": float(np.mean(ks)),
            "corr_MAE": float(np.mean(np.abs(cr - cs))), "W1_per_feature": [float(x) for x in wd]}


def window_corr_spectrum(Xs, i, j):
    a = Xs[:, :, i] - Xs[:, :, i].mean(1, keepdims=True)
    b = Xs[:, :, j] - Xs[:, :, j].mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
