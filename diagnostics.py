"""Blind-well diagnostics used in the error analysis (R1.7, R2.28, R2.31, R3.2, R3.8).

For every target and blind well:
  * target shift: mean/std of the blind target vs. training; fraction outside the training 1-99 % range;
  * predictor-space novelty: median distance (scaled predictors) of blind samples to their 10 nearest
    real training samples, and the same distance when synthetic samples (ChemoGAN, seed 0, filtered)
    are added -> "coverage gain";
  * zone-resolved error: Delta MAE (augmented - real, seed-averaged predictions) inside
    high-heterogeneity zones (rolling std of the measured target in the top quartile) and elsewhere.
Usage: python diagnostics.py [results_dir]
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
import config as cfg, pre_processamento as prep, physics

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
SCEN = sys.argv[2] if len(sys.argv) > 2 else "chemogan_soft+none"
rows = []
for tgt in ("DWCA", "DWSI"):
    tdir = os.path.join(RES, tgt)
    fz = os.path.join(tdir, "synth_chemogan_soft_s0.npz")
    if not os.path.exists(fz):
        continue
    d = prep.build_dataset(cfg, tgt, verbose=False, merge_val=True); F = d["features"]; sc = d["scaler"]
    tr = pd.concat(d["train_wells"].values())[F].to_numpy()
    trs = sc.transform(tr)
    C = physics.ConstraintSet(F, cfg.pairs_for(tgt, False), cfg.TAU, cfg.NMR_COLS, cfg.NMR_TOL, sc)
    z = np.load(fz); Xs = z["X"]; n1 = len(tr) // cfg.WINDOW_T
    syn = Xs[:n1].reshape(-1, len(F))  # raw synthetic set (as used by ChemoGAN-P augmentation)
    nn_r = NearestNeighbors(n_neighbors=10).fit(trs[:, :-1])
    nn_rs = NearestNeighbors(n_neighbors=10).fit(np.vstack([trs[:, :-1], syn[:, :-1]]))
    # within-training reference distance (leave-well-out would be stricter; this is a scale reference)
    lo, hi = np.percentile(tr[:, -1], [1, 99])
    for w, df in d["blind_wells"].items():
        b = sc.transform(df.sort_values("DEPTH")[F].to_numpy())
        y = df.sort_values("DEPTH")[F[-1]].to_numpy()
        dr = nn_r.kneighbors(b[:, :-1])[0].mean(1)
        drs = nn_rs.kneighbors(b[:, :-1])[0].mean(1)
        rec = {"target": tgt, "well": w.replace("_BASE.las", ""), "n": len(y),
               "target_mean": y.mean(), "train_mean": tr[:, -1].mean(), "target_std": y.std(), "train_std": tr[:, -1].std(),
               "frac_outside_train_1_99": float(np.mean((y < lo) | (y > hi))),
               "kNN_dist_real": float(np.median(dr)), "kNN_dist_real+syn": float(np.median(drs)),
               "coverage_gain_%": float(100 * (1 - np.median(drs) / np.median(dr)))}
        # zone-resolved error from seed-averaged predictions
        k = w.replace(".las", "")
        P = {}
        for s in ("real", SCEN):
            fs = sorted(glob.glob(os.path.join(tdir, "pred", f"{s}_s*.npz")))
            if fs:
                P[s] = np.mean([np.load(f)[k + "__yhat"] for f in fs], 0)
        if len(P) == 2:
            rs = pd.Series(y).rolling(40, center=True, min_periods=10).std().to_numpy()
            zone = rs >= np.nanpercentile(rs, 75)
            for nm, msk in (("hetero", zone), ("other", ~zone)):
                rec[f"MAE_real_{nm}"] = float(np.mean(np.abs(y[msk] - P["real"][msk])))
                rec[f"dMAE_{nm}"] = float(np.mean(np.abs(y[msk] - P[SCEN][msk])) - rec[f"MAE_real_{nm}"])
            # error vs target level: where do improvements happen?
            q = np.quantile(y, [0, 0.25, 0.75, 1])
            for nm, (a, bq) in (("low", (q[0], q[1])), ("high", (q[2], q[3]))):
                m = (y >= a) & (y <= bq)
                rec[f"dMAE_target_{nm}"] = float(np.mean(np.abs(y[m] - P[SCEN][m])) - np.mean(np.abs(y[m] - P["real"][m])))
        rows.append(rec)
D = pd.DataFrame(rows)
os.makedirs(os.path.join(RES, "summary"), exist_ok=True)
D.to_csv(os.path.join(RES, "summary", f"diagnostics_{SCEN.replace('+', '_')}.csv"), index=False)
pd.set_option("display.width", 250)
print(D.round(4).T.to_string())
