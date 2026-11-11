"""Model-selection pilot on VALIDATION wells only (never blind wells).

Trains ChemoGAN / unconstrained GAN on the 15 training wells, takes generator snapshots
at several epochs and, for each snapshot, measures PVR, fidelity, and the change in
XGBoost R2 on the 3 validation wells when the synthetic windows are added.
"""
import sys, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, tensorflow as tf
import config as cfg, pre_processamento as prep, physics, gan_lib, gen_metrics, downstream as ds
from gan_trainer import ChemoGANTrainer

tgt = sys.argv[1]; model = sys.argv[2]; epochs = int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ls = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
snaps = [e for e in (25, 50, 100, 150, 200, 300, 400, 600, 800) if e <= epochs]
out = f"results/selection_{tgt}"; os.makedirs(out, exist_ok=True)
tf.random.set_seed(seed); np.random.seed(seed)
d = prep.build_dataset(cfg, tgt, merge_val=False)
F = d["features"]
C_full = physics.ConstraintSet(F, cfg.pairs_for(tgt), cfg.TAU, cfg.NMR_COLS, cfg.NMR_TOL, d["scaler"])
C = physics.ConstraintSet(F, cfg.pairs_for(tgt, False), cfg.TAU, cfg.NMR_COLS, cfg.NMR_TOL, d["scaler"])
Xr, yr, gr = ds.stack_wells(d["train_wells"], F)
params = ds.tune(Xr, yr, gr, n_iter=25, cache=f"{out}/xgb_params_sel.json")
base = ds.fit_predict(params, Xr, yr, d["val_wells"], F, seed)
base_m = {w: ds.metrics(y, yh) for w, (_, y, yh) in base.items()}
real2d = d["scaler"].transform(np.column_stack([Xr, yr]))
n_gen = len(Xr) // cfg.WINDOW_T
rows = []

def snap(ep, G):
    Xs = G(tf.random.normal([n_gen, cfg.LATENT_Z], seed=seed), training=False).numpy()
    P = C.to_physical(Xs); pvr, v = C.pvr(P); fid = gen_metrics.fidelity(real2d, Xs.reshape(-1, Xs.shape[-1]))
    for flt in ("none", "any"):
        keep = np.ones(len(Xs), bool) if flt == "none" else ~v["any"]
        syn = P[keep].reshape(-1, P.shape[-1])
        pr = ds.fit_predict(params, np.vstack([Xr, syn[:, :-1]]), np.concatenate([yr, syn[:, -1]]), d["val_wells"], F, seed)
        for w, (_, y, yh) in pr.items():
            m = ds.metrics(y, yh)
            rows.append(dict(model=model, ls=ls, seed=seed, epoch=ep, filter=flt, well=w, dR2=m["R2"] - base_m[w]["R2"],
                             R2=m["R2"], base_R2=base_m[w]["R2"], PVR=pvr["any"], PVR_nmr=pvr["nmr"],
                             W1=fid["W1_mean"], KS=fid["KS_mean"], corrMAE=fid["corr_MAE"], kept=keep.mean()))
    r = pd.DataFrame(rows); r = r[r.epoch == ep]
    print(f"[snap {ep}] PVR {pvr['any']:.1f} (nmr {pvr['nmr']:.1f}) W1 {fid['W1_mean']:.4f} KS {fid['KS_mean']:.3f} "
          f"corr {fid['corr_MAE']:.3f} | mean dR2 none {r[r['filter']=='none'].dR2.mean():+.4f} any {r[r['filter']=='any'].dR2.mean():+.4f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/pilot_{model}_ls{ls}_s{seed}.csv", index=False)

lam = cfg.LAMBDAS if model in ("chemogan", "chemogan_soft") else {"tv": 0, "nmr": 0, "corr": 0}
T, A = d["X_train"].shape[1:]
G = (gan_lib.make_generator_ordered(T, A, cfg.LATENT_Z, C.i_nmr, C.data_min, C.data_range)
     if model == "chemogan" else gan_lib.make_generator_1d_constrained(T, A, cfg.LATENT_Z)); D = gan_lib.make_discriminator_1d_robust(T, A)
tr = ChemoGANTrainer(G, D, C_full, cfg.LATENT_Z, lambdas=lam, label_smoothing=ls, lr=cfg.GAN_LR)
tds = tf.data.Dataset.from_tensor_slices(d["X_train"]).shuffle(len(d["X_train"]), seed=seed).batch(cfg.BATCH_SIZE)
vds = tf.data.Dataset.from_tensor_slices(d["X_val"]).batch(cfg.BATCH_SIZE)
print("base val R2:", {w[:10]: round(m["R2"], 3) for w, m in base_m.items()}, flush=True)
h = tr.fit(tds, vds, epochs, snapshot_epochs=snaps, snapshot_fn=snap, log_every=50)
json.dump(h, open(f"{out}/pilot_hist_{model}_ls{ls}_s{seed}.json", "w"))
