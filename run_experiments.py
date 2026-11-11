"""Revision-1 experiment driver (multi-seed, ablation, controlled comparison).

Example:
  python run_experiments.py --target DWSI --models chemogan gan_uncon tvae ddpm --seeds 0 1 2 3 4
Outputs (per target) in results/<target>/:
  synth_<model>_s<seed>.npz      generated windows (scaled) + violation masks
  gen_metrics.csv                 PVR components and fidelity per model/seed
  blind_metrics.csv               per-well MAE/RMSE/R2 per scenario/seed
  history_<model>_s<seed>.json    training curves
"""
import argparse, json, os, random, time, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg
import pre_processamento as prep
import physics
import gan_lib
import gen_metrics
import downstream as ds
from gan_trainer import ChemoGANTrainer
import transformer_trainer
import diffusion_trainer


def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def train_generate(model, data, C, seed, n_gen, epochs, args):
    T, A = data["X_train"].shape[1:]
    X = data["X_train"]
    t0 = time.time()
    hist = {}
    if model in ("chemogan", "gan_uncon", "gan_uncon_matched", "chemogan_common", "chemogan_soft"):
        # chemogan        : hard NMR ordering layer + TV + correlation penalties (revision 1)
        # chemogan_common : same, without the target-specific PE pair
        # chemogan_soft   : first-submission formulation (all constraints as penalties)
        # gan_uncon       : same architecture, no physics at all
        lam = {"gan_uncon": {"tv": 0.0, "nmr": 0.0, "corr": 0.0},
               "gan_uncon_matched": {"tv": 0.0, "nmr": 0.0, "corr": 0.0}}.get(model, cfg.LAMBDAS)
        if model in ("chemogan", "chemogan_common"):
            G = gan_lib.make_generator_ordered(T, A, cfg.LATENT_Z, C.i_nmr, C.data_min, C.data_range)
        else:
            G = gan_lib.make_generator_1d_constrained(T, A, cfg.LATENT_Z)
        D = gan_lib.make_discriminator_1d_robust(T, A)
        tr = ChemoGANTrainer(G, D, C, cfg.LATENT_Z, lambdas=lam, label_smoothing=args.label_smoothing, lr=cfg.GAN_LR)
        tds = tf.data.Dataset.from_tensor_slices(X).shuffle(len(X), seed=seed).batch(cfg.BATCH_SIZE)
        vds = tf.data.Dataset.from_tensor_slices(data["X_val"]).batch(cfg.BATCH_SIZE) if len(data["X_val"]) else None
        hist = tr.fit(tds, vds, epochs)
        Xs = np.concatenate([G(tf.random.normal([min(4096, n_gen - i), cfg.LATENT_Z]), training=False).numpy()
                             for i in range(0, n_gen, 4096)])
        if model in ("gan_uncon", "gan_uncon_matched"):
            # R3.3: same post-hoc rejection filter, but oversample until the retained set has the
            # same size as the unfiltered pool (volume-matched rejection sampling).
            Ce = data["C_eval"]; kept, tried = [], 0
            while sum(len(k) for k in kept) < n_gen and tried < 400 * n_gen:
                Xb = G(tf.random.normal([8192, cfg.LATENT_Z]), training=False).numpy()
                _, vb = Ce.pvr(Ce.to_physical(Xb)); kept.append(Xb[~vb["any"]]); tried += len(Xb)
            Xm = np.concatenate(kept)[:n_gen] if kept else np.zeros((0,) + Xs.shape[1:], "float32")
            print(f"   volume-matched rejection: kept {len(Xm)} of {tried} generated windows "
                  f"({100 * len(Xm) / tried:.2f}%)", flush=True)
            hist["matched_acceptance"] = float(len(Xm) / tried)
            if len(Xm) < 10:
                print("   volume-matched rejection: too few accepted windows, scenario skipped", flush=True)
            elif model == "gan_uncon_matched":
                Xs = Xm
            else:
                hist["_extra"] = {"gan_uncon_matched": Xm.astype("float32")}
    elif model == "tvae":
        gen, hist = transformer_trainer.train_transformer(
            X, T, A, cfg.LATENT_Z, epochs=args.tvae_epochs, batch_size=cfg.BATCH_SIZE,
            beta=args.tvae_beta, warmup_epochs=args.tvae_warmup, sampler=args.tvae_sampler, seed=seed)
        Xs = gen.generate(n_gen)
        gen.sampler = "prior"  # same trained model, first-submission sampling (for R1.6)
        gen.rng = np.random.default_rng(seed)
        hist["_extra"] = {"tvae_prior": gen.generate(n_gen).astype("float32")}
    elif model == "ddpm":
        man = diffusion_trainer.train_diffusion(X, T, A, epochs=args.ddpm_epochs, batch_size=cfg.BATCH_SIZE)
        Xs = np.clip(man.generate(n_gen, T, A), 0.0, 1.0)
    else:
        raise ValueError(model)
    return Xs.astype("float32"), hist, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="DWSI")
    ap.add_argument("--models", nargs="+", default=["chemogan", "gan_uncon", "tvae", "ddpm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--gan_epochs", type=int, default=300)
    ap.add_argument("--ddpm_epochs", type=int, default=300)
    ap.add_argument("--tvae_epochs", type=int, default=300)
    ap.add_argument("--tvae_beta", type=float, default=1e-5)
    ap.add_argument("--tvae_warmup", type=int, default=0)
    ap.add_argument("--tvae_sampler", default="gmm")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                    help="synthetic-to-real point ratios evaluated (subsampled from one generated pool)")
    ap.add_argument("--filters", nargs="+", default=["none", "any"])
    ap.add_argument("--xgb_iter", type=int, default=100)
    ap.add_argument("--selection", action="store_true",
                    help="model-selection mode: generators fitted on train wells, evaluated on VAL wells")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tgt = args.target
    out = os.path.join(args.out, ("selection_" if args.selection else "") + tgt)
    os.makedirs(out, exist_ok=True)
    data = prep.build_dataset(cfg, tgt, merge_val=not args.selection)
    feats = data["features"]
    eval_wells = data["val_wells"] if args.selection else data["blind_wells"]
    fit_wells = data["train_wells"] if args.selection else data["train_wells"]  # merged when final
    # C_full: training constraint set of ChemoGAN (common pairs + target-specific pair).
    # C (evaluation): PVR as defined in the manuscript = NMR chain + the two structural
    # pairs; it is also the rejection rule. The target-specific pair involves the label and
    # is therefore reported only as a diagnostic, never used to filter samples.
    C_full = physics.ConstraintSet(feats, cfg.pairs_for(tgt), cfg.TAU, cfg.NMR_COLS, cfg.NMR_TOL, data["scaler"])
    C = physics.ConstraintSet(feats, cfg.pairs_for(tgt, False), cfg.TAU, cfg.NMR_COLS, cfg.NMR_TOL, data["scaler"])

    data["C_eval"] = C
    Xr_phys = C.to_physical(data["X_train"])
    ref, _ = C.pvr(Xr_phys)
    ref_full, _ = C_full.pvr(Xr_phys)
    ref = {**ref, **{f"diag_{k}": v for k, v in ref_full.items() if k.startswith("corr_PE")}}
    json.dump({"real_reference_pvr": ref, "n_windows": int(len(Xr_phys)),
               "train_wells": sorted(fit_wells)}, open(os.path.join(out, "reference.json"), "w"), indent=2)
    print("[reference PVR on real windows]", {k: round(v, 2) for k, v in ref.items()})

    Xreal, yreal, greal = ds.stack_wells(fit_wells, feats)
    params = ds.tune(Xreal, yreal, greal, n_iter=args.xgb_iter, cache=os.path.join(out, "xgb_params.json"))
    print("[xgb params]", params)
    real_scaled_2d = data["scaler"].transform(np.column_stack([Xreal, yreal]))
    n_gen = int(round(max(args.ratios) * len(Xreal) / cfg.WINDOW_T))
    n_unit = len(Xreal) / cfg.WINDOW_T

    rows_path = os.path.join(out, f"blind_metrics{args.tag}.csv")
    gen_path = os.path.join(out, f"gen_metrics{args.tag}.csv")
    pred_dir = os.path.join(out, "pred"); os.makedirs(pred_dir, exist_ok=True)

    def record(rows, scen, seed, preds, extra):
        for w, (d, y, yh) in preds.items():
            rows.append({"target": tgt, "scenario": scen, "seed": seed, "well": w, **ds.metrics(y, yh), **extra})
        np.savez_compressed(os.path.join(pred_dir, f"{scen}_s{seed}.npz"),
                            **{w.replace(".las", "") + "__" + k: v for w, t in preds.items() for k, v in zip(("depth", "y", "yhat"), t)})

    def append(path, rows):
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    # Real-only baseline with the same seeds (XGBoost subsampling is stochastic).
    done = pd.read_csv(rows_path) if os.path.exists(rows_path) else pd.DataFrame(columns=["scenario", "seed"])
    for seed in args.seeds:
        if ((done.scenario == "real") & (done.seed == seed)).any():
            continue
        rows = []
        record(rows, "real", seed, ds.fit_predict(params, Xreal, yreal, eval_wells, feats, seed), {"n_synth": 0, "ratio": 0.0})
        append(rows_path, rows)

    def evaluate(model, Xs, seed, hist, secs):
        Xs_phys = C.to_physical(Xs)
        n1 = int(round(n_unit))
        pvr, viol1 = C.pvr(Xs_phys[:n1])  # reported PVR / fidelity on the ratio-1 set
        pfull, _ = C_full.pvr(Xs_phys[:n1])
        pvr.update({f"diag_{k}": v for k, v in pfull.items() if k.startswith("corr_PE")})
        _, viol = C.pvr(Xs_phys)
        fid = gen_metrics.fidelity(real_scaled_2d, Xs[:n1].reshape(-1, Xs.shape[-1]))
        np.savez_compressed(os.path.join(out, f"synth_{model}_s{seed}.npz"), X=Xs,
                            **{f"viol_{k}": v for k, v in viol.items()})
        json.dump({k: v for k, v in hist.items()}, open(os.path.join(out, f"history_{model}_s{seed}.json"), "w"))
        grow = {"target": tgt, "model": model, "seed": seed, "train_sec": secs,
                **{f"PVR_{k}": v for k, v in pvr.items()},
                **{k: v for k, v in fid.items() if k != "W1_per_feature"},
                "W1_per_feature": json.dumps(fid["W1_per_feature"])}
        rows = []
        for flt in args.filters:
            keep = np.ones(len(Xs), bool) if flt == "none" else ~viol[{"any": "any", "hard": "nmr", "soft": "soft"}[flt]]
            grow[f"retained_{flt}"] = float(keep.mean())
            for ratio in args.ratios:
                # The first n windows of the pool are used (i.i.d. samples), then the filter is applied,
                # so that rejection reduces the retained volume exactly as in deployment.
                n = int(round(ratio * n_unit))
                sel = keep[:n]
                syn = Xs_phys[:n][sel].reshape(-1, Xs.shape[-1])
                if len(syn) == 0:
                    continue
                tag = "" if ratio == 1.0 else f"@r{ratio:g}"
                Xa = np.vstack([Xreal, syn[:, :-1]]); ya = np.concatenate([yreal, syn[:, -1]])
                record(rows, f"{model}+{flt}{tag}", seed, ds.fit_predict(params, Xa, ya, eval_wells, feats, seed),
                       {"n_synth": int(len(syn)), "ratio": ratio})
                if ratio == 1.0:  # Train-on-synthetic, test-on-real (TSTR)
                    record(rows, f"{model}+{flt}+TSTR", seed, ds.fit_predict(params, syn[:, :-1], syn[:, -1], eval_wells, feats, seed),
                           {"n_synth": int(len(syn)), "ratio": ratio})
        append(gen_path, [grow]); append(rows_path, rows)
        print(f"   PVR {pvr['any']:.1f}% (hard {pvr['nmr']:.1f}, soft {pvr['soft']:.1f}) | W1 {fid['W1_mean']:.4f} "
              f"KS {fid['KS_mean']:.3f} corrMAE {fid['corr_MAE']:.3f} | {secs:.0f}s", flush=True)

    for model in args.models:
        Cm = C if model == "chemogan_common" else C_full
        for seed in args.seeds:
            if ((done.scenario == f"{model}+none") & (done.seed == seed)).any():
                print(f"skip {model} s{seed}"); continue
            set_seed(seed)
            print(f"\n=== {tgt} | {model} | seed {seed} ===", flush=True)
            Xs0, hist, secs = train_generate(model, data, Cm, seed, n_gen, args.gan_epochs, args)
            extra = hist.pop("_extra", {}) if isinstance(hist, dict) else {}
            for model_name, Xs in [(model, Xs0)] + list(extra.items()):
                evaluate(model_name, Xs, seed, hist, secs)
            tf.keras.backend.clear_session()
    print("done")


if __name__ == "__main__":
    main()
