"""Aggregate revision-1 results into the tables and statistics reported in the paper.

Usage: python analysis.py [results_dir]
Writes results/summary/*.csv and *.tex.
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import downstream as ds

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = os.path.join(RES, "summary"); os.makedirs(OUT, exist_ok=True)
WELLS = {"7-SPH-3-SPS_BASE.las": "7-SPH-3-SPS", "9-SPS-77A-SP_BASE.las": "9-SPS-77A", "9-SPS-97-SP_BASE.las": "9-SPS-97"}
MODEL_LABEL = {"chemogan": "ChemoGAN-H", "gan_uncon": "Unconstrained GAN", "chemogan_soft": "ChemoGAN-P",
               "gan_uncon_matched": "Unconstrained GAN + volume-matched rejection", "tvae": "Transformer-VAE (GMM)", "tvae_prior": "Transformer-VAE (prior)", "ddpm": "DDPM"}


def ms(x, d=3):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "--"
    if len(x) == 1:
        return f"{x[0]:.{d}f}"
    return f"{x.mean():.{d}f} $\\pm$ {x.std(ddof=1):.{d}f}"


def load_preds(tdir, scen, seed):
    f = os.path.join(tdir, "pred", f"{scen}_s{seed}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    out = {}
    for w in WELLS:
        k = w.replace(".las", "")
        out[w] = (z[k + "__depth"], z[k + "__y"], z[k + "__yhat"])
    return out


def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


def seed_avg_bootstrap(tdir, scen, seeds, B=2000, block=40, rng_seed=0):
    """CI of the seed-averaged paired Delta R2 per well (same depth blocks for every seed)."""
    res = {}
    P = {s: (load_preds(tdir, "real", s), load_preds(tdir, scen, s)) for s in seeds}
    P = {s: v for s, v in P.items() if v[0] is not None and v[1] is not None}
    if not P:
        return res
    rng = np.random.default_rng(rng_seed)
    for w in WELLS:
        y = next(iter(P.values()))[0][w][1]
        n = len(y); nb = int(np.ceil(n / block))
        deltas = np.empty(B)
        for b in range(B):
            st = rng.integers(0, max(1, n - block + 1), nb)
            idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
            deltas[b] = np.mean([r2(y[idx], P[s][1][w][2][idx]) - r2(y[idx], P[s][0][w][2][idx]) for s in P])
        point = np.mean([r2(y, P[s][1][w][2]) - r2(y, P[s][0][w][2]) for s in P])
        res[w] = (point, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5))
    return res


def ensemble_r2(tdir, scen, seeds):
    preds = [load_preds(tdir, scen, s) for s in seeds]
    preds = [p for p in preds if p is not None]
    if not preds:
        return {}
    return {w: r2(preds[0][w][1], np.mean([p[w][2] for p in preds], axis=0)) for w in WELLS}


def main():
    gen_all, blind_all, ref_all = [], [], {}
    for tdir in sorted(glob.glob(os.path.join(RES, "DW*"))):
        tgt = os.path.basename(tdir)
        if os.path.exists(os.path.join(tdir, "gen_metrics.csv")):
            gen_all.append(pd.read_csv(os.path.join(tdir, "gen_metrics.csv")))
        if os.path.exists(os.path.join(tdir, "blind_metrics.csv")):
            blind_all.append(pd.read_csv(os.path.join(tdir, "blind_metrics.csv")))
        if os.path.exists(os.path.join(tdir, "reference.json")):
            ref_all[tgt] = json.load(open(os.path.join(tdir, "reference.json")))["real_reference_pvr"]
    G = pd.concat(gen_all) if gen_all else pd.DataFrame()
    Bm = pd.concat(blind_all) if blind_all else pd.DataFrame()
    G.to_csv(os.path.join(OUT, "gen_all.csv"), index=False)
    Bm.to_csv(os.path.join(OUT, "blind_all.csv"), index=False)

    # ---------------- Table: generative fidelity + PVR ----------------
    lines = []
    for tgt in sorted(G.target.unique()):
        g = G[G.target == tgt]
        ref = ref_all.get(tgt, {})
        lines.append({"target": tgt, "model": "Real windows (reference)", "n": "--",
                      "PVR": f"{ref.get('any', np.nan):.1f}", "PVR_hard": f"{ref.get('nmr', np.nan):.1f}",
                      "PVR_soft": f"{ref.get('soft', np.nan):.1f}",
                      "PE_diag": f"{[v for k, v in ref.items() if k.startswith('diag')][0]:.1f}" if ref else "--",
                      "W1": "--", "KS": "--", "corrMAE": "--", "minutes": "--"})
        for m in MODEL_LABEL:
            gm = g[g.model == m]
            if gm.empty:
                continue
            diag = [c for c in gm.columns if c.startswith("PVR_diag")]
            lines.append({"target": tgt, "model": MODEL_LABEL[m], "n": len(gm),
                          "PVR": ms(gm.PVR_any, 1), "PVR_hard": ms(gm.PVR_nmr, 1), "PVR_soft": ms(gm.PVR_soft, 1),
                          "PE_diag": ms(gm[diag[0]], 1) if diag else "--",
                          "W1": ms(gm.W1_mean, 4), "KS": ms(gm.KS_mean, 3), "corrMAE": ms(gm.corr_MAE, 3),
                          "minutes": ms(gm.train_sec / 60.0, 0)})
    T1 = pd.DataFrame(lines); T1.to_csv(os.path.join(OUT, "table_generative.csv"), index=False)
    print("\n=== Generative fidelity and PVR ===\n", T1.to_string(index=False))

    # ---------------- Table: downstream per well ----------------
    rows, boot_rows, ens_rows = [], [], []
    for tgt in sorted(Bm.target.unique()):
        b = Bm[Bm.target == tgt]
        tdir = os.path.join(RES, tgt)
        base = b[b.scenario == "real"].set_index(["seed", "well"])
        for scen in sorted(b.scenario.unique()):
            bs = b[b.scenario == scen]
            seeds = sorted(bs.seed.unique())
            for w in WELLS:
                bw = bs[bs.well == w]
                if bw.empty:
                    continue
                d = [r - base.loc[(s, w), "R2"] for s, r in zip(bw.seed, bw.R2) if (s, w) in base.index]
                rows.append({"target": tgt, "scenario": scen, "well": WELLS[w], "n_seeds": len(bw),
                             "R2_mean": bw.R2.mean(), "R2_sd": bw.R2.std(ddof=1) if len(bw) > 1 else 0.0,
                             "MAE_mean": bw.MAE.mean(), "RMSE_mean": bw.RMSE.mean(),
                             "dR2_mean": np.mean(d) if d else np.nan,
                             "dR2_sd": np.std(d, ddof=1) if len(d) > 1 else 0.0,
                             "frac_improved": np.mean(np.array(d) > 0) if d else np.nan,
                             "dR2_min": np.min(d) if d else np.nan, "dR2_max": np.max(d) if d else np.nan})
            if scen != "real" and "TSTR" not in scen:
                bt = seed_avg_bootstrap(tdir, scen, seeds)
                for w, (pt, lo, hi) in bt.items():
                    boot_rows.append({"target": tgt, "scenario": scen, "well": WELLS[w], "dR2": pt, "lo": lo, "hi": hi,
                                      "significant": (lo > 0) or (hi < 0)})
            er = ensemble_r2(tdir, scen, seeds)
            for w, v in er.items():
                ens_rows.append({"target": tgt, "scenario": scen, "well": WELLS[w], "R2_ensemble": v, "n_seeds": len(seeds)})
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "table_downstream_long.csv"), index=False)
    BT = pd.DataFrame(boot_rows); BT.to_csv(os.path.join(OUT, "table_bootstrap.csv"), index=False)
    EN = pd.DataFrame(ens_rows); EN.to_csv(os.path.join(OUT, "table_ensemble.csv"), index=False)
    pd.set_option("display.width", 250)
    main_s = ["real", "chemogan+none", "chemogan+any", "gan_uncon+none", "gan_uncon+any", "chemogan_soft+any", "gan_uncon_matched+none",
              "tvae+any", "tvae+none", "tvae_prior+none", "ddpm+none", "ddpm+any"]
    if not D.empty:
        Dm = D[D.scenario.isin(main_s)]
        print("\n=== Downstream (mean over seeds) ===")
        print(Dm.pivot_table(index=["target", "scenario"], columns="well", values="R2_mean").round(4))
        print(Dm.pivot_table(index=["target", "scenario"], columns="well", values="dR2_mean").round(4))
        print(Dm.pivot_table(index=["target", "scenario"], columns="well", values="frac_improved").round(2))
        print("\n=== Average over wells (dR2) ===")
        print(D.groupby(["target", "scenario"]).dR2_mean.mean().round(4).to_string())
    # Pooled paired test: one Delta R2 per (seed, well); Wilcoxon signed-rank (two-sided)
    from scipy.stats import wilcoxon
    pooled = []
    for tgt in sorted(Bm.target.unique()) + ["ALL"]:
        b = Bm if tgt == "ALL" else Bm[Bm.target == tgt]
        base = b[b.scenario == "real"].set_index(["target", "seed", "well"]).R2
        for scen in sorted(set(b.scenario) - {"real"}):
            if "TSTR" in scen:
                continue
            bs = b[b.scenario == scen].set_index(["target", "seed", "well"]).R2
            d = (bs - base.reindex(bs.index)).dropna().to_numpy()
            if len(d) < 3:
                continue
            p_ = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            pooled.append({"target": tgt, "scenario": scen, "n_pairs": len(d), "dR2_mean": d.mean(),
                           "dR2_median": np.median(d), "frac_positive": np.mean(d > 0), "wilcoxon_p": p_})
    PL = pd.DataFrame(pooled); PL.to_csv(os.path.join(OUT, "table_pooled_test.csv"), index=False)
    if not PL.empty:
        print("\n=== Pooled paired test over seeds x wells ===")
        print(PL[~PL.scenario.str.contains("@r")].round(4).to_string(index=False))
    # Which properties of a synthetic set predict its downstream utility? (one point per run, raw synthetic set)
    from scipy.stats import spearmanr
    if not G.empty and not Bm.empty:
        rows_u = []
        for _, g in G.iterrows():
            b = Bm[(Bm.target == g.target) & (Bm.seed == g.seed)]
            base = b[b.scenario == "real"].set_index("well").R2
            a = b[b.scenario == f"{g.model}+none"].set_index("well").R2
            if len(a) == 3:
                rows_u.append({"target": g.target, "model": g.model, "seed": g.seed, "dR2": float((a - base).mean()),
                               "KS": g.KS_mean, "W1": g.W1_mean, "corrMAE": g.corr_MAE, "PVR_soft": g.PVR_soft, "PVR_hard": g.PVR_nmr})
        U = pd.DataFrame(rows_u); U.to_csv(os.path.join(OUT, "utility_vs_metrics.csv"), index=False)
        print("\n=== Spearman correlation between run-level metrics and mean dR2 (raw synthetic, all runs) ===")
        for m in ["KS", "W1", "corrMAE", "PVR_soft", "PVR_hard"]:
            r, p_ = spearmanr(U[m], U.dR2)
            print(f"  {m:9s} rho={r:+.2f} p={p_:.3f} n={len(U)}")
    if not BT.empty:
        print("\n=== Seed-averaged dR2 with depth-block bootstrap 95% CI ===")
        print(BT.round(4).to_string(index=False))
    if not EN.empty:
        print("\n=== Seed-ensemble R2 ===")
        print(EN.pivot_table(index=["target", "scenario"], columns="well", values="R2_ensemble").round(4))


if __name__ == "__main__":
    main()
