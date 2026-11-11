"""Figures for revision 1 (static, print-ready). Usage: python figures.py [results_dir] [out_dir]"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"chemogan": "#2a78d6", "gan_uncon": "#eb6834", "chemogan_soft": "#1baf7a",
       "tvae": "#eda100", "tvae_prior": "#e87ba4", "ddpm": "#008300", "gan_uncon_matched": "#6250d6", "real": INK2}
MRK = {"chemogan": "o", "gan_uncon": "s", "chemogan_soft": "^", "tvae": "D", "tvae_prior": "v", "ddpm": "P", "gan_uncon_matched": "X"}
LAB = {"chemogan": "ChemoGAN-H", "gan_uncon": "Unconstrained GAN", "chemogan_soft": "ChemoGAN-P",
       "tvae": "Transformer-VAE (GMM)", "tvae_prior": "Transformer-VAE (prior)", "ddpm": "DDPM", "gan_uncon_matched": "Unconstr. GAN + matched rejection", "real": "Real only"}
WELLS = {"7-SPH-3-SPS_BASE.las": "7-SPH-3-SPS", "9-SPS-77A-SP_BASE.las": "9-SPS-77A", "9-SPS-97-SP_BASE.las": "9-SPS-97"}

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10.5,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def fig_convergence():
    tg = [t for t in ("DWCA", "DWSI") if os.path.exists(os.path.join(RES, t, "history_chemogan_s0.json"))]
    if not tg:
        return
    fig, axes = plt.subplots(len(tg), 2, figsize=(13, 5.8 * len(tg)), squeeze=False)
    for r, t in enumerate(tg):
        h = json.load(open(os.path.join(RES, t, "history_chemogan_s0.json")))
        hu_f = os.path.join(RES, t, "history_gan_uncon_s0.json")
        hu = json.load(open(hu_f)) if os.path.exists(hu_f) else None
        ep = np.arange(1, len(h["g_adv"]) + 1)
        ax = axes[r, 0]
        ax.plot(ep, h["d_loss"], color=INK2, label="Discriminator loss")
        ax.plot(ep, h["g_adv"], color=COL["chemogan"], label="Generator adversarial loss (ChemoGAN-H)")
        if hu is not None:
            ax.plot(ep, hu["g_adv"], color=COL["gan_uncon"], lw=1.6, ls=":", label="Generator adversarial loss (unconstrained GAN)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"({'ac'[r]}) {t}: adversarial dynamics (seed 0)", loc="left")
        ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        ax = axes[r, 1]
        keys = [k for k in h if k.startswith("corr_")] + ["tv"]
        names = {"tv": "TV smoothness", "corr_NPHI_RHOB": "Corr. NPHI–RHOB", "corr_NPHI_DT": "Corr. NPHI–DT",
                 "corr_PE_DWSI": "Corr. PE–DWSI", "corr_PE_DWCA": "Corr. PE–DWCA"}
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
        sm = lambda v: pd.Series(v).rolling(10, min_periods=1).mean().to_numpy()
        for k, c in zip(keys, cols):
            ax.plot(ep, np.maximum(sm(h[k]), 1e-6), color=c, label=f"{names.get(k, k)}")
            if hu is not None and k in hu:
                ax.plot(ep, np.maximum(sm(hu[k]), 1e-6), color=c, ls=":", lw=1.8)
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Penalty value (log scale)")
        ax.set_title(f"({'bd'[r]}) {t}: petrophysical penalty terms (10-epoch mean)", loc="left")
        ax.plot([], [], color=INK2, lw=2, label="ChemoGAN-H (solid)")
        ax.plot([], [], color=INK2, ls=":", lw=1.8, label="Unconstrained GAN, not optimised (dotted)")
        ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, "fig_convergence")


def fig_tradeoff():
    G = pd.read_csv(os.path.join(RES, "summary", "gen_all.csv"))
    tg = sorted(G.target.unique())
    fig, axes = plt.subplots(1, len(tg), figsize=(6.4 * len(tg), 6.2), squeeze=False)
    for i, t in enumerate(tg):
        ax = axes[0, i]; g = G[G.target == t]
        ref = json.load(open(os.path.join(RES, t, "reference.json")))["real_reference_pvr"]["any"]
        ax.axhline(ref, color=INK2, ls="--", lw=1.5)
        ax.text(ax.get_xlim()[1] if False else 0.99, ref + 1.5, f"Real windows: {ref:.1f}%", color=INK2,
                ha="right", transform=ax.get_yaxis_transform(), fontsize=10.5)
        for m in COL:
            gm = g[g.model == m]
            if gm.empty:
                continue
            ax.scatter(gm.KS_mean, gm.PVR_any, s=70, marker=MRK[m], color=COL[m], edgecolor=SURF, linewidth=1.5,
                       label=LAB[m], zorder=3)
        ax.set_xlabel("Mean KS statistic (lower = more realistic)")
        ax.set_ylabel("PVR (%)")
        ax.set_title(f"({'ab'[i]}) {t}: admissibility vs. realism", loc="left")
        ax.set_ylim(-3, 103)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig_tradeoff")


def fig_forest():
    f = os.path.join(RES, "summary", "table_bootstrap.csv")
    if not os.path.exists(f):
        return
    B = pd.read_csv(f)
    order = ["chemogan_soft+none", "chemogan_soft+any", "chemogan+none", "chemogan+any", "gan_uncon+none", "gan_uncon+any",
             "gan_uncon_matched+none", "tvae+none", "tvae_prior+none", "ddpm+any"]
    B = B[B.scenario.isin(order)]
    tg = sorted(B.target.unique())
    fig, axes = plt.subplots(len(tg), 3, figsize=(14, 3.6 + 0.42 * len(order) * len(tg)), squeeze=False, sharex=True)
    for r, t in enumerate(tg):
        for c, w in enumerate(WELLS.values()):
            ax = axes[r, c]
            b = B[(B.target == t) & (B.well == w)]
            sc = [s for s in order if s in set(b.scenario)]
            for k, s in enumerate(sc[::-1]):
                row = b[b.scenario == s].iloc[0]
                m = s.split("+")[0]
                ax.errorbar(row.dR2, k, xerr=[[row.dR2 - row.lo], [row.hi - row.dR2]], fmt=MRK[m], color=COL[m],
                            ms=7, capsize=3, lw=1.8, mfc=COL[m] if s.endswith("any") else SURF, mew=1.8)
            ax.axvline(0, color=INK2, lw=1.2)
            ax.set_yticks(range(len(sc)))
            ax.set_yticklabels([LAB[s.split('+')[0]] + ("" if s.startswith("gan_uncon_matched") else (" + rejection" if s.endswith("any") else " (raw)")) for s in sc[::-1]]
                               if c == 0 else [])
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{t}: {w}", loc="left")
            if r == len(tg) - 1:
                ax.set_xlabel(r"$\Delta R^2$ vs. real only (mean over seeds, 95% CI)")
    fig.tight_layout()
    save(fig, "fig_forest")


def fig_profiles(target, well, scen="chemogan_soft+none"):
    tdir = os.path.join(RES, target, "pred")
    fs = sorted(glob.glob(os.path.join(tdir, f"{scen}_s*.npz")))
    fr = sorted(glob.glob(os.path.join(tdir, "real_s*.npz")))
    if not fs or not fr:
        return
    k = [w for w, v in WELLS.items() if v == well][0].replace(".las", "")
    zr = [np.load(f) for f in fr]; zs = [np.load(f) for f in fs]
    d, y = zr[0][k + "__depth"], zr[0][k + "__y"]
    pr = np.mean([z[k + "__yhat"] for z in zr], 0); ps = np.mean([z[k + "__yhat"] for z in zs], 0)
    # heterogeneity zones: rolling std of the measured target in the top quartile
    rs = pd.Series(y).rolling(40, center=True, min_periods=10).std().to_numpy()
    zone = rs >= np.nanpercentile(rs, 75)
    def r2(a, b): return 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 13), sharey=True)
    from matplotlib.patches import Patch
    for ax, p, name, col in ((axes[0], pr, "Real only", "#eb6834"), (axes[1], ps, "Real + ChemoGAN-P", COL["chemogan_soft"])):
        for z0, z1 in _runs(d, zone):
            ax.axhspan(z0, z1, color="#eda100", alpha=0.16, lw=0)
        ax.plot(y, d, color=INK, lw=1.0, label=f"Measured {target}")
        ax.plot(p, d, color=col, lw=1.6, alpha=0.9, label="Predicted")
        ax.set_title(f"({'ab'[0 if ax is axes[0] else 1]}) {name}\n$R^2$ = {r2(y, p):.3f}, MAE = {np.mean(np.abs(y - p)):.4f}", loc="left")
        ax.set_xlabel(f"{target} (dry weight fraction)")
        h, l = ax.get_legend_handles_labels()
        ax.legend(h + [Patch(color="#eda100", alpha=0.35)], l + ["High-heterogeneity zone"],
                  frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=10, ncol=1)
    axes[0].set_ylabel("Depth (m)"); axes[0].invert_yaxis()
    fig.suptitle(f"Blind well {well}: {target} (predictions averaged over seeds)", x=0.02, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, f"fig_profile_{target}_{well}")


def _runs(d, mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = d[i]
        if (not m or i == len(mask) - 1) and start is not None:
            out.append((start, d[i])); start = None
    return out


def fig_corr_spectrum():
    import config as cfg, pre_processamento as prep, gen_metrics as gm
    tg = [t for t in ("DWCA", "DWSI") if os.path.exists(os.path.join(RES, t, "synth_chemogan_s0.npz"))]
    if not tg:
        return
    fig, axes = plt.subplots(len(tg), 3, figsize=(14, 4.2 * len(tg)), squeeze=False)
    for r, t in enumerate(tg):
        d = prep.build_dataset(cfg, t, verbose=False, merge_val=True); F = d["features"]; Xr = d["X_train"]
        n = len(Xr)
        for c, (a, b, s) in enumerate(cfg.pairs_for(t)):
            ax = axes[r, c]
            bins = np.linspace(-1, 1, 41)
            rr = gm.window_corr_spectrum(Xr, F.index(a), F.index(b))
            ax.hist(rr[np.isfinite(rr)], bins=bins, density=True, color=GRID, edgecolor=INK2, lw=0.6, label="Real windows")
            for m in ("chemogan_soft", "chemogan", "gan_uncon", "tvae"):
                fz = os.path.join(RES, t, f"synth_{m}_s0.npz")
                if not os.path.exists(fz):
                    continue
                Xs = np.load(fz)["X"][:n]
                rs = gm.window_corr_spectrum(Xs, F.index(a), F.index(b))
                hgt, e = np.histogram(rs[np.isfinite(rs)], bins=bins, density=True)
                ax.step(e[:-1], hgt, where="post", color=COL[m], lw=2, label=LAB[m])
            ax.axvline(s * 0.5, color=INK2, ls="--", lw=1.2)
            ax.set_title(f"({'abcdef'[r * 3 + c]}) {t}: r({a}, {b})", loc="left")
            ax.set_xlabel("Window-wise Pearson r"); ax.set_ylabel("Density")
            if c == 0:
                ax.legend(frameon=False, fontsize=9.5, loc="upper center")
    fig.tight_layout()
    save(fig, "fig_corr_spectrum")


if __name__ == "__main__":
    for fn in (fig_convergence, fig_tradeoff, fig_forest, fig_corr_spectrum):
        try:
            fn()
        except Exception as e:  # keep going for partial results
            print(fn.__name__, "failed:", e)
    for t, w in (("DWSI", "7-SPH-3-SPS"), ("DWSI", "9-SPS-77A"), ("DWSI", "9-SPS-97"),
                 ("DWCA", "7-SPH-3-SPS"), ("DWCA", "9-SPS-77A"), ("DWCA", "9-SPS-97")):
        try:
            fig_profiles(t, w)
        except Exception as e:
            print("profile", t, w, "failed:", e)
    print("figures in", OUT)
