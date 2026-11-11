"""Generate the LaTeX tables of the revised manuscript directly from results/summary/*.csv.
Usage: python make_tables.py [results_dir] [out_dir]
"""
import json, os, sys
import numpy as np
import pandas as pd

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "tex")
os.makedirs(OUT, exist_ok=True)
S = os.path.join(RES, "summary")
G = pd.read_csv(os.path.join(S, "gen_all.csv"))
B = pd.read_csv(os.path.join(S, "blind_all.csv"))
BT = pd.read_csv(os.path.join(S, "table_bootstrap.csv"))
PL = pd.read_csv(os.path.join(S, "table_pooled_test.csv"))
WELLS = {"7-SPH-3-SPS_BASE.las": "7-SPH-3-SPS", "9-SPS-77A-SP_BASE.las": "9-SPS-77A", "9-SPS-97-SP_BASE.las": "9-SPS-97"}
LAB = {"chemogan": "ChemoGAN-H", "gan_uncon": "Unconstrained GAN", "gan_uncon_matched": "Unconstr.\\ GAN, matched rejection",
       "chemogan_soft": "ChemoGAN-P", "tvae": "Transformer-VAE (GMM)", "tvae_prior": "Transformer-VAE (prior)",
       "ddpm": "DDPM"}
ORDER = ["chemogan_soft", "chemogan", "gan_uncon", "gan_uncon_matched", "tvae", "tvae_prior", "ddpm"]


def pm(x, d):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "--"
    if len(x) == 1:
        return f"{x[0]:.{d}f}"
    return f"{x.mean():.{d}f} $\\pm$ {x.std(ddof=1):.{d}f}"


# ---------------- Table: generative quality ----------------
L = [r"\begin{tabular}{llcccccccc}", r"\hline",
     r"\textbf{Target} & \textbf{Model} & \textbf{n} & \textbf{PVR (\%)} & \textbf{Hard (\%)} & \textbf{Soft (\%)} & \textbf{PE pair (\%)} & \textbf{$w_{mean}$} & \textbf{$ks_{mean}$} & \textbf{$corr_{mae}$} \\ \hline"]
for t in sorted(G.target.unique()):
    ref = json.load(open(os.path.join(RES, t, "reference.json")))["real_reference_pvr"]
    pe = [v for k, v in ref.items() if k.startswith("diag")][0]
    L.append(f"{t} & \\textit{{Real windows (reference)}} & -- & {ref['any']:.1f} & {ref['nmr']:.1f} & {ref['soft']:.1f} & {pe:.1f} & -- & -- & -- \\\\")
    for m in ORDER:
        g = G[(G.target == t) & (G.model == m)]
        if g.empty:
            continue
        dc = f"PVR_diag_corr_PE_{t}"
        L.append(f" & {LAB[m]} & {len(g)} & {pm(g.PVR_any, 1)} & {pm(g.PVR_nmr, 1)} & {pm(g.PVR_soft, 1)} & {pm(g[dc], 1)} & "
                 f"{pm(g.W1_mean, 4)} & {pm(g.KS_mean, 3)} & {pm(g.corr_MAE, 3)} \\\\")
    L.append(r"\hline")
L.append(r"\end{tabular}")
open(os.path.join(OUT, "tab_generative.tex"), "w").write("\n".join(x for x in L if x))

# ---------------- Table: downstream R2 per well ----------------
SC = [("real", "Real only"), ("chemogan_soft+none", "+ ChemoGAN-P, raw"), ("chemogan_soft+any", "+ ChemoGAN-P, filtered"),
      ("chemogan+none", "+ ChemoGAN-H, raw"), ("chemogan+any", "+ ChemoGAN-H, filtered"),
      ("gan_uncon+none", "+ Unconstr.\\ GAN, raw"), ("gan_uncon+any", "+ Unconstr.\\ GAN, filtered"),
      ("gan_uncon_matched+none", "+ Unconstr.\\ GAN, matched rejection"),
      ("tvae+none", "+ T-VAE (GMM), raw"), ("tvae+any", "+ T-VAE (GMM), filtered"), ("tvae_prior+none", "+ T-VAE (prior), raw"),
      ("ddpm+none", "+ DDPM, raw"), ("ddpm+any", "+ DDPM, filtered")]
L = [r"\begin{tabular}{llccccc}", r"\hline",
     r"\textbf{Target} & \textbf{Training set} & \textbf{n} & \textbf{7-SPH-3-SPS} & \textbf{9-SPS-77A} & \textbf{9-SPS-97} & \textbf{Pooled $\Delta R^2$ [$p$]} \\ \hline"]
for t in sorted(B.target.unique()):
    b = B[B.target == t]
    base = b[b.scenario == "real"].set_index(["seed", "well"]).R2
    first = True
    for sc, lab in SC:
        bs = b[b.scenario == sc]
        if bs.empty:
            continue
        cells = []
        for w in WELLS:
            x = bs[bs.well == w]
            cell = f"{x.R2.mean():.3f}"
            if sc != "real":
                bt = BT[(BT.target == t) & (BT.scenario == sc) & (BT.well == WELLS[w])]
                if not bt.empty:
                    r = bt.iloc[0]
                    star = "$^{*}$" if (r.lo > 0 or r.hi < 0) else ""
                    cell += f" ({r.dR2:+.3f}){star}"
            cells.append(cell)
        if sc == "real":
            last = "--"
        else:
            p = PL[(PL.target == t) & (PL.scenario == sc)]
            pv = p.iloc[0].wilcoxon_p if not p.empty else None
            pstr = "--" if pv is None else ("$<$0.001" if pv < 0.001 else f"{pv:.3f}")
            last = f"{p.iloc[0].dR2_mean:+.3f} [{pstr}]" if not p.empty else "--"
        L.append(f"{t if first else ''} & {lab} & {bs.seed.nunique()} & " + " & ".join(cells) + f" & {last} \\\\")
        first = False
    L.append(r"\hline")
L.append(r"\end{tabular}")
open(os.path.join(OUT, "tab_downstream.tex"), "w").write("\n".join(L))

# ---------------- Table: MAE / RMSE (appendix) ----------------
L = [r"\begin{tabular}{lllccc}", r"\hline",
     r"\textbf{Target} & \textbf{Well} & \textbf{Training set} & \textbf{MAE} & \textbf{RMSE} & \textbf{$R^2$} \\ \hline"]
for t in sorted(B.target.unique()):
    for w, wl in WELLS.items():
        first = True
        for sc, lab in [x for x in SC if x[0] in ("real", "chemogan_soft+none", "chemogan+none", "gan_uncon+none", "tvae+none", "ddpm+none")]:
            x = B[(B.target == t) & (B.well == w) & (B.scenario == sc)]
            if x.empty:
                continue
            L.append(f"{t if first and w == list(WELLS)[0] else ''} & {wl if first else ''} & {lab} & {pm(x.MAE, 4)} & {pm(x.RMSE, 4)} & {pm(x.R2, 3)} \\\\")
            first = False
        L.append(r"\hline")
L.append(r"\end{tabular}")
open(os.path.join(OUT, "tab_errors_appendix.tex"), "w").write("\n".join(L))

# ---------------- Table: sensitivity (ratio, TSTR, ensemble) ----------------
EN = pd.read_csv(os.path.join(S, "table_ensemble.csv"))
L = [r"\begin{tabular}{llccc}", r"\hline", r"\textbf{Target} & \textbf{Setting (ChemoGAN-P, raw)} & \textbf{7-SPH-3-SPS} & \textbf{9-SPS-77A} & \textbf{9-SPS-97} \\ \hline"]
for t in sorted(B.target.unique()):
    rows = [("chemogan_soft+none@r0.5", "Synthetic:real = 0.5, mean $\\Delta R^2$"), ("chemogan_soft+none", "Synthetic:real = 1, mean $\\Delta R^2$"),
            ("chemogan_soft+none@r2", "Synthetic:real = 2, mean $\\Delta R^2$")]
    first = True
    D = pd.read_csv(os.path.join(S, "table_downstream_long.csv"))
    for sc, lab in rows:
        x = D[(D.target == t) & (D.scenario == sc)]
        if x.empty:
            continue
        L.append(f"{t if first else ''} & {lab} & " + " & ".join(f"{x[x.well == w].dR2_mean.iloc[0]:+.3f}" for w in WELLS.values()) + " \\\\")
        first = False
    x = D[(D.target == t) & (D.scenario == "chemogan_soft+none+TSTR")]
    xr = D[(D.target == t) & (D.scenario == "real")]
    if not x.empty:
        L.append(f" & TSTR (synthetic only), $R^2$ & " + " & ".join(f"{x[x.well == w].R2_mean.iloc[0]:.3f}" for w in WELLS.values()) + " \\\\")
        L.append(f" & Real only, $R^2$ & " + " & ".join(f"{xr[xr.well == w].R2_mean.iloc[0]:.3f}" for w in WELLS.values()) + " \\\\")
    er = EN[(EN.target == t) & (EN.scenario == "real")]; ea = EN[(EN.target == t) & (EN.scenario == "chemogan_soft+none")]
    if not er.empty and not ea.empty:
        L.append(f" & Seed ensemble, real only, $R^2$ & " + " & ".join(f"{er[er.well == w].R2_ensemble.iloc[0]:.3f}" for w in WELLS.values()) + " \\\\")
        L.append(f" & Seed ensemble, augmented, $R^2$ & " + " & ".join(f"{ea[ea.well == w].R2_ensemble.iloc[0]:.3f}" for w in WELLS.values()) + " \\\\")
    L.append(r"\hline")
L.append(r"\end{tabular}")
open(os.path.join(OUT, "tab_sensitivity.tex"), "w").write("\n".join(L))

# ---------------- Table: cost ----------------
L = [r"\begin{tabular}{lcc}", r"\hline", r"\textbf{Model} & \textbf{DWCA (min)} & \textbf{DWSI (min)} \\ \hline"]
for m in ORDER:
    if m == "tvae_prior":
        continue
    c = [pm(G[(G.target == t) & (G.model == m)].train_sec / 60, 1) for t in ("DWCA", "DWSI")]
    if all(x == "--" for x in c):
        continue
    L.append(f"{LAB[m]} & {c[0]} & {c[1]} \\\\")
L += [r"\hline", r"\end{tabular}"]
open(os.path.join(OUT, "tab_cost.tex"), "w").write("\n".join(L))
print("tables written to", OUT)

# ---------------- Table: diagnostics ----------------
DG = os.path.join(S, "diagnostics_chemogan_soft_none.csv")
if os.path.exists(DG):
    D = pd.read_csv(DG)
    L = [r"\begin{tabular}{llcccccc}", r"\hline", r"\textbf{Target} & \textbf{Well} & \textbf{Mean $\pm$ sd} & \textbf{Outside 1--99\%} & \textbf{10-NN dist.} & \textbf{Coverage gain} & \textbf{$\Delta$MAE hetero} & \textbf{$\Delta$MAE other} \\ \hline"]
    for t in ["DWCA", "DWSI"]:
        for i, (_, r) in enumerate(D[D.target == t].iterrows()):
            L.append(f"{t if i == 0 else ''} & {r.well} & {r.target_mean:.3f} $\\pm$ {r.target_std:.3f} & {100 * r.frac_outside_train_1_99:.1f}\\% & "
                     f"{r['kNN_dist_real']:.3f} & {r['coverage_gain_%']:.1f}\\% & {r.dMAE_hetero:+.4f} & {r.dMAE_other:+.4f} \\\\")
        L.append(r"\hline")
    L.append(r"\end{tabular}")
    open(os.path.join(OUT, "tab_diagnostics.tex"), "w").write("\n".join(L))
