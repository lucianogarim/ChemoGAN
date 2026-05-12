import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.model_selection import train_test_split

# 1. Ensures that Windows can find the Conda DLLs
conda_prefix = sys.prefix
bin_path = os.path.join(conda_prefix, 'Library', 'bin')

if bin_path not in os.environ['PATH']:
    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']

# 2. Configures memory growth to avoid allocation errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU ready: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)

# ---------------------------------------------------------------
# Project module imports
import pre_processamento as process
import plots
import gan_lib
import xgb
from gan_trainer import ChemoGANTrainer
import transformer_trainer
import diffusion_trainer

# ====================================================
# 1) Data Configuration and Loading
# ====================================================
data_dir = "wells"

mnemonics = ['DEPTH', 'HCAL', 'DWAL', 'DWCA', 'DWFE', 'DWSI', 'RHOB', 'GR', 'NPHI', 'PE', 'DT',
             'NMRE_FINAL', 'NMRFL_FINAL', 'NMRT_FINAL', 'HFK', 'HTHO', 'HURA', 'T2LM']

depth_filters = {
    '7-SPH-3-SPS_BASE.las': (5037, 5450),
    '7-SPH-20D-SPS_BASE.las': (5050, 5277), '7-SPH-15D-SPS_BASE.las': (5430, 5722), '1-SPS-96-SP_BASE.las': (5037, 5483),
    '3-SPS-69-SP_BASE.las': (4964, 5175), '3-SPS-82A-SP_BASE.las': (5076, 5494), '7-SPH-1-SPS_BASE.las': (4995, 5357),
    '7-SPH-4D-SPS_BASE.las': (5150, 5400), '7-SPH-5-SPS_BASE.las': (5053, 5190),
    '7-SPH-6-SPS_BASE.las': (5000, 5368), '7-SPH-7D-SPS_BASE.las': (5220, 5487), '7-SPH-14D-SPS_BASE.las': (5193, 5460),
    '7-SPH-16D-SPS_BASE.las': (5145, 5285), '7-SPH-17-SPS_BASE.las': (5031, 5200), '7-SPH-22-SPS_BASE.las': (4980, 5030),
    '8-SPH-9-SPS_BASE.las': (5128, 5356), '8-SPH-11-SPS_BASE.las': (5091, 5280),
    '8-SPH-21D-SPS_BASE.las': (5117, 5496), '8-SPH-23-SPS_BASE.las': (5105, 5478), '9-SPS-77A-SP_BASE.las': (5023, 5339),
    '9-SPS-97-SP_BASE.las': (5138, 5190)
}

blind_well_list = ['7-SPH-3-SPS_BASE.las', '9-SPS-77A-SP_BASE.las', '9-SPS-97-SP_BASE.las']
features = ['GR', 'PE', 'DT', 'NMRE_FINAL', 'NMRFL_FINAL', 'NMRT_FINAL', 'HFK', 'HTHO', 'HURA', 'T2LM', 'RHOB', 'NPHI', 'DWSI']

n_steps = 40

# Processing pipeline
well_dict = process.carregamento(data_dir, mnemonics)
well_dict = process.filtro_profundidade(well_dict, depth_filters)
well_dict = process.trata_outliers(well_dict, features)
well_dict, blind_wells = process.escolhe_pocos_cegos(well_dict, blind_well_list)

# Concatenate only to train the scaler
concatenated_wells = process.concatena_pocos(well_dict)
_, scaler = process.muda_escala(concatenated_wells, features)

# Create sequences
X = process.cria_sequencias(well_dict, features, n_steps, scaler)

# Global parameters
T = X.shape[1]
A = X.shape[2]
Z = 100
BATCH_SIZE = 512


# ====================================================
# Auxiliary Metric Functions
# ====================================================
def dist_metrics_per_feature(real, synth, features):
    """
    Computes distribution similarity metrics between real and synthetic data.
    """
    results = []

    for i, feat in enumerate(features):
        r = real[:, i]
        s = synth[:, i]

        # Independent cleaning of NaNs and infinite values
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]

        if len(r) == 0 or len(s) == 0:
            continue

        # Distribution metrics
        wd = wasserstein_distance(r, s)
        ks_stat, _ = ks_2samp(r, s)

        mean_diff = abs(np.mean(r) - np.mean(s))
        std_diff = abs(np.std(r) - np.std(s))

        results.append({
            "Feature": feat,
            "wasserstein": wd,
            "ks_stat": ks_stat,
            "mean_diff": mean_diff,
            "std_diff": std_diff
        })

    return pd.DataFrame(results)


def corr_error(real, synth):
    """
    Computes the MAE between the correlation matrices.
    """
    c_real = np.corrcoef(real, rowvar=False)
    c_synth = np.corrcoef(synth, rowvar=False)

    c_real = np.nan_to_num(c_real, nan=0.0)
    c_synth = np.nan_to_num(c_synth, nan=0.0)

    diff = np.abs(c_real - c_synth)

    return np.mean(diff)


def calculate_pvr(synth_2d, features, n_steps, scaler=None):
    """
    Computes the Physics Violation Rate (PVR).
    A window (sample of size T) is considered invalid if:
    1. It violates the NMR hierarchy (NMRT >= NMRE >= NMRFL) at any point.
    2. It violates the structural correlations (RHOB vs NPHI < -0.7 or NPHI vs DT > 0.7).
    """

    # --- Global denormalization ---
    # If a scaler is provided, the data are converted back to the real physical space,
    # where the physical constraints are evaluated.
    if scaler is not None:
        # Create a copy to avoid changing the original array outside the function
        synth_eval = scaler.inverse_transform(synth_2d)
    else:
        synth_eval = synth_2d.copy()

    # Determine how many windows (N) exist in the flattened array
    N = len(synth_eval) // n_steps

    # --- 1. NMR violation (point-level -> window-level) ---
    try:
        i_rt = features.index('NMRT')  # Adjust the name if needed, e.g., NMRT_FINAL
        i_re = features.index('NMRE')
        i_fl = features.index('NMRFL')

        tol = 1e-4
        # The comparison is now performed on real porosity values, e.g., 0.15 <= 0.20
        viol_re_rt = synth_eval[:, i_re] > (synth_eval[:, i_rt] + tol)
        viol_fl_re = synth_eval[:, i_fl] > (synth_eval[:, i_re] + tol)

        # A point violates the rule if it breaks either of the two constraints
        viol_nmr_points = viol_re_rt | viol_fl_re

        # A window violates the rule if at least one point inside it is invalid
        viol_nmr_windows = viol_nmr_points.reshape(N, n_steps).any(axis=1)
    except ValueError:
        print("Warning: NMR curves not found.")
        viol_nmr_windows = np.zeros(N, dtype=bool)

    # --- 2. Correlation violation (window-level) ---
    # Correlations also work correctly on denormalized data.
    viol_corr_windows = np.zeros(N, dtype=bool)
    try:
        i_rhob = features.index('RHOB')
        i_nphi = features.index('NPHI')
        i_dt = features.index('DT')

        # Reshape to 3D format (N, T, A) to analyze Pearson correlation per window
        synth_3d = synth_eval.reshape(N, n_steps, -1)

        for j in range(N):
            rhob_win = synth_3d[j, :, i_rhob]
            nphi_win = synth_3d[j, :, i_nphi]
            dt_win = synth_3d[j, :, i_dt]

            # RHOB vs NPHI correlation. Physics requires it to be <= -0.7.
            r_rhob_nphi = np.corrcoef(rhob_win, nphi_win)[0, 1]
            if np.isnan(r_rhob_nphi) or r_rhob_nphi > -0.7:
                viol_corr_windows[j] = True

            # NPHI vs DT correlation. Physics requires it to be >= 0.7.
            r_nphi_dt = np.corrcoef(nphi_win, dt_win)[0, 1]
            if np.isnan(r_nphi_dt) or r_nphi_dt < 0.7:
                viol_corr_windows[j] = True

    except ValueError:
        print("Warning: Conventional curves (RHOB, NPHI, DT) not found.")

    # --- 3. Final PVR calculation ---
    # Union: if the window violated NMR OR correlation constraints, it is rejected (True)
    violation_mask = viol_nmr_windows | viol_corr_windows

    pvr_score = np.mean(violation_mask) * 100.0

    return pvr_score, violation_mask


# ====================================================
# Training and Generation Function
# ====================================================
def train_and_generate(MODE, X_train, X_val, T, A, Z, BATCH_SIZE, num_examples_to_generate, features2, scaler):
    """
    Returns: syn_2d (original scale), extras
    """
    extras = {}

    if MODE == "GAN":
        train_ds = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BATCH_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(X_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        gan_generator = gan_lib.make_generator_1d_constrained(T, A, Z)
        discriminator = gan_lib.make_discriminator_1d_robust(T, A)

        trainer = ChemoGANTrainer(gan_generator, discriminator, features, Z_dim=Z)
        history = trainer.fit(train_ds, val_ds, epochs=3000)

        extras["train_history"] = history

        # Call the training convergence plotting function
        # Make sure plots.py is imported at the top (import plots)
        plots.plot_gan_convergence(history, save_path="chemogan_convergence.png")

        seed = tf.random.normal([num_examples_to_generate, Z])
        pred_btA = gan_generator(seed, training=False).numpy()

    elif MODE == "TRANSFORMER":
        decoder_model = transformer_trainer.train_transformer(
            X_train, T, A, Z_dim=Z, epochs=3000, batch_size=BATCH_SIZE
        )

        seed = tf.random.normal([num_examples_to_generate, Z])
        pred_btA = decoder_model(seed, training=False).numpy()

    elif MODE == "DIFFUSION":
        diff_manager = diffusion_trainer.train_diffusion(
            X_train, T, A, epochs=3000, batch_size=BATCH_SIZE
        )

        pred_btA = diff_manager.generate(num_examples_to_generate, T, A)
        pred_btA = np.clip(pred_btA, 0.0, 1.0)  # Ensure limits

    else:
        raise ValueError("Invalid MODE")

    # Return to the original scale to compute geological metrics
    syn_2d = scaler.inverse_transform(pred_btA.reshape(-1, A))

    return syn_2d, extras


# ====================================================
# Main Execution Function
# ====================================================
def run_all_models(well_dict, concatenated_wells, blind_wells, X, T, A, Z, BATCH_SIZE, features, scaler, n_steps):
    # 1. Real data in original scale for statistical metrics
    real_2d = concatenated_wells[features].values

    # Split
    X_train, X_val = train_test_split(X, test_size=0.15, shuffle=False, random_state=42)

    num_examples_to_generate = round(concatenated_wells.shape[0] / n_steps)

    summary_rows = []

    details = {}

    for MODE in ["GAN", "TRANSFORMER", "DIFFUSION"]:
        print(f"\n==============================")
        print(f"Running MODE = {MODE}")
        print(f"==============================")

        # syn_2d is returned in the original scale
        syn_2d, extras = train_and_generate(
            MODE, X_train, X_val, T, A, Z, BATCH_SIZE,
            num_examples_to_generate, features, scaler
        )

        # (A) Feature-wise distances in the original scale
        df_feat = dist_metrics_per_feature(real_2d, syn_2d, features)
        details[MODE] = df_feat

        w_mean = float(df_feat["wasserstein"].mean())
        ks_mean = float(df_feat["ks_stat"].mean())

        # (B) Correlation in the original scale
        corr_mae = corr_error(real_2d, syn_2d)

        # (C) Physics Violation Rate (PVR) and rejection mask
        pvr_score, window_mask = calculate_pvr(syn_2d, features, n_steps=n_steps)

        # Expand the window mask to point-level format by repeating it for n_steps
        point_mask = np.repeat(window_mask, n_steps)

        # Physics-based rejection filter. Keeps only values where the mask is False.
        filtered_syn_2d = syn_2d[~point_mask]

        print(f"   -> [PVR] {pvr_score:.2f}% of windows violated the physics constraints.")
        print(f"   -> [Filter] Generated synthetic data: {len(syn_2d)} pts | Retained valid data: {len(filtered_syn_2d)} pts")

        # (D) XGBoost. Trains only with physically valid data.
        if len(filtered_syn_2d) > 0:
            xgb_metrics = xgb.regressao_xbg(filtered_syn_2d, real_2d, blind_wells, features)
        else:
            print("   -> [Warning] No valid samples remained for XGBoost training!")
            xgb_metrics = None

        if isinstance(xgb_metrics, dict):
            rmse = xgb_metrics.get("rmse", np.nan)
            mae = xgb_metrics.get("mae", np.nan)
            r2 = xgb_metrics.get("r2", np.nan)
            results = xgb_metrics.get("resultados", np.nan)
            results.to_excel(f"comparacao_pocos_cegos_{MODE}.xlsx", index=False)
        else:
            rmse = mae = r2 = np.nan

        summary_rows.append({
            "model": MODE,
            "wasserstein_mean": w_mean,
            "ks_mean": ks_mean,
            "corr_mae": corr_mae,
            "pvr_percent": pvr_score,
            "xgb_rmse": rmse,
            "xgb_mae": mae,
            "xgb_r2": r2
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("wasserstein_mean", ascending=True)
    return df_summary, details


# ---------------------------------------------------
# FINAL CALL
# ---------------------------------------------------
df_summary, details = run_all_models(
    well_dict=well_dict,
    concatenated_wells=concatenated_wells,
    blind_wells=blind_wells,
    X=X, T=T, A=A, Z=Z, BATCH_SIZE=BATCH_SIZE,
    features=features, scaler=scaler, n_steps=n_steps
)

print("\n===== FINAL SUMMARY (comparison of the 3 models) =====")
print(df_summary)
