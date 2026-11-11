import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

# Suppress XGBoost UserWarnings, especially those related to early stopping.
warnings.filterwarnings("ignore", category=UserWarning)


def metricas(y_true, y_pred, prefix=""):
    """
    Compute and print standard regression metrics.

    Notes
    -----
    The function name is intentionally kept as `metricas` to preserve
    compatibility with the rest of the project.
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{prefix}MAE : {mae :.4f}\n{prefix}MSE : {mse :.5f}\n{prefix}R2  : {r2  :.4f}\n")

    return {"MAE": mae, "MSE": mse, "R2": r2}


def optimize_xgb(X_train, y_train):
    """
    Perform a randomized search for the best XGBoost hyperparameters.

    Parameters
    ----------
    X_train : array-like
        Scaled training features.
    y_train : array-like
        Training target values.

    Returns
    -------
    dict
        Best hyperparameters found by RandomizedSearchCV.
    """
    print("   -> Starting hyperparameter search (RandomizedSearchCV)...")

    base_model = xgb.XGBRegressor(
        booster="gbtree",
        random_state=42,
        eval_metric="rmse"
    )

    # Search space focused on reducing overfitting in complex geological data.
    param_dist = {
        "max_depth": randint(3, 20),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0.0, 0.5),
        "subsample": uniform(0.6, 0.4),         # Ranges from 0.6 to 1.0
        "colsample_bytree": uniform(0.6, 0.4),  # Ranges from 0.6 to 1.0
        "learning_rate": uniform(0.01, 0.1),
        "n_estimators": randint(10, 400),
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=150,  # Number of tested combinations; increase to refine the search.
        scoring="neg_mean_squared_error",
        cv=3,        # 3-fold cross-validation.
        verbose=0,
        n_jobs=-1,   # Use all available CPU cores.
        random_state=42,
    )

    search.fit(X_train, y_train)

    print(f"   -> Best parameters found: {search.best_params_}")
    return search.best_params_


def make_xgb(best_params=None):
    """
    Create an XGBoost regressor.

    If optimized parameters are provided, they are used. Otherwise, the original
    fallback parameters are used to preserve the previous behavior.
    """
    if best_params:
        return xgb.XGBRegressor(
            booster="gbtree",
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=10,
            **best_params,
        )

    # Safety fallback using the original fixed parameters.
    return xgb.XGBRegressor(
        booster="gbtree",
        max_depth=10,
        min_child_weight=3,
        gamma=0.002159770330420502,
        subsample=0.8368159879829131,
        colsample_bytree=0.9600160733401647,
        alpha=0.41285244400844556,
        learning_rate=0.04499101368613361,
        eval_metric="rmse",
        early_stopping_rounds=10,
        n_estimators=300,
        random_state=42,
    )


def scatter_real_vs_pred(y_train, y_pred_train, y_val, y_pred_val, title):
    """
    Plot real values versus predicted values for training and validation sets.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(np.ravel(y_train), np.ravel(y_pred_train), alpha=0.5, label="Train")
    plt.scatter(np.ravel(y_val), np.ravel(y_pred_val), alpha=0.5, label="Validation")

    mn = min(np.min(y_train), np.min(y_val), np.min(y_pred_train), np.min(y_pred_val))
    mx = max(np.max(y_train), np.max(y_val), np.max(y_pred_train), np.max(y_pred_val))

    plt.plot([mn, mx], [mn, mx], "k--", lw=2)
    plt.xlabel("Real values")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def _get_depth_col(well, preferred=("MD", "DEPTH", "Depth", "depth")):
    """
    Try to automatically find the depth column in a well DataFrame.

    Parameters
    ----------
    well : pandas.DataFrame
        Well-log DataFrame.
    preferred : tuple of str
        Candidate depth column names, in priority order.

    Returns
    -------
    str or None
        The first matching depth column, or None if no candidate is found.
    """
    cols = list(well.columns)

    for col in preferred:
        if col in cols:
            return col

    return None


def eval_wells(model, scaler2, pocos_cegos, logs, title_suffix=""):
    """
    Evaluate blind wells using metrics, scatter plots, and depth profiles.

    Notes
    -----
    The parameter name `pocos_cegos` is intentionally kept to preserve
    compatibility with the rest of the project.
    """
    print(f"Metrics applied to the blind test wells {title_suffix}")

    results = {}
    target_name = logs[-1]  # By convention, the last column is the target variable.

    for well_name, well in pocos_cegos.items():
        # 1) Find the depth column.
        depth_col = _get_depth_col(well)

        if depth_col is None:
            print(
                f"⚠️ {well_name}: depth column was not found (MD/DEPTH). "
                "Only scatter plots and metrics will be computed."
            )
        else:
            depth = well[depth_col].to_numpy().ravel()

        # 2) Prepare X/y using the same column order.
        df = well[logs].copy()
        X_well = df.iloc[:, :-1].to_numpy()
        y_well = df.iloc[:, -1].to_numpy().ravel()

        mask = np.isfinite(X_well).all(axis=1) & np.isfinite(y_well)

        if depth_col is not None:
            mask = mask & np.isfinite(depth)

        if mask.sum() < 5:
            print(f"⚠️ {well_name}: too few valid samples after filtering (n={mask.sum()}). Skipping.")
            continue

        X_ok = X_well[mask]
        y_ok = y_well[mask]

        if depth_col is not None:
            d_ok = depth[mask]

        # 3) Scale and predict.
        X_ok_scaled = scaler2.transform(X_ok)
        y_hat = model.predict(X_ok_scaled).ravel()

        print(f">> {well_name}")
        results[well_name] = metricas(y_ok, y_hat, prefix="  ")

        # 4) Real versus predicted scatter plot.
        plt.figure(figsize=(8, 6))
        plt.scatter(y_ok, y_hat, alpha=0.5, label="Real vs Predicted")

        mn = float(min(y_ok.min(), y_hat.min()))
        mx = float(max(y_ok.max(), y_hat.max()))

        plt.plot([mn, mx], [mn, mx], "k--", lw=2)
        plt.xlabel("Real values")
        plt.ylabel("Predictions")
        plt.title(f"Real vs Predicted — {well_name} {title_suffix}")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()

        # 5) Depth profile: real versus predicted.
        if depth_col is not None:
            order = np.argsort(d_ok)
            d2 = d_ok[order]
            y2 = y_ok[order]
            p2 = y_hat[order]

            plt.figure(figsize=(4, 15))
            plt.plot(y2, d2, label=f"{target_name} real")
            plt.plot(p2, d2, linestyle="--", label=f"{target_name} predicted")

            plt.gca().invert_yaxis()
            plt.xlabel(target_name)
            plt.ylabel(depth_col)
            plt.title(f"{well_name} {title_suffix}")
            plt.legend()
            plt.grid(True)
            plt.show()

    return results


def run_experiment(label, X, y, pocos_cegos, logs, plot_title_suffix="", optimize=True):
    """
    Run train/validation split, scaling, training, validation, and blind-well evaluation.

    Parameters
    ----------
    label : str
        Scenario name, for example "Real" or "Real + Synthetic".
    X : array-like
        Predictor variables.
    y : array-like
        Target variable.
    pocos_cegos : dict or None
        Blind wells used for external evaluation.
    logs : list of str or None
        Ordered list of log names. The last log is treated as the target.
    plot_title_suffix : str, optional
        Extra string appended to plot titles.
    optimize : bool, optional
        If True, automatically optimizes XGBoost hyperparameters.

    Returns
    -------
    dict
        Experiment results, including metrics, model, and scaler.
    """
    print(f"\n===== {label} =====")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )

    scaler2 = MinMaxScaler()
    X_train_scaled = scaler2.fit_transform(X_train)
    X_val_scaled = scaler2.transform(X_val)

    if optimize:
        best_params = optimize_xgb(X_train_scaled, y_train.ravel())
        model = make_xgb(best_params)

        # Early stopping is configured in the model constructor.
        model.fit(
            X_train_scaled,
            y_train.ravel(),
            eval_set=[(X_val_scaled, y_val.ravel())],
            verbose=False,
        )
    else:
        model = make_xgb()
        model.fit(
            X_train_scaled,
            y_train.ravel(),
            eval_set=[(X_val_scaled, y_val.ravel())],
            verbose=False,
        )

    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)

    print("Training metrics")
    train_metrics = metricas(y_train, y_pred_train)

    print("Validation metrics")
    val_metrics = metricas(y_val, y_pred_val)

    scatter_real_vs_pred(
        y_train,
        y_pred_train,
        y_val,
        y_pred_val,
        title=f"Real vs Predicted — {label} {plot_title_suffix}",
    )

    wells_metrics = None

    if (pocos_cegos is not None) and (logs is not None):
        wells_metrics = eval_wells(
            model,
            scaler2,
            pocos_cegos,
            logs,
            title_suffix=f"({label})",
        )

    return {
        "label": label,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "wells_metrics": wells_metrics,
        "model": model,
        "scaler": scaler2,
    }


# --- Tables and plots for blind wells ---
def resumo_wells_table(res_dicts):
    """
    Build a summary table with blind-well metrics for each scenario.

    The function name is kept as `resumo_wells_table` to preserve compatibility.
    """
    rows = []

    for res in res_dicts:
        label = res["label"]
        wells = res["wells_metrics"]

        if wells is None:
            continue

        for well_name, metrics in wells.items():
            rows.append({
                "Well": well_name,
                "Scenario": label,
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "R²": metrics["R2"],
            })

    return pd.DataFrame(rows)


def plot_r2_barras(df_wells):
    """
    Plot grouped bar charts comparing R² across wells and scenarios.

    The function name is kept as `plot_r2_barras` to preserve compatibility.
    """
    scenarios = list(df_wells["Scenario"].unique())
    wells = list(df_wells["Well"].unique())

    x = np.arange(len(wells))
    width = 0.8 / max(1, len(scenarios))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, scenario in enumerate(scenarios):
        r2_vals = df_wells[df_wells["Scenario"] == scenario].set_index("Well").loc[wells]["R²"]
        ax.bar(x + i * width, r2_vals, width, label=scenario)

    ax.set_xticks(x + (len(scenarios) - 1) * width / 2)
    ax.set_xticklabels(wells, rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Comparison of R² by well and scenario")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_metric_barras(df_wells, metric="MAE", titulo=None):
    """
    Plot grouped bar charts comparing a selected metric across wells and scenarios.

    Parameters
    ----------
    df_wells : pandas.DataFrame
        Blind-well summary table.
    metric : str
        Metric column to plot.
    titulo : str or None
        Optional custom title. Kept for backward compatibility with existing calls.
    """
    scenarios = list(df_wells["Scenario"].unique())
    wells = list(df_wells["Well"].unique())

    x = np.arange(len(wells))
    width = 0.8 / max(1, len(scenarios))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, scenario in enumerate(scenarios):
        vals = df_wells[df_wells["Scenario"] == scenario].set_index("Well").loc[wells][metric]
        ax.bar(x + i * width, vals, width, label=scenario)

    ax.set_xticks(x + (len(scenarios) - 1) * width / 2)
    ax.set_xticklabels(wells, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(titulo or f"Comparison of {metric} by well and scenario")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def resumo(res):
    """
    Return a compact tuple with train and validation MAE/R² values.

    The function name is kept as `resumo` to preserve compatibility.
    """
    return (
        res["train_metrics"]["MAE"],
        res["train_metrics"]["R2"],
        res["val_metrics"]["MAE"],
        res["val_metrics"]["R2"],
    )


def regressao_xbg(predictions_filtradas, pocos_concatenados, pocos_cegos, logs):
    """
    Main XGBoost regression function.

    Parameters
    ----------
    predictions_filtradas : numpy.ndarray
        Filtered synthetic predictions. They are expected to be already cleaned
        from hallucinated samples by rejection sampling.
    pocos_concatenados : numpy.ndarray
        Concatenated real well data. The last column is treated as the target.
    pocos_cegos : dict
        Blind wells used for external validation.
    logs : list of str
        Ordered list of logs. The last entry is treated as the target.

    Returns
    -------
    dict
        Summary metrics and the blind-well comparison table.

    Notes
    -----
    The function name is intentionally kept as `regressao_xbg` because it is
    called by the main experiment script.
    """
    # X and y for each scenario. The last column is the target variable.
    X_real, y_real = pocos_concatenados[:, :-1], pocos_concatenados[:, -1]
    X_synth, y_synth = predictions_filtradas[:, :-1], predictions_filtradas[:, -1]

    X_both = np.concatenate([X_synth, X_real], axis=0)
    y_both = np.concatenate([y_synth, y_real], axis=0)

    # =========================
    # 2) Run scenarios
    # =========================
    # Optimize XGBoost for both the original data and the combined dataset.
    res_real = run_experiment("Real", X_real, y_real, pocos_cegos, logs, optimize=True)
    res_both = run_experiment("Real + Synthetic", X_both, y_both, pocos_cegos, logs, optimize=True)

    # Build and display the blind-well comparison table.
    df_wells = resumo_wells_table([res_real, res_both])

    print("\n" + df_wells.sort_values(["Well", "Scenario"]).to_string(index=False))

    df_pivot = df_wells.pivot_table(
        index="Well",
        columns="Scenario",
        values=["MAE", "MSE", "R²"],
    )

    print("\n== Pivot by well (MAE/MSE/R² by scenario) ==")
    print(df_pivot)

    df_wells.to_csv("comparacao_pocos_cegos.csv", index=False)

    plot_r2_barras(df_wells)
    plot_metric_barras(df_wells, metric="MAE", titulo="MAE by well and scenario")
    plot_metric_barras(df_wells, metric="MSE", titulo="MSE by well and scenario")

    real_summary = resumo(res_real)
    both_summary = resumo(res_both)

    print("\n=== SUMMARY (MAE, R2) — train | validation ===")
    print(
        f"A) Real          : MAE_train={real_summary[0]:.4f}, R2_train={real_summary[1]:.3f} | "
        f"MAE_val={real_summary[2]:.4f}, R2_val={real_summary[3]:.3f}"
    )
    print(
        f"B) Real + Synth  : MAE_train={both_summary[0]:.4f}, R2_train={both_summary[1]:.3f} | "
        f"MAE_val={both_summary[2]:.4f}, R2_val={both_summary[3]:.3f}"
    )

    # Return the RMSE and R² from the "Real + Synthetic" scenario so they can be
    # used in the general summary table (df_summary), if desired.
    return {
        "mae": both_summary[2],
        "rmse": np.sqrt(both_summary[2]),  # Preserved original approximation.
        "r2": both_summary[3],
        "resultados": df_wells,
    }
