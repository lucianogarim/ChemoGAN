"""Downstream regression (XGBoost) with leakage-safe validation (revision 1).

Changes w.r.t. the first submission (R1.5, R4.7, R2.27, R4.5):
* hyperparameters are tuned ONCE per target with GroupKFold over real training wells
  (no window/point from a validation fold well appears in its training fold) and are
  then frozen for every scenario (real-only and all augmented variants);
* no random point-level validation split and no early stopping on a mixed
  real+synthetic validation set;
* RMSE is computed as sqrt(MSE) (the first submission reported MSE in the RMSE column
  and returned sqrt(MAE) internally);
* per-well uncertainty via a paired moving-block bootstrap along depth.
"""
import json
import os
import numpy as np
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

SEARCH_SPACE = {
    "max_depth": randint(3, 20),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0.0, 0.5),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "learning_rate": uniform(0.01, 0.1),
    "n_estimators": randint(10, 400),
}


def stack_wells(wells, features):
    X, y, g = [], [], []
    for name, df in wells.items():
        v = df[features].to_numpy()
        X.append(v[:, :-1]); y.append(v[:, -1]); g += [name] * len(v)
    return np.vstack(X), np.concatenate(y), np.array(g)


def tune(X, y, groups, n_iter=60, n_splits=5, seed=42, cache=None):
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    search = RandomizedSearchCV(
        xgb.XGBRegressor(booster="gbtree", random_state=seed, n_jobs=2, tree_method="hist"),
        SEARCH_SPACE, n_iter=n_iter, scoring="neg_mean_squared_error",
        cv=GroupKFold(n_splits=n_splits), random_state=seed, n_jobs=1, verbose=0)
    search.fit(X, y, groups=groups)
    best = {k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in search.best_params_.items()}
    best["cv_rmse"] = float(np.sqrt(-search.best_score_))
    if cache:
        json.dump(best, open(cache, "w"), indent=2)
    return best


def make_model(params, seed):
    p = {k: v for k, v in params.items() if k != "cv_rmse"}
    p["max_depth"] = int(p["max_depth"]); p["min_child_weight"] = int(p["min_child_weight"])
    p["n_estimators"] = int(p["n_estimators"])
    return xgb.XGBRegressor(booster="gbtree", random_state=seed, n_jobs=2, tree_method="hist", **p)


def metrics(y, yh):
    return {"MAE": float(mean_absolute_error(y, yh)),
            "RMSE": float(np.sqrt(mean_squared_error(y, yh))),
            "R2": float(r2_score(y, yh))}


def fit_predict(params, X_train, y_train, eval_wells, features, seed):
    """Fit once, predict every evaluation well. Returns {well: (depth, y, yhat)}."""
    m = make_model(params, seed)
    m.fit(X_train, y_train)
    out = {}
    for name, df in eval_wells.items():
        d = df.sort_values("DEPTH")
        v = d[features].to_numpy()
        out[name] = (d["DEPTH"].to_numpy(), v[:, -1], m.predict(v[:, :-1]))
    return out


def block_bootstrap_delta(y, yh_a, yh_b, block=40, B=2000, seed=0):
    """Paired moving-block bootstrap of metric(b) - metric(a) along depth."""
    rng = np.random.default_rng(seed)
    n = len(y)
    nb = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    d = {"R2": [], "MAE": [], "RMSE": []}
    for _ in range(B):
        st = rng.integers(0, starts_max, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        ma, mb = metrics(y[idx], yh_a[idx]), metrics(y[idx], yh_b[idx])
        for k in d:
            d[k].append(mb[k] - ma[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in d.items()}
