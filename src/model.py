"""Predictive modeling of crop yield (tonnes/hectare) for the India crop production dataset.

Input : CSV with columns State_Name, District_Name, Crop_Year, Season, Crop, Area, Production
Output: outputs/results.json, outputs/figures/*.png, outputs/tables/*.csv

Usage : python src/model.py --input data/raw/crop_production.csv
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
SEED = 42
REQUIRED = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"]
CAT = ["State_Name", "Crop", "Season"]
NUM_BASE = ["Crop_Year", "log_area"]
NUM_HIST = ["yield_lag1", "yield_prior_mean"]
KEY = ["State_Name", "District_Name", "Crop", "Season"]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- data
def load_and_clean(path):
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise SystemExit(f"Input is missing expected columns: {missing}\nFound: {list(raw.columns)}")
    log = {"rows_raw": int(len(raw))}
    df = raw[REQUIRED].copy()
    for c in ["State_Name", "District_Name", "Season", "Crop"]:
        df[c] = df[c].astype("string").str.strip()
    df = df.drop_duplicates().dropna()
    df = df[df["Area"] > 0]
    df = df[~df["Crop"].str.contains("coconut", case=False, na=False)].copy()  # coconut is in nuts, not tonnes
    df["Crop_Year"] = df["Crop_Year"].astype(int)
    df["Yield"] = df["Production"] / df["Area"]
    df = df[df["Yield"] > 0]
    log["rows_after_basic_cleaning"] = int(len(df))
    # Remove extreme yield outliers (3 x IQR on log-yield within each crop): likely unit / entry errors
    ly = np.log(df["Yield"])
    q = ly.groupby(df["Crop"]).quantile([0.25, 0.75]).unstack()
    iqr = q[0.75] - q[0.25]
    lo = df["Crop"].map(q[0.25] - 3 * iqr)
    hi = df["Crop"].map(q[0.75] + 3 * iqr)
    keep = (ly >= lo) & (ly <= hi)
    log["extreme_outliers_removed"] = int((~keep).sum())
    df = df[keep].copy()
    log["rows_final"] = int(len(df))
    return df.reset_index(drop=True), log


def build_features(df):
    """Target = log(1 + yield). History features only use years BEFORE the row's year (no leakage)."""
    df = df.copy()
    df["y"] = np.log1p(df["Yield"])
    df["log_area"] = np.log1p(df["Area"])
    gy = df.groupby(KEY + ["Crop_Year"], as_index=False).agg(ysum=("y", "sum"), ycnt=("y", "count"), ymean=("y", "mean"))
    gy = gy.sort_values(KEY + ["Crop_Year"])
    g = gy.groupby(KEY)
    gy["yield_prior_mean"] = g["ysum"].cumsum().shift(1) / g["ycnt"].cumsum().shift(1)
    first = ~gy.duplicated(KEY)                       # first year of every series has no history
    gy.loc[first, "yield_prior_mean"] = np.nan
    lag = gy[KEY + ["Crop_Year", "ymean"]].copy()
    lag["Crop_Year"] += 1                             # value of year t-1 becomes feature of year t
    lag = lag.rename(columns={"ymean": "yield_lag1"})
    gy = gy.merge(lag, on=KEY + ["Crop_Year"], how="left")
    df = df.merge(gy[KEY + ["Crop_Year", "yield_lag1", "yield_prior_mean"]].drop_duplicates(KEY + ["Crop_Year"]),
                  on=KEY + ["Crop_Year"], how="left")
    for c in CAT:
        df[c] = df[c].astype(object)
    return df


# ----------------------------------------------------------------------------- models
def prep_linear(nums):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), nums)])


def prep_tree(nums, impute):
    num_tf = SimpleImputer(strategy="median", add_indicator=True) if impute else "passthrough"
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT),
        ("num", num_tf, nums)])


def hgb_pipe(nums, **params):
    base = dict(categorical_features=[0, 1, 2], random_state=SEED)
    base.update(params)
    return Pipeline([("prep", prep_tree(nums, impute=False)), ("m", HistGradientBoostingRegressor(**base))])


def metrics(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(np.clip(p_log, 0, None))
    rel = np.abs(p - y) / np.maximum(y, 1e-9)
    return {"log_mae": float(mean_absolute_error(y_log, p_log)),
            "log_rmse": float(np.sqrt(mean_squared_error(y_log, p_log))),
            "log_r2": float(r2_score(y_log, p_log)),
            "mae_tha": float(mean_absolute_error(y, p)),
            "rmse_tha": float(np.sqrt(mean_squared_error(y, p))),
            "r2_raw": float(r2_score(y, p)),
            "mdape_pct": float(100 * np.median(rel)),
            "within25_pct": float(100 * np.mean(rel <= 0.25)),
            "bias_log": float(np.mean(p_log - y_log))}


def main(args):
    t0 = time.time()
    out = Path(args.outdir)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    df, clean_log = load_and_clean(args.input)
    R = {"input_file": Path(args.input).name, "cleaning": clean_log}

    # ---- keep fully reported years only, then split by time
    counts = df.groupby("Crop_Year").size()
    valid = sorted(counts[counts >= 0.5 * counts.median()].index.tolist())
    dropped = sorted(set(counts.index) - set(valid))
    df = build_features(df[df["Crop_Year"].isin(valid)])
    n_test_years = 3
    test_years, train_years = valid[-n_test_years:], valid[:-n_test_years]
    train, test = df[df["Crop_Year"].isin(train_years)].copy(), df[df["Crop_Year"].isin(test_years)].copy()
    R["data"] = {"rows_model": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
                 "train_years": [int(train_years[0]), int(train_years[-1])],
                 "test_years": [int(y) for y in test_years], "dropped_partial_years": [int(y) for y in dropped],
                 "history_coverage_test_pct": round(float(100 * test["yield_lag1"].notna().mean()), 1),
                 "crops": int(df["Crop"].nunique()), "states": int(df["State_Name"].nunique())}
    nums_full = NUM_BASE + NUM_HIST
    R["features"] = {"categorical": CAT, "numeric_base": NUM_BASE, "history": NUM_HIST}
    y_tr, y_te = train["y"].values, test["y"].values
    X_tr, X_te = train[CAT + nums_full], test[CAT + nums_full]

    results = {}
    preds = {}

    # ---- 1. baseline: median log-yield of the crop in the training years
    med = train.groupby("Crop")["y"].median()
    p_base = test["Crop"].map(med).fillna(train["y"].median()).values
    results["Baseline (crop median)"] = metrics(y_te, p_base)
    preds["Baseline (crop median)"] = p_base

    # ---- 2. ridge regression
    ridge = Pipeline([("prep", prep_linear(nums_full)), ("m", Ridge(alpha=1.0))]).fit(X_tr, y_tr)
    preds["Ridge regression"] = ridge.predict(X_te)
    results["Ridge regression"] = metrics(y_te, preds["Ridge regression"])

    # ---- 3. random forest
    rf = Pipeline([("prep", prep_tree(nums_full, impute=True)),
                   ("m", RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_samples=0.5,
                                               n_jobs=-1, random_state=SEED))]).fit(X_tr, y_tr)
    preds["Random forest"] = rf.predict(X_te)
    results["Random forest"] = metrics(y_te, preds["Random forest"])

    # ---- 4. gradient boosting default
    hgb = hgb_pipe(nums_full).fit(X_tr, y_tr)
    preds["Gradient boosting (default)"] = hgb.predict(X_te)
    results["Gradient boosting (default)"] = metrics(y_te, preds["Gradient boosting (default)"])

    # ---- 5. gradient boosting tuned (random search, 3-fold CV on a training subsample)
    sub = train.sample(min(len(train), 60000), random_state=SEED)
    search = RandomizedSearchCV(
        hgb_pipe(nums_full),
        {"m__learning_rate": loguniform(0.03, 0.3), "m__max_leaf_nodes": [15, 31, 63, 127],
         "m__min_samples_leaf": [20, 50, 100], "m__l2_regularization": [0.0, 0.1, 1.0],
         "m__max_iter": [200, 400]},
        n_iter=10, cv=KFold(3, shuffle=True, random_state=SEED), scoring="neg_mean_absolute_error",
        random_state=SEED, n_jobs=1)
    search.fit(sub[CAT + nums_full], sub["y"].values)
    best = {k.replace("m__", ""): (float(v) if isinstance(v, (np.floating, float)) else int(v))
            for k, v in search.best_params_.items()}
    tuned = hgb_pipe(nums_full, **best).fit(X_tr, y_tr)
    name_t = "Gradient boosting (tuned)"
    preds[name_t] = tuned.predict(X_te)
    results[name_t] = metrics(y_te, preds[name_t])
    R["tuning"] = {"best_params": best, "cv_mae_log_subsample": round(float(-search.best_score_), 4),
                   "n_iter": 10, "cv_folds": 3, "tuning_rows": int(len(sub))}
    cv_r2 = cross_val_score(hgb_pipe(nums_full, **best), sub[CAT + nums_full], sub["y"].values,
                            cv=KFold(3, shuffle=True, random_state=SEED), scoring="r2")
    R["tuning"]["cv_r2_mean"] = round(float(cv_r2.mean()), 3)
    R["tuning"]["cv_r2_std"] = round(float(cv_r2.std()), 3)

    # ---- 6. ablation: tuned model WITHOUT history features
    nums_nohist = NUM_BASE
    abl = hgb_pipe(nums_nohist, **best).fit(train[CAT + nums_nohist], y_tr)
    name_a = "Gradient boosting (no history features)"
    preds[name_a] = abl.predict(test[CAT + nums_nohist])
    results[name_a] = metrics(y_te, preds[name_a])

    # ---- overfitting check
    R["train_vs_test"] = {"train_log_r2": round(float(r2_score(y_tr, tuned.predict(X_tr))), 4),
                          "test_log_r2": round(results[name_t]["log_r2"], 4)}

    order = ["Baseline (crop median)", "Ridge regression", "Random forest", "Gradient boosting (default)", name_t, name_a]
    R["models"] = [{"model": m, **{k: round(v, 4) for k, v in results[m].items()}} for m in order]
    pd.DataFrame(R["models"]).to_csv(tabdir / "model_comparison.csv", index=False)
    best_model = max(order[:5], key=lambda m: results[m]["log_r2"])
    R["best_model"] = best_model
    final_pred = preds[best_model]
    final_pipe = {"Ridge regression": ridge, "Random forest": rf, "Gradient boosting (default)": hgb,
                  name_t: tuned}.get(best_model)

    # ---- Fig 1: model comparison
    cmp = pd.DataFrame(R["models"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=cmp, y="model", x="log_r2", ax=axes[0], color="#2d6a4f")
    axes[0].set(title="R-squared on test years (log yield) - higher is better", xlabel="R2", ylabel="")
    sns.barplot(data=cmp, y="model", x="mdape_pct", ax=axes[1], color="#c0803a")
    axes[1].set(title="Median absolute % error - lower is better", xlabel="%", ylabel="", yticklabels=[])
    save(fig, figdir / "01_model_comparison.png")

    # ---- Fig 2: actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hb = ax.hexbin(y_te, final_pred, gridsize=60, bins="log", cmap="YlGn", mincnt=1)
    lim = [min(y_te.min(), final_pred.min()), max(y_te.max(), final_pred.max())]
    ax.plot(lim, lim, "--", color="#b23a48", label="perfect prediction")
    ax.set(title=f"Actual vs predicted ({best_model})", xlabel="Actual log(1+yield)", ylabel="Predicted log(1+yield)")
    ax.legend()
    fig.colorbar(hb, ax=ax, label="records (log)")
    save(fig, figdir / "02_actual_vs_predicted.png")

    # ---- Fig 3: residuals
    resid = final_pred - y_te
    R["residuals"] = {"mean": round(float(resid.mean()), 4), "std": round(float(resid.std()), 4)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.histplot(resid, bins=70, ax=axes[0], color="#4c9f70")
    axes[0].axvline(0, color="#b23a48", ls="--")
    axes[0].set(title="Residual distribution (predicted - actual, log scale)", xlabel="Residual")
    axes[1].scatter(final_pred[::10], resid[::10], s=3, alpha=0.3, color="#2d6a4f")
    axes[1].axhline(0, color="#b23a48", ls="--")
    axes[1].set(title="Residuals vs predicted value", xlabel="Predicted log(1+yield)", ylabel="Residual")
    save(fig, figdir / "03_residuals.png")

    # ---- Permutation importance (on the best model, test sample)
    if final_pipe is not None:
        cols = CAT + nums_full
        ts = test.sample(min(len(test), 15000), random_state=SEED)
        pi = permutation_importance(final_pipe, ts[cols], ts["y"].values, n_repeats=3, random_state=SEED,
                                    scoring="neg_mean_absolute_error", n_jobs=1)
        imp = pd.DataFrame({"feature": cols, "importance": pi.importances_mean, "std": pi.importances_std}
                           ).sort_values("importance", ascending=False)
        imp.to_csv(tabdir / "permutation_importance.csv", index=False)
        R["importance"] = imp.round(4).to_dict("records")
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        sns.barplot(data=imp, y="feature", x="importance", ax=ax, color="#2d6a4f")
        ax.set(title="Permutation importance (increase in log-MAE when shuffled)", xlabel="Increase in error", ylabel="")
        save(fig, figdir / "04_feature_importance.png")

    # ---- Error by crop
    tt = test.assign(pred=np.expm1(np.clip(final_pred, 0, None)), actual=np.expm1(y_te))
    tt["rel"] = np.abs(tt["pred"] - tt["actual"]) / tt["actual"]
    tt["abs"] = np.abs(tt["pred"] - tt["actual"])
    crop_err = (tt.groupby("Crop").agg(n=("rel", "size"), mdape_pct=("rel", lambda s: 100 * s.median()),
                                       mae_tha=("abs", "mean"), mean_yield=("actual", "mean"))
                .query("n >= 100").sort_values("n", ascending=False).head(12))
    crop_err.round(3).to_csv(tabdir / "error_by_crop.csv")
    R["crop_errors"] = crop_err.round(3).reset_index().rename(columns={"Crop": "crop"}).to_dict("records")
    if len(crop_err):
        fig, ax = plt.subplots(figsize=(8, 5))
        ce = crop_err.sort_values("mdape_pct")
        sns.barplot(x=ce["mdape_pct"], y=ce.index, ax=ax, color="#c0803a")
        ax.set(title="Median absolute % error by crop (test years)", xlabel="%", ylabel="")
        save(fig, figdir / "05_error_by_crop.png")

    with open(out / "results.json", "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"Done in {time.time() - t0:.0f}s. Best model: {best_model} "
          f"(R2 log = {results[best_model]['log_r2']:.3f}, median error = {results[best_model]['mdape_pct']:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/crop_production.csv")
    ap.add_argument("--outdir", default="outputs")
    main(ap.parse_args())
