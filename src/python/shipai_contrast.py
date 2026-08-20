import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Config
# --------------------------
OUTPUT_ROOT = Path("../../results/simulation")
PATTERNS = ["sp-static_*", "sp-density-60_*", "sp-risk-60_*", "shipai_*"]
LAYER_IDS = [7, 8, 9]  # selected_layer_ids
RESULTS_DIR = Path("../../results/analysis/shipai_contrast")
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def scenario_from_project_name(name: str) -> str:
    if name.startswith("sp-static_"):
        return "static_field"
    if name.startswith("sp-density-60_"):
        return "density_field"
    if name.startswith("sp-risk-60_"):
        return "hazard+density"

    # Legacy names like shipai_9.
    idx = int(name.split("_")[-1])
    if 1 <= idx <= 20:
        return "static_field"
    elif 21 <= idx <= 40:
        return "density_field"
    elif 41 <= idx <= 60:
        return "hazard+density"
    else:
        return "other"

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def flatten_one(metrics: dict) -> dict:
    proj = metrics.get("project", "unknown")
    row = {"project": proj, "scenario": scenario_from_project_name(proj)}

    # ---- time_efficiency
    te = metrics.get("time_efficiency", {})
    for k in ["T_0.90", "T_0.95", "T_0.99", "alive_start", "alive_end", "t_start", "t_end", "total_n"]:
        row[k] = te.get(k, np.nan)

    # ---- spatial_efficiency
    se = metrics.get("spatial_efficiency", {})
    layers = se.get("layers", [])
    # layers list order corresponds to selected_layer_ids; we map by file name or by order
    # We'll try to infer layer id from file "layer_7.bin"
    for layer in layers:
        file_name = layer.get("file", "")
        layer_id = None
        if "layer_" in file_name and file_name.endswith(".bin"):
            try:
                layer_id = int(file_name.replace("layer_", "").replace(".bin", ""))
            except Exception:
                layer_id = None

        if layer_id is None:
            continue
        if layer_id not in LAYER_IDS:
            continue

        prefix = f"layer{layer_id}_"
        keep = [
            "mean", "std", "min", "max", "q90", "q95", "q99",
            "count_ge_max_mul_90", "count_ge_max_mul_95", "count_ge_max_mul_99",
            "freq_ge_max_mul_90", "freq_ge_max_mul_95", "freq_ge_max_mul_99",
            "valid_n", "zero_n_total"
        ]
        for k in keep:
            row[prefix + k] = layer.get(k, np.nan)

    return row

def bootstrap_ci(x: np.ndarray, n=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    boots = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return (lo, hi)

def summarize_group(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for scen, g in df.groupby("scenario"):
        for c in cols:
            arr = g[c].to_numpy(dtype=float)
            m = np.nanmean(arr)
            s = np.nanstd(arr, ddof=1)
            med = np.nanmedian(arr)
            lo, hi = bootstrap_ci(arr)
            rows.append({
                "scenario": scen,
                "metric": c,
                "mean": m,
                "std": s,
                "median": med,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n": np.sum(~np.isnan(arr))
            })
    return pd.DataFrame(rows)

def add_relative_change(summary: pd.DataFrame, baseline="static_field") -> pd.DataFrame:
    # add delta% vs baseline mean
    base = summary[summary["scenario"] == baseline].set_index("metric")["mean"].to_dict()
    def rel(metric, val):
        b = base.get(metric, np.nan)
        if np.isnan(b) or b == 0:
            return np.nan
        return (val - b) / b * 100.0

    summary["delta_pct_vs_static"] = summary.apply(lambda r: rel(r["metric"], r["mean"]), axis=1)
    return summary

def boxplot_by_scenario(df: pd.DataFrame, metric: str, ylabel: str = None, fname: str = None):
    order = ["static_field", "density_field", "hazard+density"]
    data = [df[df["scenario"] == s][metric].dropna().to_numpy(dtype=float) for s in order]
    plt.figure()
    plt.boxplot(data, tick_labels=order, showfliers=True)
    plt.ylabel(ylabel or metric)
    plt.title(metric)
    plt.tight_layout()
    out = PLOTS_DIR / (fname or f"{metric}.png")
    plt.savefig(out, dpi=200)
    plt.close()

def main():
    json_paths = []
    for pattern in PATTERNS:
        for p in sorted(OUTPUT_ROOT.glob(f"{pattern}/metrics_summary.json")):
            json_paths.append(p)
    json_paths = list(dict.fromkeys(json_paths))

    if len(json_paths) == 0:
        raise FileNotFoundError(
            f"No metrics_summary.json found under {OUTPUT_ROOT.resolve()} with patterns {PATTERNS}. "
            f"Check your path."
        )

    rows = []
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(flatten_one(metrics))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "shipai_metrics_flat.csv", index=False, encoding="utf-8-sig")

    # metrics to summarize
    time_cols = ["T_0.90", "T_0.95", "T_0.99", "alive_end"]
    spatial_cols = []
    for lid in LAYER_IDS:
        spatial_cols += [
            f"layer{lid}_mean",
            f"layer{lid}_q95",
            f"layer{lid}_q99",
            f"layer{lid}_max",
            f"layer{lid}_freq_ge_max_mul_95",
        ]
    cols = time_cols + spatial_cols

    summary = summarize_group(df, cols)
    summary = add_relative_change(summary, baseline="static_field")
    summary.to_csv(RESULTS_DIR / "shipai_group_summary.csv", index=False, encoding="utf-8-sig")

    # plots: time
    for m in ["T_0.90", "T_0.95", "T_0.99", "alive_end"]:
        boxplot_by_scenario(df, m, ylabel=m, fname=f"time_{m}.png")

    # plots: spatial tail
    for lid in LAYER_IDS:
        boxplot_by_scenario(df, f"layer{lid}_q99", ylabel=f"layer{lid} q99", fname=f"layer{lid}_q99.png")
        boxplot_by_scenario(df, f"layer{lid}_freq_ge_max_mul_95",
                            ylabel=f"layer{lid} freq>=0.95*max", fname=f"layer{lid}_freq_maxmul95.png")

    print("[OK] Wrote shipai_metrics_flat.csv, shipai_group_summary.csv and plots/*.png")

if __name__ == "__main__":
    main()
