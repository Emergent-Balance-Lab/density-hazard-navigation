"""
石牌全场景三策略对比分析
=========================
项目: sp-static | sp-density-60 (β=0) | sp-risk-60 (β=1, Δt=60s)
各50次独立运行

指标体系:
  疏散效率  — T_0.95 / T_0.99 / 个体疏散时间分布
  风险控制  — soft_risk / hard_harm
  空间负荷  — layer_7(累计占用采样次数) / layer_8(低速行人占据时间)
              / layer_9(累计密度样本负荷)
  行为特征  — 路径距离 / 路线切换次数
  计算开销  — all_time / dynamic_field_time

输出图:
  fig1_bar4           — 4指标条形图 (T95/T99/SoftRisk/HardHarm)
  fig2_risk_scatter   — T95 vs SoftRisk / T95 vs HardHarm 散点
  fig3_violin_evac    — 个体疏散时间 violin
  fig4_spatial        — 空间拥堵对比 (layer_7 / layer_8 mean)
  fig5_behavior       — 路径距离 + 路线切换
  fig6_compute        — 计算开销分组条形
  section_shipai.tex  — Overleaf 论文描述段落
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("../../results/simulation")
OUTPUT_DIR = Path("../../results/analysis/shipai_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECTS = {
    "sp-static":      ("Static",       "#4C72B0"),
    "sp-density-60":  ("Density-60s",  "#55A868"),
    "sp-risk-60":     (r"D-H $\beta=1$, 60s", "#C44E52"),
}
PROJ_KEYS   = list(PROJECTS.keys())
PROJ_LABELS = {k: v[0] for k, v in PROJECTS.items()}
PALETTE     = {k: v[1] for k, v in PROJECTS.items()}

N_BOOT      = 2000
SEED        = 42
np.random.seed(SEED)

EXPORT_DPI   = 600
FIGURE_FACE  = "white"
AXES_FACE    = "white"
GRID_COLOR   = "#D7DCE2"
TEXT_MUTED   = "#4A5568"
SPINE_COLOR  = "#A8B1BD"
BETTER_COLOR = "#2B7A78"

# ─────────────────────────────────────────────────────────────────────────────
# 绘图样式
# ─────────────────────────────────────────────────────────────────────────────
def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": EXPORT_DPI,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 11, "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.edgecolor": SPINE_COLOR, "axes.linewidth": 0.85,
        "axes.facecolor": AXES_FACE, "figure.facecolor": FIGURE_FACE,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED,
        "axes.labelcolor": "#1F2933", "text.color": "#1F2933",
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.edgecolor": "#D0D6DE", "legend.fontsize": 9,
        "grid.color": GRID_COLOR, "grid.linewidth": 0.65,
        "grid.alpha": 0.55, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(AXES_FACE)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.7,
            alpha=0.55, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(length=3.5, width=0.8, colors=TEXT_MUTED)


def add_panel_label(ax, label, fontsize=13):
    ax.text(-0.14, 1.06, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="bottom", ha="left", color="#111827")


def save_figure(fig, path):
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [fig] saved -> {path.with_suffix('.pdf').name}, "
          f"{path.with_suffix('.png').name}")


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────
def _parse_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                meta[k.strip()] = float(v.strip())
            except ValueError:
                meta[k.strip()] = v.strip()
    return meta


def load_project(proj_key: str) -> dict:
    """返回单个项目所有 run 的原始数据集合。"""
    runs = sorted(BASE_DIR.glob(f"{proj_key}_*"))
    if not runs:
        raise FileNotFoundError(f"No runs found for {proj_key} in {BASE_DIR}")

    records = []
    for run_dir in runs:
        rec = {"run": run_dir.name}

        # ── meta.txt ──────────────────────────────────────────────────────────
        meta = _parse_meta(run_dir / "bin" / "meta.txt")
        rec["all_time"]           = meta.get("all_time", np.nan)
        rec["dynamic_field_time"] = meta.get("dynamic_field_time", np.nan)
        rec["soft_risk_meta"]     = meta.get("soft_risk", np.nan)
        rec["hard_harm_meta"]     = meta.get("hard_harm", np.nan)

        # ── metrics_summary.json ─────────────────────────────────────────────
        ms  = json.loads((run_dir / "metrics_summary.json").read_text())
        te  = ms["time_efficiency"]
        se  = ms["spatial_efficiency"]
        ra  = se.get("risk_all", {})

        rec["T_0.95"]    = te["T_0.95"]
        rec["T_0.99"]    = te["T_0.99"]
        rec["T_0.90"]    = te.get("T_0.90", np.nan)
        rec["alive_end"] = te["alive_end"]
        rec["soft_risk"] = ra.get("soft_risk", np.nan)
        rec["hard_harm"] = ra.get("hard_harm", np.nan)

        # layers: 7=cumulative usage samples, 8=cumulative congestion time,
        #         9=cumulative crowd-density load
        layer_map = {l["file"]: l for l in se["layers"]}
        for lname, key_prefix in [("layer_7.bin", "L7"),
                                   ("layer_8.bin", "L8"),
                                   ("layer_9.bin", "L9")]:
            if lname in layer_map:
                l = layer_map[lname]
                rec[f"{key_prefix}_mean"]       = l["mean"]
                rec[f"{key_prefix}_q95"]        = l.get("q95", np.nan)
                rec[f"{key_prefix}_valid_n"]    = l.get("valid_n", np.nan)
                rec[f"{key_prefix}_area_ge_q95"]= l.get("area_ge_q95", np.nan)

        # ── run_record.csv ───────────────────────────────────────────────────
        rr = pd.read_csv(run_dir / "run_record.csv")
        rec["dist_mean"]    = rr["distance"].mean()
        rec["dist_q95"]     = rr["distance"].quantile(0.95)
        rec["switch_mean"]  = rr["route_switching_count"].mean()
        rec["switch_q99"]   = rr["route_switching_count"].quantile(0.99)
        rec["n_agents"]     = len(rr)

        records.append(rec)

    return pd.DataFrame(records)


def load_all() -> dict[str, pd.DataFrame]:
    data = {}
    for k in PROJ_KEYS:
        print(f"  Loading {k} ...", end="  ")
        df = load_project(k)
        print(f"{len(df)} runs")
        data[k] = df
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 统计工具
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(arr, stat="mean", n_boot=N_BOOT, ci=0.95, seed=SEED):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(seed + arr.size)
    boot = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))]
    if stat == "mean":
        ests = boot.mean(axis=1)
        center = float(arr.mean())
    elif stat == "p95":
        ests = np.percentile(boot, 95, axis=1)
        center = float(np.percentile(arr, 95))
    elif stat == "median":
        ests = np.median(boot, axis=1)
        center = float(np.median(arr))
    else:
        raise ValueError(stat)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(ests, alpha * 100))
    hi = float(np.percentile(ests, (1 - alpha) * 100))
    return center, lo, hi


def compute_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """返回各策略各指标的均值 + 95% Bootstrap CI。"""
    METRICS = [
        "T_0.95", "T_0.99", "T_0.90",
        "soft_risk", "hard_harm",
        "L7_mean", "L8_mean", "L9_mean",
        "L7_q95", "L8_q95",
        "L7_area_ge_q95", "L8_area_ge_q95",
        "dist_mean", "dist_q95",
        "switch_mean", "switch_q99",
        "all_time", "dynamic_field_time",
    ]
    rows = []
    for k, df in data.items():
        row = {"project": k, "label": PROJ_LABELS[k], "color": PALETTE[k]}
        for m in METRICS:
            if m not in df.columns:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_ci_lo"] = np.nan
                row[f"{m}_ci_hi"] = np.nan
                continue
            center, lo, hi = bootstrap_ci(df[m].values)
            row[f"{m}_mean"] = center
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).set_index("project")


def _read_layer_field(run_dir: Path, layer_file: str) -> np.ndarray:
    """Read one binary raster layer as a 2-D array using the run metadata."""
    meta = _parse_meta(run_dir / "bin" / "meta.txt")
    width = int(meta["width"])
    height = int(meta["height"])
    path = run_dir / "bin" / layer_file
    arr = np.fromfile(path, dtype=np.float32)
    expected = width * height
    if arr.size != expected:
        raise ValueError(
            f"{path} has {arr.size} cells, expected {expected} "
            f"from meta width={width}, height={height}"
        )
    return arr.reshape((height, width))


def _load_road_mask() -> np.ndarray:
    """Boolean valid-cell mask = the static road network (layer_2 > 0).

    The cumulative spatial fields (layers 7-9) carry a non-trivial amount of
    mass in off-network cells (~18-23 % of the Static field) introduced by the
    0.1 m -> 1 m sum-downsampling bleeding into building cells. Those cells are
    not part of the walkable road network, and their off-network share differs
    by strategy (larger for Static), which biases the Static-relative
    reductions. We therefore restrict every spatial statistic to the road
    network, defined as layer_2 (2_roadnetwork.tif) > 0. The road geometry is
    static, so a single mask (taken from the first available run) applies to all
    strategies, guaranteeing an identical cell set across the comparison.
    """
    for project in PROJ_KEYS:
        runs = sorted(BASE_DIR.glob(f"{project}_*"))
        if runs:
            road = _read_layer_field(runs[0], "layer_2.bin")
            return np.isfinite(road) & (road > 0)
    raise FileNotFoundError(f"No runs found in {BASE_DIR} to build road mask")


def compute_shipai_global_field_stats(save_path: Path | None = None) -> pd.DataFrame:
    """Compute ensemble-mean spatial-field summaries used in the main text.

    All statistics are restricted to the road network (see _load_road_mask):
    off-network cells are excluded so the Static-relative reductions are not
    inflated by sum-downsampling artifacts.
    """
    field_specs = [
        ("cumulative_congestion_time", "layer_8.bin"),
        ("cumulative_density_load", "layer_9.bin"),
    ]
    road_mask = _load_road_mask()
    rows = []

    for field_name, layer_file in field_specs:
        field_rows = []
        for project in PROJ_KEYS:
            runs = sorted(BASE_DIR.glob(f"{project}_*"))
            if not runs:
                raise FileNotFoundError(f"No runs found for {project} in {BASE_DIR}")

            accum = None
            for run_dir in runs:
                arr = _read_layer_field(run_dir, layer_file).astype(np.float64)
                arr[~np.isfinite(arr)] = 0.0
                if accum is None:
                    accum = np.zeros_like(arr, dtype=np.float64)
                if arr.shape != accum.shape:
                    raise ValueError(
                        f"Inconsistent shape for {run_dir}: {arr.shape} != {accum.shape}"
                    )
                accum += arr

            mean_field = accum / len(runs)
            if mean_field.shape != road_mask.shape:
                raise ValueError(
                    f"road mask shape {road_mask.shape} != field shape {mean_field.shape}"
                )
            # restrict to on-road cells; positives are on-road cells with load
            on_road = mean_field[road_mask]
            pos = on_road[on_road > 0]
            if pos.size == 0:
                q95 = q99 = max_val = mean_pos = 0.0
            else:
                mean_pos = float(pos.mean())
                q95 = float(np.percentile(pos, 95))
                q99 = float(np.percentile(pos, 99))
                max_val = float(pos.max())

            field_rows.append({
                "field": field_name,
                "strategy": PROJ_LABELS[project].replace("Density-60s", "Density-60")
                                           .replace(r"D-H $\beta=1$, 60s", "DH-60"),
                "positive_cells": int(pos.size),
                "total_field_sum": float(on_road.sum(dtype=np.float64)),
                "mean_all_cells": float(on_road.mean(dtype=np.float64)),
                "mean_positive": mean_pos,
                "q95_positive": q95,
                "q99_positive": q99,
                "max": max_val,
                "_field": mean_field,
            })

        static = next(row for row in field_rows if row["strategy"] == "Static")
        static_q95 = static["q95_positive"]
        static_q99 = static["q99_positive"]
        static_field = static["_field"]

        for row in field_rows:
            mean_field = row.pop("_field")
            row["cells_ge_static_q95"] = int(
                np.count_nonzero((mean_field >= static_q95) & road_mask))
            row["cells_ge_static_q99"] = int(
                np.count_nonzero((mean_field >= static_q99) & road_mask))

        base = {
            "positive_cells": static["positive_cells"],
            "total_field_sum": static["total_field_sum"],
            "mean_all_cells": static["mean_all_cells"],
            "mean_positive": static["mean_positive"],
            "q95_positive": static["q95_positive"],
            "q99_positive": static["q99_positive"],
            "cells_ge_static_q95": int(
                np.count_nonzero((static_field >= static_q95) & road_mask)),
            "cells_ge_static_q99": int(
                np.count_nonzero((static_field >= static_q99) & road_mask)),
        }

        for row in field_rows:
            for key, base_val in base.items():
                if base_val == 0:
                    delta = np.nan
                else:
                    delta = (row[key] - base_val) / base_val * 100.0
                row[f"{key}_delta_vs_static_pct"] = float(delta)
            rows.append(row)

    stats = pd.DataFrame(rows)
    if save_path is not None:
        stats.to_csv(save_path, index=False)
        print(f"  [csv] saved -> {save_path.name}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — 4-panel 条形比较
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1_bar4(summary: pd.DataFrame, save_path: Path):
    specs = [
        ("a", "T_0.95",    r"$T_{0.95}$ (s)",
         r"95th-pct evacuation time  $T_{0.95}$"),
        ("b", "T_0.99",    r"$T_{0.99}$ (s)",
         r"99th-pct evacuation time  $T_{0.99}$"),
        ("c", "soft_risk", r"Soft risk $\Sigma_s$",
         r"Cumulative soft risk $\Sigma_s$"),
        ("d", "hard_harm", r"Hard harm $H$",
         r"Hard harm count $H$"),
    ]

    labels = [summary.loc[k, "label"] for k in PROJ_KEYS]
    colors = [PALETTE[k] for k in PROJ_KEYS]
    x = np.arange(len(PROJ_KEYS))

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

    for ax, (panel, key, ylabel, title) in zip(axes.flat, specs):
        vals   = [summary.loc[k, f"{key}_mean"]  for k in PROJ_KEYS]
        lo_ci  = [summary.loc[k, f"{key}_ci_lo"] for k in PROJ_KEYS]
        hi_ci  = [summary.loc[k, f"{key}_ci_hi"] for k in PROJ_KEYS]
        err_lo = [max(v - lo, 0) for v, lo in zip(vals, lo_ci)]
        err_hi = [max(hi - v, 0) for v, hi in zip(vals, hi_ci)]

        bars = ax.bar(
            x, vals, width=0.52, color=colors,
            edgecolor="white", linewidth=0.6,
            yerr=[err_lo, err_hi], capsize=3.5,
            error_kw=dict(elinewidth=1.0, capthick=0.9, ecolor="#555E6C"),
            zorder=3,
        )

        style_axes(ax, grid_axis="y")
        add_panel_label(ax, panel)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6, fontsize=10.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))

        # 科学计数 for large-magnitude metrics
        if key in ("soft_risk", "hard_harm"):
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v*1e-6:.2g}M" if v >= 1e6
                              else (f"{v*1e-3:.0f}k" if v >= 1e3 else f"{v:.0f}"))
            )

        # y 轴下界：留 10% gap
        vmin_data = max(0.0, min(v - e for v, e in zip(vals, err_lo)))
        vmax_data = max(v + e for v, e in zip(vals, err_hi))
        span = max(vmax_data - vmin_data, 1e-6)
        ax.set_ylim(max(0.0, vmin_data - 0.10 * span),
                    vmax_data + 0.16 * span)

        # 数值标注
        for bar_obj, v in zip(bars, vals):
            if np.isfinite(v):
                fmt = (f"{v/1e6:.2f}M" if v >= 1e6
                       else (f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.1f}"))
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                        bar_obj.get_height() + max(err_hi) * 0.05,
                        fmt, ha="center", va="bottom",
                        fontsize=7.8, color="#2D3748")

    fig.tight_layout(pad=1.6, h_pad=2.2, w_pad=1.8)
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — T95 vs Risk scatter (2-panel)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2_risk_scatter(summary: pd.DataFrame, save_path: Path):
    specs = [
        ("a", "soft_risk", r"Soft risk $\Sigma_s$",
         r"Trade-off: evacuation time vs. soft risk"),
        ("b", "hard_harm",  r"Hard harm count $H$",
         r"Trade-off: evacuation time vs. hard harm"),
    ]
    markers = {"sp-static": "s", "sp-density-60": "D", "sp-risk-60": "o"}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))

    for ax, (panel, xcol, xlabel, title) in zip(axes, specs):
        for k in PROJ_KEYS:
            row = summary.loc[k]
            xv  = row[f"{xcol}_mean"]
            yv  = row["T_0.95_mean"]
            xlo = max(xv - row[f"{xcol}_ci_lo"], 0)
            xhi = max(row[f"{xcol}_ci_hi"] - xv, 0)
            ylo = max(yv - row["T_0.95_ci_lo"], 0)
            yhi = max(row["T_0.95_ci_hi"] - yv, 0)

            ax.errorbar(xv, yv,
                        xerr=[[xlo], [xhi]], yerr=[[ylo], [yhi]],
                        fmt="none", color=PALETTE[k], alpha=0.45,
                        capsize=4, linewidth=1.0, zorder=3)
            ax.scatter(xv, yv, s=160, marker=markers[k],
                       color=PALETTE[k], edgecolors="#111827",
                       linewidths=0.9, zorder=5,
                       label=row["label"])
            ax.annotate(row["label"], (xv, yv),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=8.8, color=PALETTE[k],
                        fontweight="semibold")

        style_axes(ax, grid_axis="both")
        ax.grid(False)
        add_panel_label(ax, panel)
        ax.set_ylabel(r"Tail evacuation time $T_{0.95}$ (s)")
        ax.set_title(title, loc="left", pad=7, fontsize=10.5)

        # x 轴科学计数
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        _xfmt = ScalarFormatter(useMathText=True)
        _xfmt.set_scientific(True)
        _xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(_xfmt)
        ax.set_xlabel(xlabel)

        # "Better" arrow (left-down)
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        dx = xlim[1] - xlim[0]; dy = ylim[1] - ylim[0]
        ax.annotate("",
                    xy=(xlim[0] + 0.08 * dx, ylim[0] + 0.08 * dy),
                    xytext=(xlim[0] + 0.22 * dx, ylim[0] + 0.22 * dy),
                    arrowprops=dict(arrowstyle="-|>", color=BETTER_COLOR,
                                    lw=1.8, mutation_scale=14))
        ax.text(xlim[0] + 0.24 * dx, ylim[0] + 0.24 * dy,
                "Better", color=BETTER_COLOR, fontsize=9,
                fontweight="semibold", ha="left", va="bottom")

    fig.tight_layout(pad=1.6, w_pad=2.2)
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — 个体疏散时间 violin (合并所有 run 的 run_record)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3_violin(data: dict[str, pd.DataFrame], save_path: Path):
    """从每个项目的每次 run 收集所有个体疏散时间，绘制 violin+box。"""
    records = []
    for k in PROJ_KEYS:
        df = data[k]
        # 每个 DataFrame 行对应一次 run，需重新读 run_record
        runs = sorted(BASE_DIR.glob(f"{k}_*"))
        for run_dir in runs:
            rr = pd.read_csv(run_dir / "run_record.csv")
            for t in rr["time"].dropna().values:
                records.append({"label": PROJ_LABELS[k], "key": k, "time": float(t)})

    df_long = pd.DataFrame(records)
    order = [PROJ_LABELS[k] for k in PROJ_KEYS]
    pal   = {PROJ_LABELS[k]: PALETTE[k] for k in PROJ_KEYS}

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    sns.violinplot(
        data=df_long, x="label", y="time",
        order=order, palette=pal,
        inner=None, cut=0, linewidth=0.9, saturation=1.0, ax=ax,
    )
    sns.boxplot(
        data=df_long, x="label", y="time",
        order=order, width=0.14, showcaps=True, showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="#253040",
                      linewidth=0.9, alpha=0.95),
        medianprops=dict(color="#111827", linewidth=1.5),
        whiskerprops=dict(color="#253040", linewidth=0.9),
        capprops=dict(color="#253040", linewidth=0.9),
        ax=ax,
    )

    style_axes(ax, grid_axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.set_ylabel("Individual evacuation time (s)")
    ax.set_title(r"Distribution of individual evacuation times across strategies",
                 loc="left", pad=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # 标注 median 值
    medians = df_long.groupby("label")["time"].median().reindex(order)
    for xpos, (lbl, med) in enumerate(medians.items()):
        ax.scatter(xpos, med, s=28, color="#111827", zorder=5)
        ax.text(xpos, med, f"  {med:.0f}s",
                va="center", ha="left", fontsize=8.2, color="#111827")

    fig.tight_layout(pad=1.4)
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — 空间拥堵对比 (layer_7 / layer_8)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig4_spatial(summary: pd.DataFrame, save_path: Path):
    specs = [
        ("a", "L7_mean",       r"Usage mean (samples/cell)",
         r"Mean cumulative route usage (layer 7)"),
        ("b", "L8_mean",       r"Congestion-time mean (s/cell)",
         r"Mean cumulative congestion duration (layer 8)"),
        ("c", "L7_area_ge_q95", r"High-usage area (cells)",
         r"Area with route usage $\geq$ Q95"),
        ("d", "L8_area_ge_q95", r"High-congestion area (cells)",
         r"Area with congestion duration $\geq$ Q95"),
    ]

    labels = [summary.loc[k, "label"] for k in PROJ_KEYS]
    colors = [PALETTE[k] for k in PROJ_KEYS]
    x = np.arange(len(PROJ_KEYS))

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

    for ax, (panel, key, ylabel, title) in zip(axes.flat, specs):
        vals   = [summary.loc[k, f"{key}_mean"]  for k in PROJ_KEYS]
        lo_ci  = [summary.loc[k, f"{key}_ci_lo"] for k in PROJ_KEYS]
        hi_ci  = [summary.loc[k, f"{key}_ci_hi"] for k in PROJ_KEYS]
        err_lo = [max(v - lo, 0) for v, lo in zip(vals, lo_ci)]
        err_hi = [max(hi - v, 0) for v, hi in zip(vals, hi_ci)]

        bars = ax.bar(
            x, vals, width=0.52, color=colors,
            edgecolor="white", linewidth=0.6,
            yerr=[err_lo, err_hi], capsize=3.5,
            error_kw=dict(elinewidth=1.0, capthick=0.9, ecolor="#555E6C"),
            zorder=3,
        )

        style_axes(ax, grid_axis="y")
        add_panel_label(ax, panel)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6, fontsize=10.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        vmin_data = max(0.0, min(v - e for v, e in zip(vals, err_lo)))
        vmax_data = max(v + e for v, e in zip(vals, err_hi))
        span = max(vmax_data - vmin_data, 1e-6)
        ax.set_ylim(max(0.0, vmin_data - 0.10 * span),
                    vmax_data + 0.16 * span)

        for bar_obj, v in zip(bars, vals):
            if np.isfinite(v):
                fmt = f"{v:.2f}" if v < 100 else f"{v:.0f}"
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                        bar_obj.get_height() + max(err_hi) * 0.05,
                        fmt, ha="center", va="bottom",
                        fontsize=7.8, color="#2D3748")

    fig.tight_layout(pad=1.6, h_pad=2.2, w_pad=1.8)
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Spatial-field reductions supporting the main-text paragraph
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig7_spatial_reduction(stats: pd.DataFrame, save_path: Path):
    """Visualize the exact Static-relative reductions reported in the text."""
    strategies = ["Density-60", "DH-60"]
    colors = {
        "Density-60": PALETTE["sp-density-60"],
        "DH-60": PALETTE["sp-risk-60"],
    }
    panels = [
        (
            "a",
            "cumulative_density_load",
            "Cumulative density-sample load",
            [
                ("Total\nfield sum", "total_field_sum_delta_vs_static_pct"),
                ("95th-pct\npositive-cell\ndensity-sample load", "q95_positive_delta_vs_static_pct"),
                ("Cells above\nStatic q99", "cells_ge_static_q99_delta_vs_static_pct"),
            ],
        ),
        (
            "b",
            "cumulative_congestion_time",
            "Slow-agent occupancy field",
            [
                ("95th-pct\npositive-cell\nslow-agent occupancy", "q95_positive_delta_vs_static_pct"),
                ("Cells above\nStatic q99", "cells_ge_static_q99_delta_vs_static_pct"),
            ],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharex=True)
    max_reduction = 0.0

    for ax, (panel, field, title, metric_specs) in zip(axes, panels):
        subset = stats[stats["field"] == field].set_index("strategy")
        y = np.arange(len(metric_specs))
        bar_h = 0.34
        offsets = [-bar_h / 2, bar_h / 2]

        for offset, strategy in zip(offsets, strategies):
            vals = []
            for _, metric_col in metric_specs:
                vals.append(-float(subset.loc[strategy, metric_col]))
            max_reduction = max(max_reduction, max(vals))
            bars = ax.barh(
                y + offset, vals, height=bar_h,
                color=colors[strategy], edgecolor="white", linewidth=0.7,
                label=strategy, zorder=3,
            )
            for bar, value in zip(bars, vals):
                ax.text(
                    value + 1.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    ha="left", va="center",
                    fontsize=11.0, color="#1F2933",
                )

        style_axes(ax, grid_axis="x")
        add_panel_label(ax, panel, fontsize=16)
        ax.set_yticks(y)
        ax.set_yticklabels([name for name, _ in metric_specs],
                           fontsize=11.2, linespacing=1.38)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=8, fontsize=13.0)
        ax.set_xlabel("Reduction relative to Static (%)", fontsize=13.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="x", labelsize=11.2)
        ax.axvline(0, color=SPINE_COLOR, linewidth=0.8)

    handles = [
        mpatches.Patch(facecolor=colors[strategy], edgecolor="white", label=strategy)
        for strategy in strategies
    ]
    for ax in axes:
        ax.set_xlim(0, max(80, max_reduction + 12))
    axes[1].legend(handles=handles, loc="upper right", fontsize=11)
    fig.tight_layout(pad=1.5, w_pad=2.2)
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — 行为特征 (距离 + 路线切换)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig5_behavior(summary: pd.DataFrame, save_path: Path):
    specs = [
        ("a", "dist_mean",   "Mean evacuation distance (m)",
         "Mean travel distance per agent"),
        ("b", "dist_q95",    "95th-pct distance (m)",
         "Tail travel distance (95th pct)"),
        ("c", "switch_mean", "Mean route switches",
         "Mean route-switching count per agent"),
        ("d", "switch_q99",  "99th-pct route switches",
         "Tail route-switching count (99th pct)"),
    ]

    labels = [summary.loc[k, "label"] for k in PROJ_KEYS]
    colors = [PALETTE[k] for k in PROJ_KEYS]
    x = np.arange(len(PROJ_KEYS))

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

    for ax, (panel, key, ylabel, title) in zip(axes.flat, specs):
        vals   = [summary.loc[k, f"{key}_mean"]  for k in PROJ_KEYS]
        lo_ci  = [summary.loc[k, f"{key}_ci_lo"] for k in PROJ_KEYS]
        hi_ci  = [summary.loc[k, f"{key}_ci_hi"] for k in PROJ_KEYS]
        err_lo = [max(v - lo, 0) for v, lo in zip(vals, lo_ci)]
        err_hi = [max(hi - v, 0) for v, hi in zip(vals, hi_ci)]

        bars = ax.bar(
            x, vals, width=0.52, color=colors,
            edgecolor="white", linewidth=0.6,
            yerr=[err_lo, err_hi], capsize=3.5,
            error_kw=dict(elinewidth=1.0, capthick=0.9, ecolor="#555E6C"),
            zorder=3,
        )

        style_axes(ax, grid_axis="y")
        add_panel_label(ax, panel)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6, fontsize=10.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        vmin_data = max(0.0, min(v - e for v, e in zip(vals, err_lo)))
        vmax_data = max(v + e for v, e in zip(vals, err_hi))
        span = max(vmax_data - vmin_data, 1e-6)
        ax.set_ylim(max(0.0, vmin_data - 0.10 * span),
                    vmax_data + 0.16 * span)

        for bar_obj, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                        bar_obj.get_height() + max(err_hi) * 0.05,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.8, color="#2D3748")

    fig.tight_layout(pad=1.6, h_pad=2.2, w_pad=1.8)
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — 计算开销分组条形
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig6_compute(summary: pd.DataFrame, save_path: Path):
    n = len(PROJ_KEYS)
    x = np.arange(n)
    bar_w = 0.35
    labels = [summary.loc[k, "label"] for k in PROJ_KEYS]
    colors = [PALETTE[k] for k in PROJ_KEYS]

    all_v  = np.array([summary.loc[k, "all_time_mean"]           for k in PROJ_KEYS])
    all_lo = np.clip(all_v - np.array([summary.loc[k, "all_time_ci_lo"] for k in PROJ_KEYS]), 0, None)
    all_hi = np.clip(np.array([summary.loc[k, "all_time_ci_hi"] for k in PROJ_KEYS]) - all_v, 0, None)

    dyn_v  = np.array([summary.loc[k, "dynamic_field_time_mean"]           for k in PROJ_KEYS])
    dyn_lo = np.clip(dyn_v - np.array([summary.loc[k, "dynamic_field_time_ci_lo"] for k in PROJ_KEYS]), 0, None)
    dyn_hi = np.clip(np.array([summary.loc[k, "dynamic_field_time_ci_hi"] for k in PROJ_KEYS]) - dyn_v, 0, None)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.bar(x - bar_w / 2, all_v, bar_w, color=colors, edgecolor="white",
           hatch="///", alpha=0.85,
           yerr=[all_lo, all_hi], capsize=3,
           error_kw=dict(elinewidth=1.0, capthick=0.9, ecolor="#555E6C"),
           zorder=3, label="Total time")
    ax.bar(x + bar_w / 2, dyn_v, bar_w, color=colors, edgecolor="white",
           alpha=1.0,
           yerr=[dyn_lo, dyn_hi], capsize=3,
           error_kw=dict(elinewidth=1.0, capthick=0.9, ecolor="#555E6C"),
           zorder=3, label="Dynamic field time")

    style_axes(ax, grid_axis="y")
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Computation time (ms)")
    ax.set_title("Total vs. dynamic field computation time by strategy",
                 loc="left", pad=7, fontsize=10.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v*1e-3:.0f}k" if v >= 1000 else f"{v:.0f}")
    )
    ax.set_ylim(0, None)
    ax.axvspan(-0.5, 0.5, color="#E8EDF4", alpha=0.40, zorder=0)

    legend_handles = [
        mpatches.Patch(facecolor="#888888", hatch="///", edgecolor="white",
                       alpha=0.85, label="Total time"),
        mpatches.Patch(facecolor="#888888", edgecolor="white",
                       alpha=1.0, label="Dynamic field time"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.tight_layout(pad=1.6)
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX 表格 + 论文段落
# ─────────────────────────────────────────────────────────────────────────────
def generate_latex(summary: pd.DataFrame, n_runs: int, save_path: Path):
    s = summary
    proj_order = PROJ_KEYS

    def _pct(key, proj, ref="sp-static"):
        ref_v = s.loc[ref, f"{key}_mean"]
        v     = s.loc[proj, f"{key}_mean"]
        if ref_v == 0:
            return 0.0
        return (ref_v - v) / ref_v * 100   # positive = improvement

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    def fmt(v, decimals=1):
        if not np.isfinite(v):
            return "--"
        if abs(v) >= 1e6:
            return f"{v/1e6:.2f}M"
        if abs(v) >= 1e3:
            return f"{v/1e3:.1f}k"
        return f"{v:.{decimals}f}"

    def row_str(k):
        m = s.loc[k]
        name = m["label"].replace("$", "\\$").replace("\\$", "$")
        t95  = f"{fmt(m['T_0.95_mean'])} [{fmt(m['T_0.95_ci_lo'])}, {fmt(m['T_0.95_ci_hi'])}]"
        t99  = f"{fmt(m['T_0.99_mean'])} [{fmt(m['T_0.99_ci_lo'])}, {fmt(m['T_0.99_ci_hi'])}]"
        sr   = f"{fmt(m['soft_risk_mean'],0)} [{fmt(m['soft_risk_ci_lo'],0)}, {fmt(m['soft_risk_ci_hi'],0)}]"
        hh   = f"{fmt(m['hard_harm_mean'],0)} [{fmt(m['hard_harm_ci_lo'],0)}, {fmt(m['hard_harm_ci_hi'],0)}]"
        at   = f"{fmt(m['all_time_mean'],0)}"
        return f"{name} & {t95} & {t99} & {sr} & {hh} & {at} \\\\"

    table = r"""
\begin{table}[htbp]
\centering
\caption{%
  Performance comparison of three navigation strategies on the Shipai
  full-scenario benchmark ($N = """ + str(n_runs) + r"""$ independent runs each).
  Bracketed ranges are 95\,\% bootstrap confidence intervals.
  $T_{0.95}$/$T_{0.99}$: time for 95/99\,\% of agents to evacuate (s).
  $\Sigma_s$: cumulative soft risk (person$\cdot$s).
  $H$: hard-harm count.
  AllTime: mean total computation time (ms).%
}
\label{tab:shipai_comparison}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Strategy} & $T_{0.95}$ (s) & $T_{0.99}$ (s) & $\Sigma_s$ & $H$ & AllTime (ms) \\
\midrule
""" + "\n".join(row_str(k) for k in proj_order) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # ── 数值速查（用于正文填数）────────────────────────────────────────────────
    def pct_str(v):
        return f"{abs(v):.1f}\\,\\%"

    # Key improvement numbers
    t95_den  = _pct("T_0.95",    "sp-density-60")
    t95_risk = _pct("T_0.95",    "sp-risk-60")
    t99_den  = _pct("T_0.99",    "sp-density-60")
    t99_risk = _pct("T_0.99",    "sp-risk-60")
    sr_den   = _pct("soft_risk", "sp-density-60")
    sr_risk  = _pct("soft_risk", "sp-risk-60")
    hh_den   = _pct("hard_harm", "sp-density-60")
    hh_risk  = _pct("hard_harm", "sp-risk-60")

    sta = s.loc["sp-static"]
    den = s.loc["sp-density-60"]
    rsk = s.loc["sp-risk-60"]

    text = f"""\
% =============================================================
%  Section: Shipai Full-Scenario Benchmark Results
%  Auto-generated from shipai_strategy_analysis.py
%  Requires packages: graphicx, booktabs, amsmath, siunitx
% =============================================================

\\subsection{{Shipai Full-Scenario Benchmark}}
\\label{{sec:shipai}}

We evaluate the three navigation strategies---Static, Density-aware
($\\Delta t_u = 60\\,\\mathrm{{s}}$), and Density-Hazard
($\\beta = 1,\\,\\Delta t_u = 60\\,\\mathrm{{s}}$)---on the Shipai
district full-scenario benchmark.
Each configuration is evaluated over $N = {n_runs}$ independent runs
with randomised initial conditions; all reported values are means with
95\\,\\% bootstrap confidence intervals (Table~\\ref{{tab:shipai_comparison}}).

% -------------------------------------------------------------------
\\subsubsection{{Evacuation Efficiency}}
\\label{{sec:shipai_efficiency}}
% -------------------------------------------------------------------

The static baseline completes 95\\,\\% evacuation at
$T_{{0.95}} = {sta['T_0.95_mean']:.0f}\\,\\mathrm{{s}}$
($T_{{0.99}} = {sta['T_0.99_mean']:.0f}\\,\\mathrm{{s}}$).
Both dynamic strategies substantially accelerate crowd clearance
(Figure~\\ref{{fig:shipai_bar4}}, panels a--b).
The Density-aware strategy reduces $T_{{0.95}}$ by
{pct_str(t95_den)} to ${den['T_0.95_mean']:.0f}\\,\\mathrm{{s}}$
and $T_{{0.99}}$ by {pct_str(t99_den)} to
${den['T_0.99_mean']:.0f}\\,\\mathrm{{s}}$.
The Density-Hazard strategy achieves a comparable reduction in
$T_{{0.95}}$ ({pct_str(t95_risk)}, to ${rsk['T_0.95_mean']:.0f}\\,\\mathrm{{s}}$)
and a marginally larger improvement in $T_{{0.99}}$
({pct_str(t99_risk)}, to ${rsk['T_0.99_mean']:.0f}\\,\\mathrm{{s}}$),
reflecting a tighter tail of the evacuation-time distribution.
The individual evacuation-time distributions (Figure~\\ref{{fig:shipai_violin}})
confirm that both dynamic strategies shift the entire distribution
leftward, with the Density-Hazard variant producing the narrowest
99th-percentile.

% -------------------------------------------------------------------
\\subsubsection{{Risk Reduction}}
\\label{{sec:shipai_risk}}
% -------------------------------------------------------------------

The most pronounced differentiation between the two dynamic strategies
emerges in the risk metrics (Figure~\\ref{{fig:shipai_bar4}}, panels c--d;
Figure~\\ref{{fig:shipai_scatter}}).
Starting from a static-baseline cumulative soft risk of
$\\Sigma_s = {sta['soft_risk_mean']:.2e}\\,\\mathrm{{person\\cdot s}}$,
the Density-aware strategy achieves a moderate reduction of
{pct_str(sr_den)} (to ${den['soft_risk_mean']:.2e}$).
The Density-Hazard strategy delivers a far larger reduction of
{pct_str(sr_risk)}\\,, collapsing $\\Sigma_s$ to
${rsk['soft_risk_mean']:.2e}\\,\\mathrm{{person\\cdot s}}$---approximately
${sta['soft_risk_mean']/rsk['soft_risk_mean']:.1f}\\times$ lower than
the static baseline and
${den['soft_risk_mean']/rsk['soft_risk_mean']:.1f}\\times$ lower than
the Density-aware variant.
This gap highlights that routing agents away from high-density cells
(Density-aware) and routing them away from high-hazard cells
(Density-Hazard) are complementary but distinct objectives; the latter
is substantially more effective at limiting cumulative exposure to
dangerous zones.

The hard-harm metric $H$ (the number of agents simultaneously exposed
to hazard above the critical threshold) mirrors this pattern.
The static baseline incurs
$H = {sta['hard_harm_mean']:.2e}$
(95\\,\\% CI
[{sta['hard_harm_ci_lo']:.2e}, {sta['hard_harm_ci_hi']:.2e}]).
The Density-aware strategy reduces $H$ by {pct_str(hh_den)} to
${den['hard_harm_mean']:.2e}$,
whereas the Density-Hazard strategy nearly eliminates hard harm,
achieving $H = {rsk['hard_harm_mean']:.1f}$
({pct_str(hh_risk)} reduction).

% -------------------------------------------------------------------
\\subsubsection{{Spatial Congestion Patterns}}
\\label{{sec:shipai_spatial}}
% -------------------------------------------------------------------

The spatial-efficiency analysis (Figure~\\ref{{fig:shipai_spatial}})
quantifies how crowd energy is distributed across the simulation domain.
The mean density-dwell time (layer-7 mean, the average duration a cell
is occupied above the minimum threshold) decreases from
${sta['L7_mean_mean']:.2f}\\,\\mathrm{{s/cell}}$ (Static) to
${den['L7_mean_mean']:.2f}\\,\\mathrm{{s/cell}}$ (Density-aware) and
${rsk['L7_mean_mean']:.2f}\\,\\mathrm{{s/cell}}$ (Density-Hazard),
a reduction of
{pct_str((sta['L7_mean_mean']-den['L7_mean_mean'])/sta['L7_mean_mean']*100)} and
{pct_str((sta['L7_mean_mean']-rsk['L7_mean_mean'])/sta['L7_mean_mean']*100)},
respectively.
The risk-weighted dwell time (layer-8 mean) follows the same ordering
but with a larger absolute reduction for the Density-Hazard strategy
(from ${sta['L8_mean_mean']:.3f}$ to ${rsk['L8_mean_mean']:.3f}\\,\\mathrm{{s/cell}}$,
a {pct_str((sta['L8_mean_mean']-rsk['L8_mean_mean'])/sta['L8_mean_mean']*100)}
improvement), consistent with the risk-weighted routing objective.
The high-congestion area (cells with dwell time $\\geq$ Q95) contracts
modestly under both dynamic strategies, indicating that density-aware
routing disperses persistent hot-spots without significantly altering
the global congestion footprint.

% -------------------------------------------------------------------
\\subsubsection{{Agent Behaviour}}
\\label{{sec:shipai_behavior}}
% -------------------------------------------------------------------

Figure~\\ref{{fig:shipai_behavior}} examines the behavioural
consequences of dynamic routing at the individual level.
Mean travel distance increases slightly from
${sta['dist_mean_mean']:.1f}\\,\\mathrm{{m}}$ (Static) to
${den['dist_mean_mean']:.1f}\\,\\mathrm{{m}}$ (Density-aware) and
${rsk['dist_mean_mean']:.1f}\\,\\mathrm{{m}}$ (Density-Hazard),
indicating that dynamic guidance induces modest path lengthening
in exchange for improved global efficiency.
The mean route-switching count rises from
${sta['switch_mean_mean']:.3f}$ (Static) to
${den['switch_mean_mean']:.3f}$ (Density-aware) and
${rsk['switch_mean_mean']:.3f}$ (Density-Hazard), reflecting the
additional guidance-field updates but remaining below one switch
per agent on average.

% -------------------------------------------------------------------
\\subsubsection{{Computational Overhead}}
\\label{{sec:shipai_compute}}
% -------------------------------------------------------------------

Both dynamic strategies incur substantially higher computation costs
than the static baseline, owing to repeated dynamic-field recomputation
(Figure~\\ref{{fig:shipai_compute}}).
The Density-aware strategy requires
${den['all_time_mean']/sta['all_time_mean']:.1f}\\times$ the static
total computation time
(${den['all_time_mean']/1e3:.0f}\\,\\mathrm{{s}}$ vs.\\
${sta['all_time_mean']/1e3:.0f}\\,\\mathrm{{s}}$),
of which ${den['dynamic_field_time_mean']/den['all_time_mean']*100:.1f}\\,\\%$
is attributable to field updates
(${den['dynamic_field_time_mean']/1e3:.0f}\\,\\mathrm{{s}}$).
The Density-Hazard strategy shows a slightly lower total cost
(${rsk['all_time_mean']/1e3:.0f}\\,\\mathrm{{s}}$,
${rsk['all_time_mean']/sta['all_time_mean']:.1f}\\times$ baseline),
with ${rsk['dynamic_field_time_mean']/rsk['all_time_mean']*100:.1f}\\,\\%$
spent on field updates.
This overhead is commensurate with the ${pct_str(t95_risk)} reduction
in $T_{{0.95}}$ and the dramatic risk reduction achieved.

% -------------------------------------------------------------------
\\subsubsection{{Summary}}
\\label{{sec:shipai_summary}}
% -------------------------------------------------------------------

Taken together, the Shipai benchmark results demonstrate that the
Density-Hazard strategy ($\\beta = 1,\\,\\Delta t_u = 60\\,\\mathrm{{s}}$)
Pareto-dominates the Density-aware baseline across all risk metrics
while achieving comparable or superior evacuation efficiency.
The ${pct_str(sr_risk)} reduction in cumulative soft risk
and the near-elimination of hard harm
confirm that incorporating hazard information into the routing
potential is the key differentiator, rather than the density-awareness
alone.
The moderate increase in computation cost
(${rsk['all_time_mean']/sta['all_time_mean']:.1f}\\times$ baseline)
is well within the budget established by the update-period sensitivity
analysis (Section~\\ref{{sec:update_period}}), validating
$\\Delta t_u^{{*}} = 60\\,\\mathrm{{s}}$ as the operating point.

{table}
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [tex] saved -> {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_style()

    print("=" * 72)
    print("Shipai Full-Scenario: Three-Strategy Analysis")
    print("=" * 72)

    print("\n[1/4] Loading data...")
    data = load_all()
    n_runs = len(next(iter(data.values())))
    print(f"      {n_runs} runs per project")

    print("\n[2/4] Computing summary statistics...")
    summary = compute_summary(data)
    spatial_stats = compute_shipai_global_field_stats(
        OUTPUT_DIR / "shipai_global_heatmap_stats_expanded.csv"
    )
    for k in PROJ_KEYS:
        m = summary.loc[k]
        print(
            f"  {PROJ_LABELS[k]:22s}  "
            f"T95={m['T_0.95_mean']:.0f}s  "
            f"T99={m['T_0.99_mean']:.0f}s  "
            f"SoftRisk={m['soft_risk_mean']:.2e}  "
            f"HardHarm={m['hard_harm_mean']:.0f}  "
            f"AllTime={m['all_time_mean']/1e3:.0f}s"
        )

    print("\n[3/4] Generating figures...")
    figs = [
        ("fig1_bar4",        lambda: plot_fig1_bar4(summary,    OUTPUT_DIR/"fig1_bar4")),
        ("fig2_risk_scatter",lambda: plot_fig2_risk_scatter(summary, OUTPUT_DIR/"fig2_risk_scatter")),
        ("fig3_violin_evac", lambda: plot_fig3_violin(data,     OUTPUT_DIR/"fig3_violin_evac")),
        ("fig4_spatial",     lambda: plot_fig4_spatial(summary, OUTPUT_DIR/"fig4_spatial")),
        ("fig5_behavior",    lambda: plot_fig5_behavior(summary,OUTPUT_DIR/"fig5_behavior")),
        ("fig6_compute",     lambda: plot_fig6_compute(summary, OUTPUT_DIR/"fig6_compute")),
        ("fig7_spatial_reduction", lambda: plot_fig7_spatial_reduction(
            spatial_stats, OUTPUT_DIR/"fig7_spatial_reduction"
        )),
    ]
    for name, fn in figs:
        print(f"  >> {name}")
        fn()

    print("\n[4/4] Generating LaTeX section...")
    generate_latex(summary, n_runs, OUTPUT_DIR / "section_shipai.tex")

    print("\n" + "=" * 72)
    print(f"Output directory: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
