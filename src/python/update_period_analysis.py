"""
Field Update Period Analysis
============================
分析场更新周期 (Field Update Period) 对疏散性能与计算代价的影响，
寻找最优更新周期。

输入项目:
    static-0    → 静态导航 (无动态场更新)
    density-10  → 动态场，每 10s 更新一次
    density-30  → 动态场，每 30s 更新一次
    density-60  → 动态场，每 60s 更新一次
    density-120 → 动态场，每 120s 更新一次
    density-180 → 动态场，每 180s 更新一次

数据来源 (每次 run):
    {OUTPUT_ROOT}/{project}_{run_id}/bin/meta.txt
        → all_time (ms)          总迭代时间
        → dynamic_field_time (ms) 动态场迭代时间
    {OUTPUT_ROOT}/{project}_{run_id}/metrics_summary.json
        → time_efficiency: T_0.95, T_0.99
        → spatial_efficiency.layers[layer_8]: 低速行人占据时间场
            mean         → 正值格元平均低速行人占据时间 M_8^+ (agent s)
            q95          → 正值格元低速行人占据时间 95th pct
            valid_n      → 非零 Layer-8 空间足迹 N_8^+ (cells)
            area_ge_q95  → 高低速占据格元数 (Top-5% 格元数)

输出:
    ../../results/analysis/update_period_analysis/
        fig1_quality_metrics.pdf/png   质量指标对比 (4面板)
        fig2_compute_time.pdf/png      计算时间对比
        fig3_tradeoff.pdf/png          质量-效率权衡曲线
        fig4_efficiency.pdf/png        边际收益分析
        summary_table.tex              LaTeX 汇总表

依赖: numpy, pandas, scipy, matplotlib, seaborn
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter

warnings.filterwarnings("ignore")

# ============================================================================
# 配置
# ============================================================================

OUTPUT_ROOT = Path("../../results/simulation")
OUTPUT_DIR  = Path("../../results/analysis/update_period_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 项目配置: key → (显示名, 更新周期秒 / None=静态)
PROJECTS: dict[str, tuple[str, float | None]] = {
    "static-0":   ("Static",   None),
    "density-10":  ("10 s",     10.0),
    "density-30":  ("30 s",     30.0),
    "density-60":  ("60 s",     60.0),
    "density-120": ("120 s",   120.0),
    "density-180": ("180 s",   180.0),
}

# 每个项目的 run 数（自动检测，也可手动指定上限）
MAX_RUNS = 20

# 拥堵指标来自的 layer id
CONG_LAYER_ID = 8   # cumu_congestion_time_field (speed < 0.5 时累计)

# Bootstrap
N_BOOTSTRAP  = 2000
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

# ── 风格 ─────────────────────────────────────────────────────────────────────
EXPORT_DPI    = 600
FIGURE_FACE   = "white"
AXES_FACE     = "white"
GRID_COLOR    = "#D7DCE2"
TEXT_MUTED    = "#4A5568"
SPINE_COLOR   = "#A8B1BD"
BETTER_COLOR  = "#2B7A78"

# 颜色映射
PALETTE: dict[str, str] = {
    "static-0":   "#4C72B0",
    "density-10":  "#1B9E77",
    "density-30":  "#66A61E",
    "density-60":  "#E6AB02",
    "density-120": "#D95F02",
    "density-180": "#7570B3",
}
# 动态策略按更新周期排列用的颜色序列（渐变）
DYN_KEYS = ["density-10", "density-30", "density-60", "density-120", "density-180"]

# 计算时间的颜色
COLOR_ALL_TIME  = "#C44E52"
COLOR_DYN_TIME  = "#4C72B0"


# ============================================================================
# Plotting helpers
# ============================================================================

def set_publication_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": EXPORT_DPI,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.85,
        "axes.facecolor": AXES_FACE,
        "figure.facecolor": FIGURE_FACE,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "axes.labelcolor": "#1F2933",
        "text.color": "#1F2933",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#D0D6DE",
        "legend.fontsize": 9,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.65,
        "grid.alpha": 0.6,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(AXES_FACE)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.7, alpha=0.55, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(length=3.5, width=0.8, colors=TEXT_MUTED)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14, 1.06, label,
        transform=ax.transAxes,
        fontsize=13, fontweight="bold",
        va="bottom", ha="left", color="#111827",
    )


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    save_path = Path(save_path)
    pdf_path = save_path.with_suffix(".pdf")
    png_path = save_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [fig] saved → {pdf_path.name},  {png_path.name}")


# ============================================================================
# 数据加载
# ============================================================================

def parse_meta_txt(meta_path: Path) -> dict[str, float]:
    """解析 meta.txt → key=value 的字典（值均转为 float）"""
    result: dict[str, float] = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, _, v = line.partition("=")
                try:
                    result[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return result


def extract_layer_metrics(layers_list: list, layer_id: int) -> dict[str, float] | None:
    """从 spatial_efficiency.layers 中取出 layer_id 对应的指标字典"""
    target_file = f"layer_{layer_id}.bin"
    for layer in layers_list:
        if layer.get("file") == target_file:
            return layer
    return None


def load_project_runs(project_key: str) -> pd.DataFrame:
    """
    读取一个项目前缀下所有 run 的数据，返回 DataFrame。
    每行对应一次 run，列包含：
        run_id, T_0.95, T_0.99,
        cong_intensity_mean, cong_intensity_q95,
        cong_area_total, cong_area_top5pct,
        all_time, dynamic_field_time,
        soft_risk, hard_harm
    """
    records = []
    for run_id in range(1, MAX_RUNS + 1):
        run_dir = OUTPUT_ROOT / f"{project_key}_{run_id}"
        if not run_dir.exists():
            continue

        meta_path    = run_dir / "bin" / "meta.txt"
        metrics_path = run_dir / "metrics_summary.json"

        if not meta_path.exists() or not metrics_path.exists():
            continue

        meta = parse_meta_txt(meta_path)
        with open(metrics_path, encoding="utf-8") as f:
            js = json.load(f)

        time_eff    = js.get("time_efficiency", {})
        spatial_eff = js.get("spatial_efficiency", {})
        layers_list = spatial_eff.get("layers", [])
        cong_layer  = extract_layer_metrics(layers_list, CONG_LAYER_ID)

        rec: dict = {
            "run_id":      run_id,
            "project":     project_key,
            # 时间指标
            "T_0.95":      time_eff.get("T_0.95",  np.nan),
            "T_0.99":      time_eff.get("T_0.99",  np.nan),
            # 计算代价
            "all_time":         meta.get("all_time",          np.nan),
            "dynamic_field_time": meta.get("dynamic_field_time", np.nan),
            # 风险
            "soft_risk":   meta.get("soft_risk",  np.nan),
            "hard_harm":   meta.get("hard_harm",  np.nan),
        }

        if cong_layer is not None:
            rec["cong_intensity_mean"]   = cong_layer.get("mean",         np.nan)
            rec["cong_intensity_q95"]    = cong_layer.get("q95",          np.nan)
            rec["cong_area_total"]       = cong_layer.get("valid_n",      np.nan)  # 曾拥堵格数
            rec["cong_area_top5pct"]     = cong_layer.get("area_ge_q95",  np.nan)  # Top-5% 高拥堵格数
            rec["cong_max"]              = cong_layer.get("max",          np.nan)
        else:
            for col in ["cong_intensity_mean", "cong_intensity_q95",
                        "cong_area_total", "cong_area_top5pct", "cong_max"]:
                rec[col] = np.nan

        records.append(rec)

    if not records:
        raise FileNotFoundError(
            f"No valid runs found for project '{project_key}' under {OUTPUT_ROOT}"
        )
    return pd.DataFrame(records)


def load_all_data() -> pd.DataFrame:
    """加载所有项目的数据，合并为一个 DataFrame。"""
    frames = []
    for key in PROJECTS:
        print(f"  Loading '{key}' ...", end="  ")
        df = load_project_runs(key)
        print(f"{len(df)} runs")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# 统计工具
# ============================================================================

def bootstrap_ci(
    arr: np.ndarray,
    statistic: str = "mean",
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed + arr.size)
    boot = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))]
    if statistic == "mean":
        estimates = boot.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(boot, axis=1)
    elif statistic == "p95":
        estimates = np.percentile(boot, 95, axis=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic!r}")
    lo = float(np.percentile(estimates, (1 - ci) / 2 * 100))
    hi = float(np.percentile(estimates, (1 + ci) / 2 * 100))
    return lo, hi


def aggregate(df: pd.DataFrame, col: str, seed_offset: int = 0) -> dict:
    """返回某列的 mean / median / CI (Bootstrap 95%)。"""
    arr = df[col].dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"mean": np.nan, "median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}
    mean_ci   = bootstrap_ci(arr, "mean",   seed=RANDOM_SEED + seed_offset)
    return {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "ci_lo":  mean_ci[0],
        "ci_hi":  mean_ci[1],
        "n":      int(arr.size),
    }


def compute_project_stats(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 project 汇总统计所有指标，返回宽表 DataFrame。
    每行 = 一个 project，列 = {metric}_{stat}。
    """
    METRICS = [
        "T_0.95", "T_0.99",
        "cong_intensity_mean", "cong_intensity_q95",
        "cong_area_total", "cong_area_top5pct",
        "all_time", "dynamic_field_time",
        "soft_risk",
    ]
    rows = []
    for idx, (key, (label, period)) in enumerate(PROJECTS.items()):
        sub = df_all[df_all["project"] == key]
        row: dict = {
            "project": key,
            "label":   label,
            "period":  period,   # None = static
            "color":   PALETTE[key],
        }
        for i, metric in enumerate(METRICS):
            stats = aggregate(sub, metric, seed_offset=idx * 100 + i * 7)
            row[f"{metric}_mean"]  = stats["mean"]
            row[f"{metric}_ci_lo"] = stats["ci_lo"]
            row[f"{metric}_ci_hi"] = stats["ci_hi"]
            row[f"{metric}_n"]     = stats["n"]
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Figure 1 — Quality metrics comparison (4-panel bar chart)
# ============================================================================

def plot_quality_metrics(stats: pd.DataFrame, df_all: pd.DataFrame, save_path: Path) -> None:
    """
    4-panel bar chart:
        (a) T_0.95          (b) T_0.99
        (c) Congestion intensity (mean, s)
        (d) Congestion area (total congested cells)
    每个 bar = 某 project 的均值，误差棒 = 95% Bootstrap CI
    """
    projects = list(PROJECTS.keys())
    labels   = [PROJECTS[k][0] for k in projects]
    colors   = [PALETTE[k] for k in projects]
    x        = np.arange(len(projects))

    panel_specs = [
        ("a", "T_0.95",              r"$T_{0.95}$ (s)",
         r"Tail evacuation time $T_{0.95}$"),
        ("b", "T_0.99",              r"$T_{0.99}$ (s)",
         r"Extreme evacuation time $T_{0.99}$"),
        ("c", "cong_intensity_mean", r"Congestion intensity (s)",
         r"Mean congestion duration (layer 8 mean)"),
        ("d", "cong_area_total",     r"Congested area (cells)",
         r"Total congested area (layer 8 valid cells)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))

    for ax, (panel, metric, ylabel, title) in zip(axes.flat, panel_specs):
        vals   = stats[f"{metric}_mean"].to_numpy()
        ci_lo  = stats[f"{metric}_ci_lo"].to_numpy()
        ci_hi  = stats[f"{metric}_ci_hi"].to_numpy()
        err_lo = np.clip(vals - ci_lo, 0, None)
        err_hi = np.clip(ci_hi - vals, 0, None)

        bars = ax.bar(
            x, vals,
            width=0.55,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            yerr=[err_lo, err_hi],
            capsize=3.5,
            error_kw=dict(elinewidth=1.0, capthick=0.9, ecolor="#555E6C"),
            zorder=3,
        )

        # 在 bar 顶部标注数值
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(err_hi) * 0.06 + (vals.max() - vals.min()) * 0.01,
                    f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.8, color="#2D3748",
                )

        style_axes(ax, grid_axis="y")
        add_panel_label(ax, panel)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6, fontsize=10.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))

        # y 轴下界不从 0 开始，突出差异
        vmin_bar = max(0.0, (vals - err_lo).min())
        vmax_bar = (vals + err_hi).max()
        span = vmax_bar - vmin_bar
        ax.set_ylim(max(0.0, vmin_bar - 0.12 * span), vmax_bar + 0.16 * span)

        # Static 区域背景
        ax.axvspan(-0.5, 0.5, color="#E8EDF4", alpha=0.45, zorder=0)

    # 标注 Static 列
    for ax in axes.flat:
        ylim = ax.get_ylim()
        ax.text(0, ylim[1] * 0.98, "Static", ha="center", va="top",
                fontsize=8, color=TEXT_MUTED, style="italic")

    fig.tight_layout(pad=1.6, h_pad=2.2, w_pad=1.8)
    save_figure(fig, save_path)


# ============================================================================
# Figure 2 — Computation time
# ============================================================================

def plot_compute_time(stats: pd.DataFrame, df_all: pd.DataFrame, save_path: Path) -> None:
    """
    单面板分组条形图：每个 project 并列显示 all_time 和 dynamic_field_time。
    """
    projects = list(PROJECTS.keys())
    labels   = [PROJECTS[k][0] for k in projects]
    colors   = [PALETTE[k] for k in projects]
    n = len(projects)
    x = np.arange(n)
    bar_w = 0.35   # 每组两根 bar，各宽 0.35

    all_vals  = stats["all_time_mean"].to_numpy()
    all_lo    = np.clip(all_vals - stats["all_time_ci_lo"].to_numpy(), 0, None)
    all_hi    = np.clip(stats["all_time_ci_hi"].to_numpy() - all_vals, 0, None)

    dyn_vals  = stats["dynamic_field_time_mean"].to_numpy()
    dyn_lo    = np.clip(dyn_vals - stats["dynamic_field_time_ci_lo"].to_numpy(), 0, None)
    dyn_hi    = np.clip(stats["dynamic_field_time_ci_hi"].to_numpy() - dyn_vals, 0, None)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))

    # Total bars (left of each group, hatched)
    ax.bar(
        x - bar_w / 2, all_vals, bar_w,
        color=colors, edgecolor="white", linewidth=0.6,
        hatch="///", alpha=0.85,
        yerr=[all_lo, all_hi], capsize=3,
        error_kw=dict(elinewidth=1.0, capthick=0.9, ecolor="#555E6C"),
        zorder=3, label="Total time",
    )
    # Dynamic bars (right of each group, solid)
    ax.bar(
        x + bar_w / 2, dyn_vals, bar_w,
        color=colors, edgecolor="white", linewidth=0.6,
        alpha=1.0,
        yerr=[dyn_lo, dyn_hi], capsize=3,
        error_kw=dict(elinewidth=1.0, capthick=0.9, ecolor="#555E6C"),
        zorder=3, label="Dynamic field time",
    )

    style_axes(ax, grid_axis="y")
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Computation time (ms)", fontsize=11)
    ax.set_title("Total vs. dynamic field computation time by update period",
                 loc="left", pad=7, fontsize=10.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1e-3:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.set_ylim(0, None)

    # 静态列背景
    ax.axvspan(-0.5, 0.5, color="#E8EDF4", alpha=0.40, zorder=0)

    # 图例：用两个代理 patch（斜线 vs 实心）
    legend_handles = [
        Patch(facecolor="#888888", hatch="///", edgecolor="white",
              alpha=0.85, label="Total time"),
        Patch(facecolor="#888888", edgecolor="white",
              alpha=1.0, label="Dynamic field time"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.tight_layout(pad=1.6)
    save_figure(fig, save_path)


# ============================================================================
# Figure 3 — Trade-off curves (quality vs. computation)
# ============================================================================

def plot_tradeoff(stats: pd.DataFrame, save_path: Path) -> None:
    """
    2-panel scatter + curve:
        (a) T_0.95  vs  all_time        (x = computation, y = quality)
        (b) Layer-8 positive-cell mean  vs  dynamic_field_time
    动态策略连线（更新周期从小到大），静态作为参考点。
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))

    specs = [
        ("a",
         "all_time_mean", r"Total computation time (ms)",
         "T_0.95_mean",   r"Tail evacuation time $T_{0.95}$ (s)",
         "all_time_ci_lo", "all_time_ci_hi",
         "T_0.95_ci_lo",  "T_0.95_ci_hi",
         r"Evacuation time vs. total computation"),
        ("b",
         "dynamic_field_time_mean", r"Dynamic field update time (ms)",
         "cong_intensity_mean_mean", r"Mean positive-cell slow-agent occupancy $M_8^+$ (agent s)",
         "dynamic_field_time_ci_lo", "dynamic_field_time_ci_hi",
         "cong_intensity_mean_ci_lo", "cong_intensity_mean_ci_hi",
         r"Slow-agent occupancy vs. dynamic-field time"),
    ]

    dyn_stats  = stats[stats["period"].notna()].copy()
    stat_stats = stats[stats["period"].isna()].copy()

    for ax, (panel, xcol, xlabel, ycol, ytitle,
             xclo, xchi, yclo, ychi, title) in zip(axes, specs):

        # 动态策略曲线（按更新周期升序，即 x 轴 computation 递减）
        dyn  = dyn_stats.sort_values("period")
        xs   = dyn[xcol].to_numpy()
        ys   = dyn[ycol].to_numpy()
        cols = dyn["color"].to_numpy()

        ax.plot(xs, ys, color="#9B2226", linewidth=1.8, alpha=0.85,
                linestyle="--", zorder=2)

        for (_, row), xi, yi, ci in zip(dyn.sort_values("period").iterrows(),
                                          xs, ys, cols):
            ax.scatter(xi, yi, s=100, color=ci,
                       edgecolors="#111827", linewidths=0.9, zorder=4)
            # CI 误差棒
            xerr_lo = max(xi - row[xclo], 0)
            xerr_hi = max(row[xchi] - xi, 0)
            yerr_lo = max(yi - row[yclo], 0)
            yerr_hi = max(row[ychi] - yi, 0)
            ax.errorbar(xi, yi,
                        xerr=[[xerr_lo], [xerr_hi]],
                        yerr=[[yerr_lo], [yerr_hi]],
                        fmt="none", color=ci, alpha=0.4,
                        capsize=3, linewidth=0.9, zorder=3)
            # 标注更新周期
            period_label = f"{int(row['period'])}s"
            ax.annotate(period_label, (xi, yi),
                        textcoords="offset points", xytext=(6, 5),
                        fontsize=8.5, color=ci, fontweight="semibold")

        # 静态参考点
        for _, srow in stat_stats.iterrows():
            sx = srow[xcol]
            sy = srow[ycol]
            ax.scatter(sx, sy, s=140, marker="s",
                       color=srow["color"], edgecolors="#111827",
                       linewidths=0.9, zorder=5)
            ax.annotate("Static", (sx, sy),
                        textcoords="offset points", xytext=(-6, 8),
                        fontsize=8.5, color=srow["color"],
                        fontweight="bold", ha="right")

        # 无辅助线，仅保留轴框
        style_axes(ax, grid_axis="y")
        ax.grid(False)
        add_panel_label(ax, panel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ytitle)
        ax.set_title(title, loc="left", pad=7, fontsize=10.5)

        # x 轴科学计数（例：2×10⁴），减少刻度密度
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        _xfmt = ScalarFormatter(useMathText=True)
        _xfmt.set_scientific(True)
        _xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(_xfmt)

        # 添加 "Better" 方向箭头（左下 = 更少计算 + 更短疏散时间）
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        ax.annotate(
            "", xy=(xlim[0] + 0.08 * dx, ylim[0] + 0.08 * dy),
            xytext=(xlim[0] + 0.22 * dx, ylim[0] + 0.22 * dy),
            arrowprops=dict(arrowstyle="-|>", color=BETTER_COLOR,
                            lw=1.8, mutation_scale=14),
        )
        ax.text(xlim[0] + 0.24 * dx, ylim[0] + 0.24 * dy,
                "Better", color=BETTER_COLOR, fontsize=9,
                fontweight="semibold", ha="left", va="bottom")

    fig.tight_layout(pad=1.6, w_pad=2.2)
    save_figure(fig, save_path)


# ============================================================================
# Figure 4 — Marginal benefit analysis
# ============================================================================

def plot_marginal_benefit(stats: pd.DataFrame, save_path: Path) -> None:
    """
    展示随更新周期变化的边际收益（仅动态策略，按更新周期升序）：
        (a) T_0.95  vs  update period
        (b) T_0.99  vs  update period
        (c) Mean positive-cell slow-agent occupancy  vs  update period
        (d) Nonzero Layer-8 footprint  vs  update period
    Static 用水平参考线。
    标注拐点（diminishing returns elbow via second-order difference）。
    """
    dyn   = stats[stats["period"].notna()].sort_values("period").reset_index(drop=True)
    stat_ = stats[stats["period"].isna()]

    periods   = dyn["period"].to_numpy()
    log_p     = np.log10(periods)

    metric_specs = [
        ("a", "T_0.95_mean",              "T_0.95_ci_lo",          "T_0.95_ci_hi",
         r"$T_{0.95}$ (s)",               r"Tail evacuation time $T_{0.95}$",
         "T_0.95_mean"),
        ("b", "T_0.99_mean",              "T_0.99_ci_lo",          "T_0.99_ci_hi",
         r"$T_{0.99}$ (s)",               r"Extreme evacuation time $T_{0.99}$",
         "T_0.99_mean"),
        ("c", "cong_intensity_mean_mean", "cong_intensity_mean_ci_lo", "cong_intensity_mean_ci_hi",
         r"$M_8^+$ (agent s)",            r"Mean positive-cell slow-agent occupancy",
         "cong_intensity_mean_mean"),
        ("d", "cong_area_total_mean",     "cong_area_total_ci_lo", "cong_area_total_ci_hi",
         r"$N_8^+$ (cells)",              r"Nonzero Layer-8 footprint",
         "cong_area_total_mean"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)

    for ax, (panel, ycol, ci_lo_col, ci_hi_col, ylabel, title, ref_col) in \
            zip(axes.flat, metric_specs):

        y    = dyn[ycol].to_numpy()
        cilo = dyn[ci_lo_col].to_numpy()
        cihi = dyn[ci_hi_col].to_numpy()

        style_axes(ax, grid_axis="both")
        add_panel_label(ax, panel)

        # CI 带
        ax.fill_between(periods, cilo, cihi,
                        alpha=0.18, color="#9B2226", zorder=1)
        # 主曲线
        ax.plot(periods, y, color="#9B2226", linewidth=1.8, zorder=3)
        # 各点
        for pi, yi, ci in zip(periods, y, dyn["color"].to_numpy()):
            ax.scatter(pi, yi, s=55, color=ci,
                       edgecolors="#2F3A4A", linewidths=0.7, zorder=4)

        # Static 参考线
        for _, srow in stat_.iterrows():
            sval = srow[ref_col]
            ax.axhline(sval, color=PALETTE["static-0"], linestyle="--",
                       linewidth=1.3, alpha=0.85, label="Static")
            ax.text(periods[0], sval, "  Static", va="bottom",
                    fontsize=8, color=PALETTE["static-0"])

        # Elbow 检测（最大二阶差分）
        if len(y) >= 3:
            d2 = np.abs(np.diff(np.diff(y)))
            elbow_idx = int(np.argmax(d2)) + 1
            elbow_p   = periods[elbow_idx]
            ax.axvline(elbow_p, color="#6B7280", linestyle=":",
                       linewidth=1.2, alpha=0.8, zorder=2)
            ylim_cur = ax.get_ylim()
            ax.text(elbow_p, ylim_cur[0] + (ylim_cur[1] - ylim_cur[0]) * 0.04,
                    f"  {int(elbow_p)}s\n  (elbow)",
                    va="bottom", fontsize=8, color="#6B7280")

        ax.set_xscale("log")
        ax.set_xticks(periods)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6, fontsize=10.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    for ax in axes[1]:
        ax.set_xlabel("Update period (s)")

    legend_handles = [
        Line2D([0], [0], color="#9B2226", linewidth=1.8,
               label="Dynamic (mean)"),
        Patch(facecolor="#9B2226", alpha=0.22, edgecolor="none",
              label="95% bootstrap CI"),
        Line2D([0], [0], color=PALETTE["static-0"], linestyle="--",
               linewidth=1.3, label="Static baseline"),
        Line2D([0], [0], color="#6B7280", linestyle=":",
               linewidth=1.2, label="Elbow (diminishing returns)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=True,
               edgecolor="#D0D6DE", fontsize=8.8)
    fig.tight_layout(pad=1.6, h_pad=2.0, w_pad=1.8, rect=[0, 0.07, 1, 1.0])
    save_figure(fig, save_path)


# ============================================================================
# Figure 5 — Compute time vs update period (line plot)
# ============================================================================

def plot_compute_vs_period(stats: pd.DataFrame, save_path: Path) -> None:
    """
    展示计算时间随更新周期的变化（仅动态策略），
    并叠加 dynamic_field_time / all_time 比值曲线。
    帮助判断在不同更新周期下，动态场计算占整体代价的比重。
    """
    dyn = stats[stats["period"].notna()].sort_values("period").reset_index(drop=True)
    periods = dyn["period"].to_numpy()
    all_t   = dyn["all_time_mean"].to_numpy()
    dyn_t   = dyn["dynamic_field_time_mean"].to_numpy()
    ratio   = np.where(all_t > 0, dyn_t / all_t * 100, np.nan)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    style_axes(ax1, grid_axis="both")

    # all_time
    ax1.fill_between(periods,
                     dyn["all_time_ci_lo"].to_numpy(),
                     dyn["all_time_ci_hi"].to_numpy(),
                     alpha=0.15, color=COLOR_ALL_TIME)
    ax1.plot(periods, all_t, "o-", color=COLOR_ALL_TIME,
             linewidth=2.0, markersize=7, label="Total time")

    # dynamic_field_time
    ax1.fill_between(periods,
                     dyn["dynamic_field_time_ci_lo"].to_numpy(),
                     dyn["dynamic_field_time_ci_hi"].to_numpy(),
                     alpha=0.15, color=COLOR_DYN_TIME)
    ax1.plot(periods, dyn_t, "s--", color=COLOR_DYN_TIME,
             linewidth=1.8, markersize=7, label="Dynamic field time")

    ax1.set_xscale("log")
    ax1.set_xticks(periods)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.set_xlabel("Update period (s)", fontsize=11)
    ax1.set_ylabel("Computation time (ms)", fontsize=11)
    ax1.set_title("Computation cost vs. field update period",
                  loc="left", pad=7, fontsize=11)

    # 比值 (右轴)
    ax2 = ax1.twinx()
    ax2.plot(periods, ratio, "^:", color="#9B2226",
             linewidth=1.4, markersize=6, label="Dynamic / Total (%)")
    ax2.set_ylabel("Dynamic field / Total (%)", color="#9B2226", fontsize=10)
    ax2.tick_params(axis="y", colors="#9B2226")
    ax2.spines["right"].set_color("#9B2226")
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(r for r in ratio if np.isfinite(r)) * 1.3)

    # 合并图例
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc="upper right", fontsize=9)

    fig.tight_layout(pad=1.4)
    save_figure(fig, save_path)


# ============================================================================
# Figure 6 — Violin distribution (T_0.95)
# ============================================================================

def plot_violin(df_all: pd.DataFrame, save_path: Path) -> None:
    """
    全项目 T_0.95 分布对比 (Violin + Box)。
    Static 背景阴影区分。
    """
    projects = list(PROJECTS.keys())
    labels   = [PROJECTS[k][0] for k in projects]
    palette  = {PROJECTS[k][0]: PALETTE[k] for k in projects}

    records = []
    for key in projects:
        sub = df_all[df_all["project"] == key]
        for v in sub["T_0.95"].dropna().to_numpy():
            records.append({"Strategy": PROJECTS[key][0], "T_0.95": v})
    df_long = pd.DataFrame(records)

    order = [PROJECTS[k][0] for k in projects]
    fig, ax = plt.subplots(figsize=(11.0, 5.5))

    sns.violinplot(
        data=df_long, x="Strategy", y="T_0.95",
        order=order, palette=palette,
        inner=None, cut=0, linewidth=0.9, saturation=1.0, ax=ax,
    )
    sns.boxplot(
        data=df_long, x="Strategy", y="T_0.95",
        order=order, width=0.18, showcaps=True, showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="#253040",
                      linewidth=0.9, alpha=0.95),
        medianprops=dict(color="#111827", linewidth=1.5),
        whiskerprops=dict(color="#253040", linewidth=0.9),
        capprops=dict(color="#253040", linewidth=0.9),
        ax=ax,
    )

    style_axes(ax, grid_axis="y")
    ax.set_axisbelow(True)
    ax.axvspan(-0.5, 0.5, color="#E8EDF4", alpha=0.40, zorder=0)
    ax.axvline(0.5, color=SPINE_COLOR, linestyle=":", linewidth=1.0, alpha=0.9)
    ax.set_xlabel("")
    ax.set_ylabel(r"$T_{0.95}$ (s)", fontsize=11)
    ax.set_title(r"Distribution of $T_{0.95}$ by update period",
                 loc="left", pad=8, fontsize=11)

    medians = df_long.groupby("Strategy")["T_0.95"].median().reindex(order)
    for xpos, med in enumerate(medians.to_numpy()):
        if np.isfinite(med):
            ax.scatter(xpos, med, s=28, color="#111827", zorder=5)
            ax.text(xpos + 0.1, med, f"  {med:.0f}s",
                    va="center", ha="left", fontsize=8.2, color="#111827")

    y_top = ax.get_ylim()[1]
    ax.text(0.0,  y_top * 0.985, "Static",   ha="center", va="top",
            fontsize=9, color=TEXT_MUTED, style="italic")
    ax.text(3.0,  y_top * 0.985, r"Dynamic (by update period $\rightarrow$ longer)",
            ha="center", va="top", fontsize=9, color=TEXT_MUTED)

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()
    save_figure(fig, save_path)


# ============================================================================
# LaTeX summary table
# ============================================================================

def generate_latex_table(stats: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Performance and computation cost by field update period. "
        r"Values are means with 95\,\% bootstrap confidence intervals.}"
    )
    lines.append(r"\label{tab:update_period}")
    lines.append(r"\begin{tabular}{@{}lccccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Strategy} & $T_{0.95}$ (s) & $T_{0.99}$ (s) & "
        r"Cong. Intensity (s) & Cong. Area (cells) & Total Time (ms) \\"
    )
    lines.append(r"\midrule")

    for _, row in stats.iterrows():
        def fmt(col, precision=1):
            v    = row[f"{col}_mean"]
            clo  = row[f"{col}_ci_lo"]
            chi  = row[f"{col}_ci_hi"]
            if not np.isfinite(v):
                return "---"
            if precision == 0:
                return f"{v:.0f} [{clo:.0f}, {chi:.0f}]"
            return f"{v:.{precision}f} [{clo:.{precision}f}, {chi:.{precision}f}]"

        label = row["label"]
        t95   = fmt("T_0.95", 1)
        t99   = fmt("T_0.99", 1)
        ci    = fmt("cong_intensity_mean", 3)
        ca    = fmt("cong_area_total", 0)
        at    = fmt("all_time", 0)
        lines.append(f"{label} & {t95} & {t99} & {ci} & {ca} & {at} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# 主程序
# ============================================================================

def main():
    set_publication_style()

    print("=" * 72)
    print("Field Update Period Analysis")
    print("=" * 72)

    # 1. 加载数据
    print("\n[1/6] Loading data...")
    df_all = load_all_data()
    print(f"      Total: {len(df_all)} runs across {df_all['project'].nunique()} projects")

    # 2. 汇总统计
    print("\n[2/6] Computing statistics...")
    stats = compute_project_stats(df_all)
    for _, row in stats.iterrows():
        print(
            f"      {row['label']:10s}  "
            f"T95={row['T_0.95_mean']:.1f}s  "
            f"T99={row['T_0.99_mean']:.1f}s  "
            f"CongInt={row['cong_intensity_mean_mean']:.4f}s  "
            f"CongArea={row['cong_area_total_mean']:.0f}  "
            f"AllTime={row['all_time_mean']:.0f}ms  "
            f"DynTime={row['dynamic_field_time_mean']:.0f}ms"
        )

    # 3. 绘图
    print("\n[3/6] Generating figures...")

    print("  → fig1_quality_metrics")
    plot_quality_metrics(stats, df_all, OUTPUT_DIR / "fig1_quality_metrics.pdf")

    print("  → fig2_compute_time")
    plot_compute_time(stats, df_all, OUTPUT_DIR / "fig2_compute_time.pdf")

    print("  → fig3_tradeoff")
    plot_tradeoff(stats, OUTPUT_DIR / "fig3_tradeoff.pdf")

    print("  → fig4_marginal_benefit")
    plot_marginal_benefit(stats, OUTPUT_DIR / "fig4_marginal_benefit.pdf")

    print("  → fig5_compute_vs_period")
    plot_compute_vs_period(stats, OUTPUT_DIR / "fig5_compute_vs_period.pdf")

    print("  → fig6_violin")
    plot_violin(df_all, OUTPUT_DIR / "fig6_violin.pdf")

    # 4. LaTeX 表格
    print("\n[4/6] Generating LaTeX table...")
    tex = generate_latex_table(stats)
    tex_path = OUTPUT_DIR / "summary_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by update_period_analysis.py\n\n")
        f.write(tex)
    print(f"      saved → {tex_path.name}")

    # 5. 导出汇总 CSV
    print("\n[5/6] Exporting summary CSV...")
    csv_path = OUTPUT_DIR / "summary_stats.csv"
    stats.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"      saved → {csv_path.name}")

    # 6. 推荐更新周期
    print("\n[6/6] Recommendation")
    print("=" * 72)
    dyn_s = stats[stats["period"].notna()].sort_values("period")
    t95_arr = dyn_s["T_0.95_mean"].to_numpy()
    periods = dyn_s["period"].to_numpy()
    # 找 T_0.95 下降幅度最大的拐点 (二阶差分)
    if len(t95_arr) >= 3:
        d2 = np.abs(np.diff(np.diff(t95_arr)))
        elbow = int(np.argmax(d2)) + 1
        rec_period = periods[elbow]
        print(f"  Elbow of T_0.95 curve: update period = {rec_period:.0f}s")
        print(f"  → Recommended update period: {rec_period:.0f}s")
    else:
        print("  Not enough dynamic strategies to detect elbow.")
    print(f"\n  Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
