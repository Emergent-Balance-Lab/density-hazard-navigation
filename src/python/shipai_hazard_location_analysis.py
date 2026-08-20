"""
石牌案例：灾害源头位置对疏散效果的影响分析
================================================
三种灾害位置分类：
  near_exit              — 灾害源头靠近疏散出口
  near_main_road         — 灾害源头靠近主干道
  far_from_exit_and_main_road — 灾害源头远离出口与主干道

三种导航策略：
  sp-static             — 静态场
  sp-density-60         — 密度感知（Δt=60s）
  sp-risk-60            — 密度-危险复合（β=1, Δt=60s）

输出图：
  fig_hz1_grouped_bars  — 4指标分组条形图（灾害类型×策略）
  fig_hz2_interaction   — 交互折线图（灾害类型作为x轴）
  fig_hz3_reduction     — 各策略相对于Static的改善率热图+条形
  fig_hz4_violin        — T0.95 及 Σs violin 分布图
"""

import csv
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
from matplotlib.ticker import MaxNLocator, FuncFormatter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("../../results/simulation")
CSV_PATH   = Path("../../data/simulation_inputs/project_random_inputs.csv")
OUTPUT_DIR = Path("../../results/analysis/hazard_location_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 全局样式常量
# ─────────────────────────────────────────────────────────────────────────────
PROJECTS = {
    "sp-static":      ("Static",       "#4C72B0"),
    "sp-density-60":  ("Density-60s",  "#55A868"),
    "sp-risk-60":     (r"DH ($\beta=1$, 60s)", "#C44E52"),
}
PROJ_KEYS   = list(PROJECTS.keys())
PROJ_LABELS = {k: v[0] for k, v in PROJECTS.items()}
PALETTE     = {k: v[1] for k, v in PROJECTS.items()}

# 灾害位置分类
HAZARD_TYPES = [
    "near_exit",
    "near_main_road",
    "far_from_exit_and_main_road",
]
HAZARD_LABELS = {
    "near_exit":                    "Near Exit",
    "near_main_road":               "Near Main Road",
    "far_from_exit_and_main_road":  "Far from\nExit & Road",
}
HAZARD_COLORS = {
    "near_exit":                    "#E07B54",
    "near_main_road":               "#5BA4CF",
    "far_from_exit_and_main_road":  "#7DBE8E",
}

N_BOOT = 2000
SEED   = 42
np.random.seed(SEED)

EXPORT_DPI  = 300
FIGURE_FACE = "white"
AXES_FACE   = "white"
GRID_COLOR  = "#D7DCE2"
TEXT_MUTED  = "#4A5568"
SPINE_COLOR = "#A8B1BD"

# ─────────────────────────────────────────────────────────────────────────────
# 绘图辅助
# ─────────────────────────────────────────────────────────────────────────────
def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": EXPORT_DPI,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 10, "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.edgecolor": SPINE_COLOR, "axes.linewidth": 0.85,
        "axes.facecolor": AXES_FACE, "figure.facecolor": FIGURE_FACE,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED,
        "axes.labelcolor": "#1F2933", "text.color": "#1F2933",
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.edgecolor": "#D0D6DE", "legend.fontsize": 8.5,
        "grid.color": GRID_COLOR, "grid.linewidth": 0.65,
        "grid.alpha": 0.55, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(AXES_FACE)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.7,
            alpha=0.6, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(length=3.5, width=0.8, colors=TEXT_MUTED)


def save_figure(fig, stem):
    for ext in (".pdf", ".png"):
        p = OUTPUT_DIR / (stem + ext)
        fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor(),
                    dpi=EXPORT_DPI)
    plt.close(fig)
    print(f"  [fig] saved -> {stem}.pdf / .png")


def add_panel_label(ax, label):
    ax.text(-0.13, 1.06, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left",
            color="#111827")


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────
def load_hazard_map() -> dict[str, str]:
    """
    从 project_random_inputs.csv 读取每个运行的灾害位置类型。
    返回 {proj_run_name: hazard_type}，例如 {"sp-static_1": "near_exit", ...}
    """
    hmap = {}
    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if not row:
                continue
            proj_run = row[0]          # e.g. "sp-static_1"
            hazard   = row[9]          # hazard location type
            if any(proj_run.startswith(k + "_") for k in PROJ_KEYS):
                hmap[proj_run] = hazard
    return hmap


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


def load_project(proj_key: str, hazard_map: dict[str, str]) -> pd.DataFrame:
    runs = sorted(BASE_DIR.glob(f"{proj_key}_*"))
    if not runs:
        raise FileNotFoundError(f"No runs for {proj_key} in {BASE_DIR}")

    records = []
    for run_dir in runs:
        rec = {"run": run_dir.name, "proj_key": proj_key}

        # 灾害位置分类
        rec["hazard_type"] = hazard_map.get(run_dir.name, "unknown")

        # meta.txt
        meta = _parse_meta(run_dir / "bin" / "meta.txt")
        rec["all_time"]           = meta.get("all_time", np.nan)
        rec["dynamic_field_time"] = meta.get("dynamic_field_time", np.nan)

        # metrics_summary.json
        ms = json.loads((run_dir / "metrics_summary.json").read_text())
        te = ms["time_efficiency"]
        se = ms["spatial_efficiency"]
        ra = se.get("risk_all", {})

        rec["T_0.95"]    = te["T_0.95"]
        rec["T_0.99"]    = te["T_0.99"]
        rec["soft_risk"] = ra.get("soft_risk", np.nan)
        rec["hard_harm"] = ra.get("hard_harm", np.nan)

        # spatial layers
        layer_map = {l["file"]: l for l in se["layers"]}
        for lname, key_prefix in [("layer_7.bin", "L7"), ("layer_8.bin", "L8")]:
            if lname in layer_map:
                l = layer_map[lname]
                rec[f"{key_prefix}_mean"] = l["mean"]

        # run_record.csv
        rr = pd.read_csv(run_dir / "run_record.csv")
        rec["dist_mean"]   = rr["distance"].mean()
        rec["switch_mean"] = rr["route_switching_count"].mean()
        rec["n_agents"]    = len(rr)

        records.append(rec)

    return pd.DataFrame(records)


def load_all(hazard_map: dict) -> pd.DataFrame:
    frames = []
    for k in PROJ_KEYS:
        print(f"  Loading {k} ...", end="  ")
        df = load_project(k, hazard_map)
        print(f"{len(df)} runs, hazard dist: "
              + str(df["hazard_type"].value_counts().to_dict()))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 统计工具
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(arr, n_boot=N_BOOT, ci=0.95, seed=SEED):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(seed + arr.size)
    boot = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))]
    ests = boot.mean(axis=1)
    center = float(arr.mean())
    alpha = (1 - ci) / 2
    lo = float(np.percentile(ests, alpha * 100))
    hi = float(np.percentile(ests, (1 - alpha) * 100))
    return center, lo, hi


METRICS = ["T_0.95", "T_0.99", "soft_risk", "hard_harm", "L7_mean", "L8_mean",
           "dist_mean", "switch_mean"]


def compute_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 (proj_key, hazard_type) 分组计算 bootstrap CI。
    返回 MultiIndex DataFrame。
    """
    rows = []
    for proj_key in PROJ_KEYS:
        for ht in HAZARD_TYPES:
            sub = df[(df["proj_key"] == proj_key) & (df["hazard_type"] == ht)]
            row = {
                "proj_key":   proj_key,
                "hazard_type": ht,
                "label":      PROJ_LABELS[proj_key],
                "n":          len(sub),
            }
            for m in METRICS:
                if m not in sub.columns or sub[m].isna().all():
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_ci_lo"] = np.nan
                    row[f"{m}_ci_hi"] = np.nan
                    continue
                c, lo, hi = bootstrap_ci(sub[m].values)
                row[f"{m}_mean"]  = c
                row[f"{m}_ci_lo"] = lo
                row[f"{m}_ci_hi"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — 分组条形图：4指标×灾害类型（策略为颜色）
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1_grouped_bars(summary: pd.DataFrame, save_path: str):
    """
    4面板：T0.95 / T0.99 / Σs / H
    x轴 = 灾害位置类型（3个分组）
    每组 = 3根柱（对应3种策略），带95%CI误差条
    """
    specs = [
        ("a", "T_0.95",    r"$T_{0.95}$ (s)",     r"95th-pct evacuation time $T_{0.95}$"),
        ("b", "T_0.99",    r"$T_{0.99}$ (s)",     r"99th-pct evacuation time $T_{0.99}$"),
        ("c", "soft_risk", r"Soft risk $\Sigma_s$", r"Cumulative soft risk $\Sigma_s$"),
        ("d", "hard_harm", r"Hard harm $H$",        r"Hard harm count $H$"),
    ]

    n_ht   = len(HAZARD_TYPES)
    n_proj = len(PROJ_KEYS)
    bar_w  = 0.22
    x      = np.arange(n_ht)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    axes = axes.flat

    for ax, (panel, metric, ylabel, title) in zip(axes, specs):
        for pi, pk in enumerate(PROJ_KEYS):
            means, errs_lo, errs_hi = [], [], []
            for ht in HAZARD_TYPES:
                row = summary[(summary["proj_key"] == pk) &
                              (summary["hazard_type"] == ht)]
                if row.empty:
                    means.append(np.nan); errs_lo.append(0); errs_hi.append(0)
                    continue
                m  = row[f"{metric}_mean"].values[0]
                lo = row[f"{metric}_ci_lo"].values[0]
                hi = row[f"{metric}_ci_hi"].values[0]
                means.append(m)
                errs_lo.append(max(m - lo, 0))
                errs_hi.append(max(hi - m, 0))

            offset = (pi - (n_proj - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset, means, bar_w,
                color=PALETTE[pk], alpha=0.88,
                label=PROJ_LABELS[pk], zorder=3,
                linewidth=0.5, edgecolor="white",
            )
            ax.errorbar(
                x + offset, means,
                yerr=[errs_lo, errs_hi],
                fmt="none", color="#333333",
                capsize=3.5, capthick=0.9, linewidth=0.9, zorder=4,
            )

            # 标注数值
            for rect, m in zip(bars, means):
                if np.isfinite(m):
                    ym = rect.get_height()
                    fmt_val = f"{m/1e6:.1f}M" if metric in ("soft_risk",) and m > 1e5 \
                              else f"{m/1e3:.1f}k" if m > 9999 \
                              else f"{m:.0f}"
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            ym * 1.015, fmt_val,
                            ha="center", va="bottom",
                            fontsize=6.5, color="#222222", rotation=0)

        ax.set_title(title, pad=6)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [HAZARD_LABELS[ht] for ht in HAZARD_TYPES],
            fontsize=9,
        )
        style_axes(ax)
        if metric in ("soft_risk", "hard_harm"):
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _:
                    f"{v/1e6:.1f}M" if v >= 1e6 else
                    f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
            )
        add_panel_label(ax, f"({panel})")

    # 全局图例
    handles = [mpatches.Patch(color=PALETTE[k], label=PROJ_LABELS[k])
               for k in PROJ_KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle("Evacuation & Risk Metrics by Hazard Source Location",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — 交互折线图：灾害类型×策略
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2_interaction(summary: pd.DataFrame, save_path: str):
    """
    4面板（T0.95 / T0.99 / Σs / H），每面板：
    x轴 = 3种灾害位置，折线 = 3种策略，带误差带 (95%CI)
    """
    specs = [
        ("a", "T_0.95",    r"$T_{0.95}$ (s)"),
        ("b", "T_0.99",    r"$T_{0.99}$ (s)"),
        ("c", "soft_risk", r"Soft risk $\Sigma_s$"),
        ("d", "hard_harm", r"Hard harm $H$"),
    ]
    x_pos = np.arange(len(HAZARD_TYPES))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes = axes.flat

    for ax, (panel, metric, ylabel) in zip(axes, specs):
        for pk in PROJ_KEYS:
            means, los, his = [], [], []
            for ht in HAZARD_TYPES:
                row = summary[(summary["proj_key"] == pk) &
                              (summary["hazard_type"] == ht)]
                if row.empty:
                    means.append(np.nan); los.append(np.nan); his.append(np.nan)
                    continue
                means.append(row[f"{metric}_mean"].values[0])
                los.append(row[f"{metric}_ci_lo"].values[0])
                his.append(row[f"{metric}_ci_hi"].values[0])

            means = np.array(means, dtype=float)
            los   = np.array(los,   dtype=float)
            his   = np.array(his,   dtype=float)

            ax.plot(x_pos, means, "o-",
                    color=PALETTE[pk], label=PROJ_LABELS[pk],
                    linewidth=2.0, markersize=6, zorder=4)
            ax.fill_between(x_pos, los, his,
                            color=PALETTE[pk], alpha=0.18, zorder=2)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [HAZARD_LABELS[ht] for ht in HAZARD_TYPES],
            fontsize=9,
        )
        style_axes(ax, grid_axis="y")
        if metric in ("soft_risk", "hard_harm"):
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _:
                    f"{v/1e6:.1f}M" if v >= 1e6 else
                    f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
            )
        add_panel_label(ax, f"({panel})")

    handles = [Line2D([0], [0], color=PALETTE[k], lw=2,
                      marker="o", markersize=5, label=PROJ_LABELS[k])
               for k in PROJ_KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle("Strategy × Hazard Location Interaction",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — 相对改善率热图（vs Static 基准）
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3_reduction(summary: pd.DataFrame, save_path: str):
    """
    2面板（上：Density-60s 相对Static改善率，下：DH相对Static改善率）
    横轴 = 4种指标，纵轴 = 3种灾害位置
    单元格颜色 = 改善率（%），绿色=改善，红色=退化
    同时在下方绘制条形图展示 T0.95 和 Σs 的改善率
    """
    metrics_info = [
        ("T_0.95",    r"$T_{0.95}$"),
        ("T_0.99",    r"$T_{0.99}$"),
        ("soft_risk", r"$\Sigma_s$"),
        ("hard_harm", r"$H$"),
    ]
    dyn_projs = [("sp-density-60", "Density-60s"), ("sp-risk-60", r"DH ($\beta=1$, 60s)")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (pk, plabel) in zip(axes, dyn_projs):
        mat = np.zeros((len(HAZARD_TYPES), len(metrics_info)))
        annot = np.empty_like(mat, dtype=object)

        for ri, ht in enumerate(HAZARD_TYPES):
            ref_row = summary[(summary["proj_key"] == "sp-static") &
                              (summary["hazard_type"] == ht)]
            cmp_row = summary[(summary["proj_key"] == pk) &
                              (summary["hazard_type"] == ht)]
            for ci, (mkey, _) in enumerate(metrics_info):
                if ref_row.empty or cmp_row.empty:
                    mat[ri, ci]   = 0
                    annot[ri, ci] = "—"
                    continue
                ref_v = ref_row[f"{mkey}_mean"].values[0]
                cmp_v = cmp_row[f"{mkey}_mean"].values[0]
                if np.isfinite(ref_v) and ref_v != 0:
                    pct = (ref_v - cmp_v) / ref_v * 100   # positive = improvement
                    mat[ri, ci]   = pct
                    annot[ri, ci] = f"{pct:+.1f}%"
                else:
                    mat[ri, ci]   = 0
                    annot[ri, ci] = "—"

        # 热图
        vmax = max(abs(mat).max(), 5)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       aspect="auto")

        ax.set_xticks(range(len(metrics_info)))
        ax.set_xticklabels([m[1] for m in metrics_info], fontsize=10)
        ax.set_yticks(range(len(HAZARD_TYPES)))
        ax.set_yticklabels(
            [HAZARD_LABELS[ht].replace("\n", " ") for ht in HAZARD_TYPES],
            fontsize=9,
        )

        # 在每个单元格中写数值
        for ri in range(len(HAZARD_TYPES)):
            for ci in range(len(metrics_info)):
                v = mat[ri, ci]
                text_color = "black" if abs(v) < vmax * 0.6 else "white"
                ax.text(ci, ri, annot[ri, ci],
                        ha="center", va="center",
                        fontsize=10.5, fontweight="bold",
                        color=text_color)

        ax.set_title(f"{plabel}\nvs. Static baseline", fontsize=10,
                     fontweight="semibold", pad=8)

        # 网格线
        for i in range(len(HAZARD_TYPES) + 1):
            ax.axhline(i - 0.5, color="white", linewidth=1.2)
        for j in range(len(metrics_info) + 1):
            ax.axvline(j - 0.5, color="white", linewidth=1.2)

        plt.colorbar(im, ax=ax, label="Improvement vs. Static (%)",
                     shrink=0.85, pad=0.02)

    fig.suptitle("Relative Improvement by Hazard Location and Strategy",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Violin / 箱形图：T0.95 和 Σs 的分布
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig4_violin(df: pd.DataFrame, save_path: str):
    """
    2行×3列：行 = T0.95 / soft_risk，列 = 3种灾害位置
    每个小图 = violin（策略为x轴颜色）
    """
    metrics_info = [
        ("T_0.95",    r"$T_{0.95}$ (s)",         None),
        ("soft_risk", r"Soft risk $\Sigma_s$",     1e6),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharey="row")

    for row_i, (metric, ylabel, scale) in enumerate(metrics_info):
        for col_i, ht in enumerate(HAZARD_TYPES):
            ax = axes[row_i, col_i]

            sub = df[df["hazard_type"] == ht].copy()
            if scale:
                sub[metric] = sub[metric] / scale

            parts = ax.violinplot(
                [sub[sub["proj_key"] == pk][metric].dropna().values
                 for pk in PROJ_KEYS],
                positions=range(len(PROJ_KEYS)),
                showmedians=True, showextrema=False, widths=0.65,
            )
            for pi, (body, pk) in enumerate(zip(parts["bodies"], PROJ_KEYS)):
                body.set_facecolor(PALETTE[pk])
                body.set_alpha(0.72)
                body.set_edgecolor("#444444")
                body.set_linewidth(0.6)
            parts["cmedians"].set_color("#222222")
            parts["cmedians"].set_linewidth(1.5)

            # 叠加 strip
            for pi, pk in enumerate(PROJ_KEYS):
                vals = sub[sub["proj_key"] == pk][metric].dropna().values
                jitter = np.random.default_rng(SEED + pi).uniform(
                    -0.12, 0.12, size=len(vals))
                ax.scatter(pi + jitter, vals, s=7, alpha=0.45,
                           color=PALETTE[pk], zorder=3, linewidths=0)

            ax.set_xticks(range(len(PROJ_KEYS)))
            ax.set_xticklabels(
                [PROJ_LABELS[pk] for pk in PROJ_KEYS],
                fontsize=8.5, rotation=15, ha="right",
            )
            style_axes(ax)

            if col_i == 0:
                lbl = ylabel if scale is None else f"{ylabel} ($\\times10^6$)"
                ax.set_ylabel(lbl)
            if row_i == 0:
                ax.set_title(HAZARD_LABELS[ht].replace("\n", " "),
                             fontsize=10, pad=5)

            # 各组样本数
            n_vals = [len(sub[sub["proj_key"] == pk]) for pk in PROJ_KEYS]
            ax.set_xlabel(
                "  ".join(f"n={n}" for n in n_vals),
                fontsize=7.5, color=TEXT_MUTED,
            )

    fig.suptitle(r"Distribution of $T_{0.95}$ and $\Sigma_s$ by Hazard Location",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — 策略改善率条形图：DH vs Density vs Static 按灾害类型细分
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig5_delta_bars(summary: pd.DataFrame, save_path: str):
    """
    2×2 面板：T0.95 / T0.99 / Σs / H 的改善率（%，相对Static）
    x轴 = 灾害类型，双条（Density vs DH）
    """
    specs = [
        ("a", "T_0.95",    r"$\Delta T_{0.95}$ (% vs Static)"),
        ("b", "T_0.99",    r"$\Delta T_{0.99}$ (% vs Static)"),
        ("c", "soft_risk", r"$\Delta\Sigma_s$ (% vs Static)"),
        ("d", "hard_harm", r"$\Delta H$ (% vs Static)"),
    ]
    dyn_projs = [
        ("sp-density-60", "Density-60s",         "#55A868"),
        ("sp-risk-60",    r"DH ($\beta=1$, 60s)", "#C44E52"),
    ]

    n_ht  = len(HAZARD_TYPES)
    bar_w = 0.3
    x     = np.arange(n_ht)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes = axes.flat

    for ax, (panel, metric, ylabel) in zip(axes, specs):
        for pi, (pk, plabel, color) in enumerate(dyn_projs):
            pcts, err_lo, err_hi = [], [], []
            for ht in HAZARD_TYPES:
                ref = summary[(summary["proj_key"] == "sp-static") &
                              (summary["hazard_type"] == ht)]
                cmp = summary[(summary["proj_key"] == pk) &
                              (summary["hazard_type"] == ht)]
                if ref.empty or cmp.empty:
                    pcts.append(0); err_lo.append(0); err_hi.append(0)
                    continue
                ref_m  = ref[f"{metric}_mean"].values[0]
                cmp_m  = cmp[f"{metric}_mean"].values[0]
                ref_lo = ref[f"{metric}_ci_lo"].values[0]
                ref_hi = ref[f"{metric}_ci_hi"].values[0]
                cmp_lo = cmp[f"{metric}_ci_lo"].values[0]
                cmp_hi = cmp[f"{metric}_ci_hi"].values[0]
                if ref_m and np.isfinite(ref_m):
                    pct = (ref_m - cmp_m) / ref_m * 100
                    # 简单误差传播（上限/下限 CI 组合）
                    pct_lo = (ref_hi - cmp_hi) / ref_hi * 100 if ref_hi else pct
                    pct_hi = (ref_lo - cmp_lo) / ref_lo * 100 if ref_lo else pct
                    pcts.append(pct)
                    err_lo.append(max(pct - min(pct_lo, pct_hi), 0))
                    err_hi.append(max(max(pct_lo, pct_hi) - pct, 0))
                else:
                    pcts.append(0); err_lo.append(0); err_hi.append(0)

            offset = (pi - 0.5) * bar_w
            bars = ax.bar(
                x + offset, pcts, bar_w,
                color=color, alpha=0.88, label=plabel,
                zorder=3, linewidth=0.5, edgecolor="white",
            )
            ax.errorbar(
                x + offset, pcts,
                yerr=[err_lo, err_hi],
                fmt="none", color="#333333",
                capsize=3, capthick=0.9, linewidth=0.9, zorder=4,
            )
            # 数值标注
            for rect, pct in zip(bars, pcts):
                if np.isfinite(pct) and abs(pct) > 0.5:
                    ypos = rect.get_height() + (0.5 if pct >= 0 else -1.5)
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            ypos, f"{pct:.1f}%",
                            ha="center", va="bottom", fontsize=7,
                            color="#222222")

        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [HAZARD_LABELS[ht] for ht in HAZARD_TYPES],
            fontsize=9,
        )
        style_axes(ax)
        add_panel_label(ax, f"({panel})")

    handles = [mpatches.Patch(color=c, label=lbl)
               for _, lbl, c in dyn_projs]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Improvement Rate vs. Static by Hazard Location",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_figure(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 文字摘要
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_table(summary: pd.DataFrame):
    print("\n" + "=" * 80)
    print("灾害位置 × 策略 核心指标摘要")
    print("=" * 80)
    fmt = "{:<35} {:>8} {:>8} {:>12} {:>12}"
    print(fmt.format("Strategy / Hazard Type", "T0.95", "T0.99",
                     "Σs (M p·s)", "H (k)"))
    print("-" * 80)
    for ht in HAZARD_TYPES:
        print(f"\n>> {HAZARD_LABELS[ht].replace(chr(10), ' ')}")
        for pk in PROJ_KEYS:
            row = summary[(summary["proj_key"] == pk) &
                          (summary["hazard_type"] == ht)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(fmt.format(
                f"  {PROJ_LABELS[pk]} (n={r['n']:.0f})",
                f"{r['T_0.95_mean']:.0f}",
                f"{r['T_0.99_mean']:.0f}",
                f"{r['soft_risk_mean']/1e6:.2f}",
                f"{r['hard_harm_mean']/1e3:.1f}",
            ))


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_style()

    print("Loading hazard location map from CSV ...")
    hazard_map = load_hazard_map()
    print(f"  {len(hazard_map)} run entries loaded")

    print("\nLoading simulation output data ...")
    df = load_all(hazard_map)
    print(f"\nTotal runs loaded: {len(df)}")
    print("Unknown hazard types:",
          df[df["hazard_type"] == "unknown"]["run"].tolist())

    print("\nComputing bootstrap CIs by (strategy × hazard type) ...")
    summary = compute_group_summary(df)
    print_summary_table(summary)

    print("\nGenerating figures ...")
    plot_fig1_grouped_bars(summary, "fig_hz1_grouped_bars")
    plot_fig2_interaction(summary,  "fig_hz2_interaction")
    plot_fig3_reduction(summary,    "fig_hz3_reduction")
    plot_fig4_violin(df,            "fig_hz4_violin")
    plot_fig5_delta_bars(summary,   "fig_hz5_delta_bars")

    print("\nAll done. Output in:", OUTPUT_DIR.resolve())
