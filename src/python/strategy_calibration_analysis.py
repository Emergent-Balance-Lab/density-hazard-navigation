"""
Macro-level Validation: Strategy Comparison and Risk-weight Sensitivity
=======================================================================
对应论文 Section 4.X:

4.X.1  Baseline strategy comparison
        Fixed-parameter comparison: Static vs Density-aware vs Density-Hazard (β = β₀)
        Metrics: T_0.95  /  T_0.99  /  Sigma_s (population soft-risk)  /  Hard Harm

4.X.2  Sensitivity to risk weight β
        Only for Density-Hazard, β ∈ {0.1, 0.5, 1.0, 5.0, 10.0}
        Trade-off curve / Diminishing returns / Recommended operating range

七种策略映射:
    static    → Static baseline
    density   → Density-aware baseline
    risk_01   → Density-Hazard  β = 0.1
    risk_05   → Density-Hazard  β = 0.5
    risk_10   → Density-Hazard  β = 1.0   ← 默认 β₀
    risk_50   → Density-Hazard  β = 5.0
    risk_100  → Density-Hazard  β = 10.0

数据来源:
    优先读取  ../../results/analysis/uncertainty/paired_samples_all.csv  (含7种策略的完整数据)
    次选读取  ../../results/analysis/uncertainty/paired_samples.csv      (3种策略, 其余策略用插值模拟)
    均不存在时: 自动生成合成数据（用于调试）

依赖: numpy, pandas, scipy, matplotlib, seaborn
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无 GUI 环境也能保存 PDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

warnings.filterwarnings("ignore")

# ============================================================================
# 配置
# ============================================================================

DATA_FULL    = Path("../../results/analysis/uncertainty/paired_samples_all.csv")
DATA_PARTIAL = Path("../../results/analysis/uncertainty/paired_samples.csv")
OUTPUT_DIR   = Path("../../results/analysis/macro_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 七种策略：key=后缀, value=显示名
STRATEGY_KEYS  = ["static", "density", "risk_01", "risk_05", "risk_10", "risk_50", "risk_100"]
STRATEGY_NAMES = {
    "static":   "Static",
    "density":  "Density",
    "risk_01":  r"D-H $\beta=0.1$",
    "risk_05":  r"D-H $\beta=0.5$",
    "risk_10":  r"D-H $\beta=1.0$",
    "risk_50":  r"D-H $\beta=5.0$",
    "risk_100": r"D-H $\beta=10.0$",
}

# β 值对应（用于 4.X.2 轴刻度）
BETA_VALUES = {
    "risk_01":  0.1,
    "risk_05":  0.5,
    "risk_10":  1.0,
    "risk_50":  5.0,
    "risk_100": 10.0,
}

# 4.X.1 比较的基线策略组
BASELINE_STRATEGIES  = ["static", "density", "risk_10"]   # risk_10 作为 β₀ = 1.0
# 4.X.2 β 敏感性策略组（只含 risk_* 系列，按 β 排序）
BETA_STRATEGIES      = ["risk_01", "risk_05", "risk_10", "risk_50", "risk_100"]

# 主要指标
TIME_METRIC     = "T_0.95"      # 对应列: T_0.95_{strategy}
TIME_METRIC_99  = "T_0.99"      # 对应列: T_0.99_{strategy}
EXPOSURE_METRIC = "soft_risk"   # 对应列: soft_risk_{strategy}
HARM_METRIC     = "hard_harm"   # 对应列: hard_harm_{strategy}

# Bootstrap
N_BOOTSTRAP  = 1000
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

# ── Outlier filtering ────────────────────────────────────────────────────────
# 过滤方法: "iqr" | "percentile" | "zscore"
# 三种方法对每条时间指标列独立检测，只要某列把某行标为异常，整行删除。
OUTLIER_METHOD = "iqr"     # 推荐 "iqr"（Tukey fences），鲁棒性最好

# IQR 方法: 超出 [Q1 - k·IQR, Q3 + k·IQR] 视为异常
# k=1.5 → 标准离群; k=3.0 → 极端离群（更保守，只删最离谱的）
OUTLIER_IQR_K = 3.0

# Percentile 方法: 删除低于 lo 或高于 hi 百分位的行
OUTLIER_PERCENTILE_LO = 1.0
OUTLIER_PERCENTILE_HI = 99.0

# Z-score 方法: |z| > thresh 视为异常
OUTLIER_ZSCORE_THRESH = 3.5

# 用于离群检测的列前缀列表（只对这些指标做判断）
# 可以加入 "soft_risk" 等；留空则只检测时间指标
OUTLIER_CHECK_METRICS = [TIME_METRIC]   # e.g. ["T_0.95", "soft_risk"]

# ── 硬性下界过滤（Early-stop / 仿真崩溃防护）────────────────────────────────
# 时间指标的物理下界：任何 T 低于此值即视为仿真失败（整行删除）
# 依据：哪怕只有1个人需要疏散，T_0.95 也不可能 < 这个值
# 建议设为你场景中单人最短可能疏散时间的一半，留一点余量
T_MIN_VALID = 30.0      # 秒；低于此值 → Early-stop / 仿真崩溃

# ── 跨策略一致性过滤 ─────────────────────────────────────────────────────────
# 同一场景（同一行）内，任意两策略的 T 差距超过 CROSS_RATIO 倍时，
# 认为该场景至少有一个策略的结果不可信，整行删除。
# 例：某策略 T=5s，其余 T=200s → ratio=40 >> 阈值 → 删除
# 设为 None 则跳过此步骤
CROSS_RATIO_MAX = 5.0   # 最大允许的 max(T) / min(T) 比值（同一行内所有策略）

# 颜色方案
PALETTE = {
    "static":   "#4C72B0",
    "density":  "#55A868",
    "risk_01":  "#FDBF6F",
    "risk_05":  "#FF7F00",
    "risk_10":  "#C44E52",
    "risk_50":  "#9467BD",
    "risk_100": "#8C564B",
}

RECOMMENDED_BETA = "risk_10"
EXPORT_DPI = 600
FIGURE_FACE = "white"
AXES_FACE   = "white"
GRID_COLOR = "#D7DCE2"
TEXT_MUTED = "#4A5568"
SPINE_COLOR = "#A8B1BD"
BETTER_COLOR = "#2B7A78"
RISK_CURVE_COLOR = "#9B2226"
BASELINE_FILL = "#E8EDF4"


# ============================================================================
# Plotting helpers
# ============================================================================

def set_publication_style() -> None:
    """Journal-style plotting defaults shared by all figures."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": EXPORT_DPI,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
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
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.7, alpha=0.6, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(length=3.5, width=0.8, colors=TEXT_MUTED)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14, 1.06, label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111827",
    )


def format_thousands(value, _pos=None) -> str:
    return f"{value:,.0f}"


def format_beta(beta: float) -> str:
    return f"{beta:g}"


def metric_formatter(metric_key: str):
    if metric_key == "E_q95":
        return FuncFormatter(format_thousands)
    if metric_key == "H_mean":
        return FuncFormatter(lambda v, _pos: f"{v:.2f}")
    return FuncFormatter(lambda v, _pos: f"{v:.0f}")


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    save_path = Path(save_path)
    pdf_path = save_path.with_suffix(".pdf")
    png_path = save_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [fig] saved -> {pdf_path.name}, {png_path.name}")


def compute_axis_limits(values, pad: float = 0.08):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = finite.min()
    vmax = finite.max()
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.1, 1.0)
        return vmin - delta, vmax + delta
    span = vmax - vmin
    return vmin - pad * span, vmax + pad * span


def get_ci_errors(center: float, interval: tuple[float, float]):
    lo, hi = interval
    return [max(center - lo, 0.0)], [max(hi - center, 0.0)]


def add_better_arrow(ax: plt.Axes, text: str = "Better") -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0
    start = (x0 + 0.20 * dx, y0 + 0.22 * dy)
    end = (x0 + 0.06 * dx, y0 + 0.08 * dy)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=BETTER_COLOR, lw=1.8, mutation_scale=14),
    )
    ax.text(
        end[0], end[1], text,
        color=BETTER_COLOR,
        fontsize=9,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )


def build_tradeoff_points(metrics: dict, strategies: list[str], x_key: str, y_key: str):
    points = []
    for idx, strategy in enumerate(strategies):
        points.append({
            "strategy": strategy,
            "label": STRATEGY_NAMES[strategy],
            "x": metrics[strategy][x_key],
            "y": metrics[strategy][y_key],
            "x_ci": metrics[strategy].get(f"{x_key}_ci"),
            "y_ci": metrics[strategy].get(f"{y_key}_ci"),
            "color": PALETTE[strategy],
            "index": idx,
        })
    return points


# ============================================================================
# 数据加载 / 模拟
# ============================================================================

def _simulate_all_strategies(n_samples: int = 200) -> pd.DataFrame:
    """当无真实数据时生成合成数据（调试用）。"""
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    for i in range(1, n_samples + 1):
        row = {
            "sample_index": i,
            "num_g_x": rng.integers(1800, 3000),
            "num_g_y": rng.integers(1800, 3000),
            "distribution_mode_id": rng.integers(0, 4),
            "hazard_class_id": rng.integers(0, 4),
            "disaster_id": rng.integers(0, 20),
            "safe_keep_prob": rng.uniform(0.7, 1.0),
            "n_open_exits": rng.integers(6, 14),
        }
        base_t = 120 + rng.normal(0, 20)
        base_e = 800_000 + rng.normal(0, 150_000)

        # static: 最慢、风险最高
        t95_s = max(80, base_t + 80 + rng.normal(0, 15))
        row["T_0.95_static"]   = t95_s
        row["T_0.99_static"]   = t95_s * rng.uniform(1.10, 1.25)
        row["soft_risk_static"]= max(0,  base_e + 500_000 + rng.normal(0, 80_000))
        row["hard_harm_static"]= float(rng.binomial(3, 0.05))

        # density: 适中
        t95_d = max(80, base_t + 40 + rng.normal(0, 12))
        row["T_0.95_density"]   = t95_d
        row["T_0.99_density"]   = t95_d * rng.uniform(1.10, 1.22)
        row["soft_risk_density"]= max(0,  base_e + 200_000 + rng.normal(0, 70_000))
        row["hard_harm_density"]= float(rng.binomial(3, 0.03))

        # risk_* 系列: β 越大风险越低但时间效率递减
        for key, beta in BETA_VALUES.items():
            t_gain  = 20 * np.log1p(beta) / np.log1p(10.0)
            e_gain  = 150_000 * beta / (beta + 2.0)
            noise_t = rng.normal(0, 10)
            noise_e = rng.normal(0, 60_000)
            t95_r   = max(80, base_t + 40 - t_gain + noise_t)
            row[f"T_0.95_{key}"]   = t95_r
            row[f"T_0.99_{key}"]   = t95_r * rng.uniform(1.08, 1.20)
            row[f"soft_risk_{key}"]= max(0,  base_e + 200_000 - e_gain + noise_e)
            row[f"hard_harm_{key}"]= float(rng.binomial(3, max(0.001, 0.03 - 0.003 * beta)))

        rows.append(row)

    return pd.DataFrame(rows)


def _extend_partial(df_partial: pd.DataFrame) -> pd.DataFrame:
    """
    当只有 paired_samples.csv（含 static/density/risk_10）时，
    用简单线性插值 / 外推 模拟缺失策略列，仅供演示。
    真实使用时请提供包含全部策略列的完整 CSV。
    """
    df = df_partial.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)

    for metric in [TIME_METRIC, TIME_METRIC_99, EXPOSURE_METRIC, HARM_METRIC]:
        col_s  = f"{metric}_static"
        col_d  = f"{metric}_density"
        col_r10 = f"{metric}_risk_10"

        for col in [col_s, col_d, col_r10]:
            if col not in df.columns:
                df[col] = np.nan

        # 以 density 和 risk_10 之间的差距等比缩放
        diff = (df[col_d].values - df[col_r10].values) if col_d in df.columns else 0
        noise = lambda scale: rng.normal(0, scale, n)

        if f"{metric}_risk_01" not in df.columns:
            df[f"{metric}_risk_01"] = df[col_r10].values + 0.6 * diff + noise(diff.std() * 0.15 if hasattr(diff, 'std') else 1)
        if f"{metric}_risk_05" not in df.columns:
            df[f"{metric}_risk_05"] = df[col_r10].values + 0.25 * diff + noise(diff.std() * 0.10 if hasattr(diff, 'std') else 1)
        if f"{metric}_risk_50" not in df.columns:
            df[f"{metric}_risk_50"] = df[col_r10].values - 0.15 * diff + noise(diff.std() * 0.08 if hasattr(diff, 'std') else 1)
        if f"{metric}_risk_100" not in df.columns:
            df[f"{metric}_risk_100"] = df[col_r10].values - 0.22 * diff + noise(diff.std() * 0.08 if hasattr(diff, 'std') else 1)

    # hard_harm 不能为负（仅影响插值列，真实数据不受影响）
    for s in ["risk_01", "risk_05", "risk_50", "risk_100"]:
        col = f"{HARM_METRIC}_{s}"
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def filter_outliers(df: pd.DataFrame, available_strategies: list) -> pd.DataFrame:
    """
    三层行级过滤，按顺序执行，任意一层标记的行整行删除：

    Layer 1 — 物理下界（Early-stop / 仿真崩溃）
        任意策略的时间指标 < T_MIN_VALID → 视为仿真未正常完成，整行删除。

    Layer 2 — 跨策略一致性
        同一行内 max(T) / min(T) > CROSS_RATIO_MAX → 某策略结果异常，整行删除。
        (仅对 TIME_METRIC 列做检查)

    Layer 3 — 单列分布过滤（IQR / Percentile / Z-score）
        对 OUTLIER_CHECK_METRICS × available_strategies 的每列独立检测，
        超出阈值的行整行删除。

    输出: 过滤后的 DataFrame（重置 index）及打印逐层报告。
    """
    n_orig = len(df)
    bad_mask = pd.Series(False, index=df.index)
    report_lines = []
    layer_counts = {"l1": 0, "l2": 0, "l3": 0}

    # ── Layer 1: 物理下界 ─────────────────────────────────────────────────────
    t_cols = [f"{TIME_METRIC}_{s}" for s in available_strategies
              if f"{TIME_METRIC}_{s}" in df.columns]
    if t_cols and T_MIN_VALID is not None:
        below_min = df[t_cols].lt(T_MIN_VALID).any(axis=1)
        n_l1 = below_min.sum()
        if n_l1:
            report_lines.append(
                f"  [Layer 1] T < {T_MIN_VALID}s (Early-stop/crash)  "
                f"→ {n_l1} rows removed"
            )
            # 报告具体哪些列触发
            for col in t_cols:
                cnt = (df[col] < T_MIN_VALID).sum()
                if cnt:
                    report_lines.append(f"             {col}: {cnt} values below {T_MIN_VALID}s")
        layer_counts["l1"] = int((below_min & ~bad_mask).sum())
        bad_mask |= below_min

    # ── Layer 2: 跨策略一致性 ─────────────────────────────────────────────────
    if t_cols and CROSS_RATIO_MAX is not None and len(t_cols) >= 2:
        t_sub = df[t_cols].copy()
        # 只在所有列均有效（>0）时做比值检查，避免除零
        valid_rows = t_sub.gt(0).all(axis=1)
        row_max = t_sub.max(axis=1)
        row_min = t_sub.min(axis=1)
        ratio = row_max / row_min.replace(0, np.nan)
        cross_bad = valid_rows & (ratio > CROSS_RATIO_MAX)
        n_l2 = cross_bad.sum()
        if n_l2:
            report_lines.append(
                f"  [Layer 2] max(T)/min(T) > {CROSS_RATIO_MAX} (cross-strategy inconsistency)  "
                f"→ {n_l2} rows removed  "
                f"(max ratio seen: {ratio[valid_rows].max():.1f})"
            )
        layer_counts["l2"] = int((cross_bad & ~bad_mask).sum())
        bad_mask |= cross_bad

    # ── Layer 3: 单列分布过滤（IQR / Percentile / Z-score）──────────────────
    layer3_lines = []
    layer3_bad = pd.Series(False, index=df.index)
    pre_layer3_mask = bad_mask.copy()
    for metric in OUTLIER_CHECK_METRICS:
        for s in available_strategies:
            col = f"{metric}_{s}"
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if len(vals) == 0:
                continue

            if OUTLIER_METHOD == "iqr":
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lo = q1 - OUTLIER_IQR_K * iqr
                hi = q3 + OUTLIER_IQR_K * iqr

            elif OUTLIER_METHOD == "percentile":
                lo = vals.quantile(OUTLIER_PERCENTILE_LO / 100)
                hi = vals.quantile(OUTLIER_PERCENTILE_HI / 100)

            elif OUTLIER_METHOD == "zscore":
                mu, sigma = vals.mean(), vals.std()
                z = (df[col] - mu) / (sigma + 1e-12)
                col_bad = z.abs() > OUTLIER_ZSCORE_THRESH
                n_bad = int(col_bad.sum())
                if n_bad:
                    layer3_lines.append(
                        f"             {col:35s}  z>|{OUTLIER_ZSCORE_THRESH}|  "
                        f"→ {n_bad} rows  "
                        f"(keep: [{mu - OUTLIER_ZSCORE_THRESH*sigma:.1f}, "
                        f"{mu + OUTLIER_ZSCORE_THRESH*sigma:.1f}])"
                    )
                layer3_bad |= col_bad
                bad_mask |= col_bad
                continue

            else:
                raise ValueError(f"Unknown OUTLIER_METHOD: {OUTLIER_METHOD!r}")

            col_bad = (df[col] < lo) | (df[col] > hi)
            n_bad = int(col_bad.sum())
            if n_bad:
                layer3_lines.append(
                    f"             {col:35s}  [{lo:.1f}, {hi:.1f}]  → {n_bad} rows"
                )
            layer3_bad |= col_bad
            bad_mask |= col_bad

    layer_counts["l3"] = int((layer3_bad & ~pre_layer3_mask).sum())
    if layer3_lines:
        report_lines.append(
            f"  [Layer 3] Distribution filter ({OUTLIER_METHOD.upper()}"
            + (f", k={OUTLIER_IQR_K}" if OUTLIER_METHOD == "iqr" else "")
            + f")  → flagged columns:"
        )
        report_lines.extend(layer3_lines)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_removed = int(bad_mask.sum())
    df_clean = df[~bad_mask].reset_index(drop=True)

    print(f"\n  [filter] 3-layer outlier removal:")
    for line in report_lines:
        print(line)
    if not report_lines:
        print("  [filter] No outliers detected.")
    else:
        print(
            "  [filter] Layer-wise newly removed: "
            f"L1={layer_counts['l1']}, L2={layer_counts['l2']}, L3={layer_counts['l3']}"
        )
    print(f"  [filter] Total removed: {n_removed} / {n_orig} rows "
          f"({n_removed / n_orig * 100:.1f}%)  →  {len(df_clean)} rows remain")

    return df_clean




def load_data() -> pd.DataFrame:
    if DATA_FULL.exists():
        print(f"  [data] Loading full data from {DATA_FULL}")
        return pd.read_csv(DATA_FULL)

    if DATA_PARTIAL.exists():
        print(f"  [data] Loading partial data from {DATA_PARTIAL}, extending missing strategies via interpolation")
        df = pd.read_csv(DATA_PARTIAL)
        return _extend_partial(df)

    print("  [data] No data file found — generating synthetic data for debugging")
    return _simulate_all_strategies()


# ============================================================================
# 统计工具
# ============================================================================

def bootstrap_ci(
    arr: np.ndarray,
    statistic: str,
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
):
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
    elif statistic == "p95":
        estimates = np.percentile(boot, 95, axis=1)
    else:
        raise ValueError(f"Unknown bootstrap statistic: {statistic!r}")

    lo = float(np.percentile(estimates, (1 - ci) / 2 * 100))
    hi = float(np.percentile(estimates, (1 + ci) / 2 * 100))
    return lo, hi


def compute_metrics(df: pd.DataFrame, strategy: str) -> dict:
    """
    计算单一策略的四项指标:
        T_q95  — T_0.95 列的跨场景 95th 百分位（含 Bootstrap CI）
        T_q99  — T_0.99 列的跨场景 95th 百分位（含 Bootstrap CI）
        E_q95  — 运行级 soft_risk (Sigma_s) 列的跨场景 95th 百分位（含 Bootstrap CI）
        H_mean — hard_harm 列的跨场景均值（含 Bootstrap CI）
    """
    t95_col  = f"{TIME_METRIC}_{strategy}"
    t99_col  = f"{TIME_METRIC_99}_{strategy}"
    e_col    = f"{EXPOSURE_METRIC}_{strategy}"
    h_col    = f"{HARM_METRIC}_{strategy}"

    T95 = df[t95_col].dropna().to_numpy(dtype=float)
    T99 = df[t99_col].dropna().to_numpy(dtype=float) if t99_col in df.columns else T95 * 1.15
    E = df[e_col].dropna().to_numpy(dtype=float) if e_col in df.columns else np.zeros(len(T95), dtype=float)
    H = df[h_col].dropna().to_numpy(dtype=float) if h_col in df.columns else np.zeros(len(T95), dtype=float)

    if T95.size == 0:
        raise ValueError(f"No valid samples found for strategy {strategy!r}")

    t95  = float(np.percentile(T95, 95))
    t99  = float(np.percentile(T99, 95))
    e95  = float(np.percentile(E,   95))
    h_mu = float(np.mean(H))

    # 均值（用于 β 敏感性趋势图，CI 对称且稳定）
    t95_mean = float(np.mean(T95))
    t99_mean = float(np.mean(T99))
    e95_mean = float(np.mean(E))
    h_mean2  = h_mu   # same

    offset = STRATEGY_KEYS.index(strategy) * 1000
    t95_ci      = bootstrap_ci(T95, "p95",  seed=RANDOM_SEED + 11 + offset)
    t99_ci      = bootstrap_ci(T99, "p95",  seed=RANDOM_SEED + 29 + offset)
    e95_ci      = bootstrap_ci(E,   "p95",  seed=RANDOM_SEED + 47 + offset)
    h_ci        = bootstrap_ci(H,   "mean", seed=RANDOM_SEED + 71 + offset)

    t95_mean_ci = bootstrap_ci(T95, "mean", seed=RANDOM_SEED + 13 + offset)
    t99_mean_ci = bootstrap_ci(T99, "mean", seed=RANDOM_SEED + 31 + offset)
    e95_mean_ci = bootstrap_ci(E,   "mean", seed=RANDOM_SEED + 53 + offset)

    return {
        # ── 点估计（表格汇报用）──
        "T_q95":  t95,  "T_q95_ci":  t95_ci,
        "T_q99":  t99,  "T_q99_ci":  t99_ci,
        "E_q95":  e95,  "E_q95_ci":  e95_ci,
        "H_mean": h_mu, "H_mean_ci": h_ci,
        # ── 均值（β 敏感性趋势图用，CI 对称）──
        "T_mean":     t95_mean, "T_mean_ci":     t95_mean_ci,
        "T99_mean":   t99_mean, "T99_mean_ci":   t99_mean_ci,
        "E_mean":     e95_mean, "E_mean_ci":     e95_mean_ci,
        "H_mean2":    h_mean2,  "H_mean2_ci":    h_ci,
        "n":      len(T95),
    }


# ============================================================================
# 4.X.1  Baseline strategy comparison
# ============================================================================

def _legacy_plot_4x1_comparison(metrics: dict, save_path: Path):
    """
    四格面板:
        TL: T_0.95  (95th pct of per-scenario T_0.95, with Bootstrap CI)
        TR: T_0.99  (95th pct of per-scenario T_0.99, with Bootstrap CI)
        BL: Sigma_s  (95th pct of run-level population soft-risk, with Bootstrap CI)
        BR: Hard Harm  (mean hard-harm count, with Bootstrap CI)
    """
    strategies = BASELINE_STRATEGIES
    labels     = [STRATEGY_NAMES[s] for s in strategies]
    colors     = [PALETTE[s] for s in strategies]
    x          = np.arange(len(strategies))
    bar_w      = 0.55

    def _get(m, key):
        vals   = [m[s][key] for s in strategies]
        lo_ci  = [m[s][f"{key}_ci"][0] for s in strategies]
        hi_ci  = [m[s][f"{key}_ci"][1] for s in strategies]
        err_lo = [v - lo for v, lo in zip(vals, lo_ci)]
        err_hi = [hi - v for v, hi in zip(vals, hi_ci)]
        return vals, [err_lo, err_hi]

    def _bar_panel(ax, key, ylabel, title, fmt=".1f"):
        vals, errs = _get(metrics, key)
        bars = ax.bar(x, vals, bar_w, yerr=errs, capsize=6, color=colors,
                      edgecolor="black", linewidth=0.8,
                      error_kw=dict(elinewidth=1.2))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        offset = max(errs[1]) * 0.05 if max(errs[1]) > 0 else max(vals) * 0.01
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("4.X.1  Baseline Strategy Comparison",
                 fontsize=14, fontweight="bold", y=0.98)

    _bar_panel(axes[0, 0], "T_q95",
               r"$T_{0.95}$ (s)",    r"Tail evacuation time  $T_{0.95}$")
    _bar_panel(axes[0, 1], "T_q99",
               r"$T_{0.99}$ (s)",    r"Extreme evacuation time  $T_{0.99}$")
    _bar_panel(axes[1, 0], "E_q95",
               r"Population soft-risk $\Sigma_s$", r"Population exposure  $\Sigma_s$", fmt=".0f")
    _bar_panel(axes[1, 1], "H_mean",
               r"Hard harm $\bar{H}$",  r"Mean hard-harm count  $\bar{H}$", fmt=".2f")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] saved → {save_path.name}")


# ============================================================================
# 4.X.2  β sensitivity
# ============================================================================

def _legacy_plot_4x2_tradeoff(metrics: dict, save_path: Path):
    """
    T_0.95 vs population soft-risk trade-off scatter.
    β 系列连成曲线; Static / Density 作为参考点.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- reference points: Static & Density ---
    for s, marker in [("static", "s"), ("density", "D")]:
        ax.scatter(metrics[s]["E_q95"], metrics[s]["T_q95"],
                   color=PALETTE[s], s=120, marker=marker, zorder=5,
                   label=STRATEGY_NAMES[s], edgecolors="black", linewidths=0.8)

    # --- β curve ---
    beta_x = [metrics[s]["E_q95"]  for s in BETA_STRATEGIES]
    beta_y = [metrics[s]["T_q95"]  for s in BETA_STRATEGIES]
    beta_v = [BETA_VALUES[s]       for s in BETA_STRATEGIES]
    beta_c = [PALETTE[s]           for s in BETA_STRATEGIES]

    ax.plot(beta_x, beta_y, color="gray", linewidth=1.5, zorder=3, linestyle="--", alpha=0.6)
    for bx, by, bv, bc, s in zip(beta_x, beta_y, beta_v, beta_c, BETA_STRATEGIES):
        ax.scatter(bx, by, color=bc, s=110, zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(f"β={bv}", (bx, by), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5, color=bc)

    ax.set_xlabel(r"Population soft-risk  $\Sigma_s$", fontsize=12)
    ax.set_ylabel(r"Tail evacuation time  $T_{0.95}$ (s)", fontsize=12)
    ax.set_title("4.X.2  Trade-off curve: Evacuation Time vs Exposure", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)

    # 标注 Pareto 方向
    ax.annotate("", xy=(ax.get_xlim()[0]*1.01, ax.get_ylim()[0]*1.01),
                xytext=(ax.get_xlim()[0]*1.15, ax.get_ylim()[0]*1.15),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.text(ax.get_xlim()[0], ax.get_ylim()[0],
            "Better", color="green", fontsize=9, ha="left", va="bottom")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] saved → {save_path.name}")


def _legacy_plot_4x2_tradeoff_harm(metrics: dict, save_path: Path):
    """
    T_0.95 vs Hard-Harm trade-off scatter.
    布局与 plot_4x2_tradeoff 完全对称，仅将 X 轴换成 H_mean。
    β 系列连成虚线曲线；Static / Density 作为参考点。
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- reference points: Static & Density ---
    for s, marker in [("static", "s"), ("density", "D")]:
        hval = metrics[s]["H_mean"]
        tval = metrics[s]["T_q95"]
        ax.scatter(hval, tval,
                   color=PALETTE[s], s=150, marker=marker, zorder=5,
                   label=STRATEGY_NAMES[s], edgecolors="black", linewidths=0.8)
        # 标注名称，避免与 β 标注重叠
        ax.annotate(STRATEGY_NAMES[s], (hval, tval),
                    textcoords="offset points", xytext=(-8, 7),
                    fontsize=8.5, color=PALETTE[s], fontweight="bold")

    # --- β curve ---
    beta_x = [metrics[s]["H_mean"] for s in BETA_STRATEGIES]
    beta_y = [metrics[s]["T_q95"]  for s in BETA_STRATEGIES]
    beta_v = [BETA_VALUES[s]        for s in BETA_STRATEGIES]
    beta_c = [PALETTE[s]            for s in BETA_STRATEGIES]

    ax.plot(beta_x, beta_y, color="gray", linewidth=1.5,
            zorder=3, linestyle="--", alpha=0.6)

    for s, bx, by, bv, bc in zip(BETA_STRATEGIES, beta_x, beta_y, beta_v, beta_c):
        ax.scatter(bx, by, color=bc, s=120, zorder=5,
                   edgecolors="black", linewidths=0.8,
                   label=rf"D-H $\beta={bv}$")
        ax.annotate(f"β={bv}", (bx, by),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, color=bc)

    # --- CI error bars for β points (H_mean_ci on X, T_q95_ci on Y) ---
    for s in BETA_STRATEGIES:
        m = metrics[s]
        h_lo, h_hi = m["H_mean_ci"]
        t_lo, t_hi = m["T_q95_ci"]
        ax.errorbar(m["H_mean"], m["T_q95"],
                    xerr=[[m["H_mean"] - h_lo], [h_hi - m["H_mean"]]],
                    yerr=[[m["T_q95"] - t_lo],  [t_hi - m["T_q95"]]],
                    fmt="none", color=PALETTE[s], alpha=0.45,
                    capsize=4, linewidth=1.0, zorder=4)

    ax.set_xlabel(r"Mean hard-harm count  $\bar{H}$", fontsize=12)
    ax.set_ylabel(r"Tail evacuation time  $T_{0.95}$ (s)", fontsize=12)
    ax.set_title(r"4.X.2  Trade-off curve: Evacuation Time vs Hard Harm",
                 fontsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)

    # --- "Better" 箭头（左下方向：harm 少 & 时间短）---
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    arrow_x  = xlim[0] + xrange * 0.18
    arrow_y  = ylim[0] + yrange * 0.18
    ax.annotate("",
                xy=(arrow_x - xrange * 0.10, arrow_y - yrange * 0.10),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color="green",
                                lw=1.8, mutation_scale=14))
    ax.text(arrow_x - xrange * 0.11, arrow_y - yrange * 0.11,
            "Better", color="green", fontsize=9,
            ha="right", va="top")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] saved → {save_path.name}")


def _legacy_plot_4x2_diminishing(metrics: dict, save_path: Path):
    """
    四子图: T_0.95 / T_0.99 / E_0.95 / Hard Harm  vs β, 展示 diminishing returns.
    Static 和 Density 用水平参考线绘制.
    """
    beta_vals = [BETA_VALUES[s] for s in BETA_STRATEGIES]
    metric_specs = [
        ("T_q95",  r"$T_{0.95}$ (s)",          "T_q95_ci"),
        ("T_q99",  r"$T_{0.99}$ (s)",          "T_q99_ci"),
        ("E_q95",  r"$E_{0.95}$ (soft-risk)",  "E_q95_ci"),
        ("H_mean", r"Hard harm $\bar{H}$",      "H_mean_ci"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(r"4.X.2  Sensitivity to Risk Weight $\beta$", fontsize=13, fontweight="bold")

    axes_flat = axes.flatten()
    for ax, (key, ylabel, ci_key) in zip(axes_flat, metric_specs):
        # β curve
        y_vals = [metrics[s][key] for s in BETA_STRATEGIES]
        ax.plot(beta_vals, y_vals, "o-", color=PALETTE["risk_10"],
                linewidth=2, markersize=7, label=r"D-H ($\beta$)")

        # CI band if available
        if ci_key:
            lo = [metrics[s][ci_key][0] for s in BETA_STRATEGIES]
            hi = [metrics[s][ci_key][1] for s in BETA_STRATEGIES]
            ax.fill_between(beta_vals, lo, hi, alpha=0.15, color=PALETTE["risk_10"])

        # Reference lines: Static & Density
        for ref_s, ls in [("static", "--"), ("density", "-.")]:
            ref_y = metrics[ref_s][key]
            ax.axhline(ref_y, color=PALETTE[ref_s], linestyle=ls,
                       linewidth=1.5, label=STRATEGY_NAMES[ref_s], alpha=0.85)

        ax.set_xscale("log")
        ax.set_xticks(beta_vals)
        ax.set_xticklabels([str(v) for v in beta_vals], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel(r"Risk weight $\beta$", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        # 标注 diminishing returns 拐点 (最大曲率处，简单近似：最大二阶差分)
        if len(y_vals) >= 3:
            d2 = np.abs(np.diff(np.diff(y_vals)))
            elbow_idx = np.argmax(d2) + 1         # +1 因为 diff 缩短了
            ax.axvline(beta_vals[elbow_idx], color="gray",
                       linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(beta_vals[elbow_idx], ax.get_ylim()[0],
                    f" β={beta_vals[elbow_idx]}", va="bottom",
                    fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] saved → {save_path.name}")


def _legacy_plot_4x2_violin(df: pd.DataFrame, save_path: Path):
    """
    Violin / box plot: T_0.95 distribution for all 7 strategies side-by-side.
    """
    # 整理成 long-form
    records = []
    for s in STRATEGY_KEYS:
        col = f"{TIME_METRIC}_{s}"
        if col in df.columns:
            for v in df[col].dropna().values:
                records.append({"Strategy": STRATEGY_NAMES[s], "T_0.95": v,
                                "group": "risk" if s.startswith("risk") else s})
    df_long = pd.DataFrame(records)

    strategy_order = [STRATEGY_NAMES[s] for s in STRATEGY_KEYS if f"{TIME_METRIC}_{s}" in df.columns]
    palette = {STRATEGY_NAMES[s]: PALETTE[s] for s in STRATEGY_KEYS}

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(data=df_long, x="Strategy", y="T_0.95",
                   order=strategy_order, palette=palette,
                   inner="box", cut=0, linewidth=0.8, ax=ax)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel(r"$T_{0.95}$ (s)", fontsize=12)
    ax.set_title(r"Distribution of $T_{0.95}$ across All Strategies", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] saved → {save_path.name}")


# ============================================================================
# LaTeX 表格
# ============================================================================

# Per-strategy direct-label offsets (dx_pt, dy_pt, ha, va).
# Keep labels in the upper-right quadrant of each marker; small vertical
# staggering avoids collisions in the beta cluster without adding leader lines.
_LABEL_CFG: dict[str, tuple] = {
    "static":   ( 14,  22, "left", "bottom"),
    "density":  ( 18,  28, "left", "bottom"),
    "risk_01":  ( 14,  26, "left", "bottom"),
    "risk_05":  ( 14,  20, "left", "bottom"),
    "risk_10":  ( 14,  24, "left", "bottom"),
    "risk_50":  ( 14,  18, "left", "bottom"),
    "risk_100": ( 14,  24, "left", "bottom"),
}

def _direct_label(ax: plt.Axes, points: list[dict], is_beta: bool = True) -> None:
    """Place a text label beside each point without leader lines."""
    for point in points:
        s = point["strategy"]
        dx, dy, ha, va = _LABEL_CFG.get(s, (10, 6, "left", "center"))

        if is_beta and s in BETA_VALUES:
            bv = BETA_VALUES[s]
            txt = (rf"$\beta={bv:g}$" +
                   (r"$^*$" if s == RECOMMENDED_BETA else ""))
        else:
            txt = point["label"]

        fw = "semibold" if s == RECOMMENDED_BETA else "normal"
        ax.annotate(
            txt,
            xy=(point["x"], point["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha, va=va,
            fontsize=8.8,
            fontweight=fw,
            color=point["color"],
            annotation_clip=False,
            clip_on=False,
        )


def plot_4x1_comparison(metrics: dict, strategies: list[str], save_path: Path, sample_size: int):
    """Clean publication-style baseline comparison: bars + CI only, no clutter."""
    labels = [STRATEGY_NAMES[s] for s in strategies]
    colors = [PALETTE[s] for s in strategies]
    x = np.arange(len(strategies))
    metric_specs = [
        ("a", "T_q95",  r"$T_{0.95}$ (s)",    r"Tail evacuation time $T_{0.95}$"),
        ("b", "T_q99",  r"$T_{0.99}$ (s)",    r"Extreme evacuation time $T_{0.99}$"),
        ("c", "E_q95",  r"$E_{0.95}$",         r"Tail hazard exposure $E_{0.95}$"),
        ("d", "H_mean", r"$\bar{H}$",           r"Mean hard-harm count $\bar{H}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

    for ax, (panel, key, ylabel, title) in zip(axes.flat, metric_specs):
        vals   = [metrics[s][key] for s in strategies]
        lo_ci  = [metrics[s][f"{key}_ci"][0] for s in strategies]
        hi_ci  = [metrics[s][f"{key}_ci"][1] for s in strategies]
        err_lo = [v - lo for v, lo in zip(vals, lo_ci)]
        err_hi = [hi - v for v, hi in zip(vals, hi_ci)]

        ax.bar(
            x, vals,
            width=0.52,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            yerr=[err_lo, err_hi],
            capsize=3.5,
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
        ax.yaxis.set_major_formatter(metric_formatter(key))

        # tighten y-axis: start from a non-zero floor so bars don't look flat
        vmin_data = max(0.0, min(v - e for v, e in zip(vals, err_lo)))
        vmax_data = max(v + e for v, e in zip(vals, err_hi))
        span = vmax_data - vmin_data
        floor = max(0.0, vmin_data - 0.12 * span)
        ax.set_ylim(floor, vmax_data + 0.14 * span)

    fig.tight_layout(pad=1.6, h_pad=2.2, w_pad=1.8)
    save_figure(fig, save_path)


def _plot_tradeoff_core(
    metrics: dict,
    risk_strategies: list[str],
    ref_strategies: list[str],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    risk_points = build_tradeoff_points(metrics, risk_strategies, x_key, y_key)
    ref_points = build_tradeoff_points(metrics, ref_strategies, x_key, y_key)

    xs = [p["x"] for p in risk_points]
    ys = [p["y"] for p in risk_points]
    ax.plot(xs, ys, color=RISK_CURVE_COLOR, linewidth=2.0, alpha=0.9, zorder=2)

    for point in risk_points:
        marker = "P" if point["strategy"] == RECOMMENDED_BETA else "o"
        size   = 165 if point["strategy"] == RECOMMENDED_BETA else 125
        ax.scatter(
            point["x"], point["y"],
            s=size, marker=marker,
            color=point["color"],
            edgecolors="#111827", linewidths=0.9,
            zorder=4,
        )

    ref_markers = {"static": "s", "density": "D"}
    for point in ref_points:
        ax.scatter(
            point["x"],
            point["y"],
            s=140,
            marker=ref_markers.get(point["strategy"], "s"),
            color=point["color"],
            edgecolors="#111827",
            linewidths=0.9,
            zorder=5,
        )

    _direct_label(ax, risk_points, is_beta=True)
    _direct_label(ax, ref_points, is_beta=False)

    style_axes(ax, grid_axis="both")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=8)
    ax.yaxis.set_major_formatter(metric_formatter(y_key))
    ax.set_xlim(*compute_axis_limits([p["x"] for p in risk_points + ref_points], pad=0.22))
    ax.set_ylim(*compute_axis_limits([p["y"] for p in risk_points + ref_points], pad=0.18))

    # Scientific notation on x-axis
    if x_key == "E_q95":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1e-6:.2g}"))
        ax.set_xlabel(xlabel + r"  ($\times 10^{6}$)")
    elif x_key == "H_mean":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: "0" if v == 0 else f"{v * 1e-3:.2g}")
        )
        ax.set_xlabel(xlabel + r"  ($\times 10^{3}$)")
    else:
        ax.xaxis.set_major_formatter(metric_formatter(x_key))
        ax.set_xlabel(xlabel)

    fig.tight_layout(pad=1.4)
    save_figure(fig, save_path)


def plot_4x2_tradeoff(metrics: dict, beta_strategies: list[str], ref_strategies: list[str], save_path: Path):
    _plot_tradeoff_core(
        metrics,
        beta_strategies,
        ref_strategies,
        x_key="E_q95",
        y_key="T_q95",
        xlabel=r"Population soft-risk $\Sigma_s$ (RSU person step)",
        ylabel=r"Tail evacuation time $T_{0.95}$ (s)",
        title="Trade-off between evacuation time and hazard exposure",
        save_path=save_path,
    )


def plot_4x2_tradeoff_harm(metrics: dict, beta_strategies: list[str], ref_strategies: list[str], save_path: Path):
    _plot_tradeoff_core(
        metrics,
        beta_strategies,
        ref_strategies,
        x_key="H_mean",
        y_key="T_q95",
        xlabel=r"Mean hard-harm count $\bar{H}$",
        ylabel=r"Tail evacuation time $T_{0.95}$ (s)",
        title="Trade-off between evacuation time and hard harm",
        save_path=save_path,
    )


def plot_4x2_diminishing(metrics: dict, beta_strategies: list[str], ref_strategies: list[str], save_path: Path):
    beta_vals = [BETA_VALUES[s] for s in beta_strategies]
    metric_specs = [
        # 趋势图用均值（CI 对称），报告用 p95（T_q95 / E_q95）
        ("a", "T_mean",  r"$\bar{T}_{0.95}$ (s)",  r"Mean evacuation time $\bar{T}_{0.95}$"),
        ("b", "T99_mean",r"$\bar{T}_{0.99}$ (s)",  r"Mean extreme time $\bar{T}_{0.99}$"),
        ("c", "E_mean",  r"$\overline{\Sigma}_s$", r"Mean population soft-risk $\overline{\Sigma}_s$"),
        ("d", "H_mean2", r"$\bar{H}$",              r"Mean hard-harm count $\bar{H}$"),
    ]
    # 对应 CI key 后缀
    ci_suffix = {
        "T_mean":  "T_mean_ci",
        "T99_mean":"T99_mean_ci",
        "E_mean":  "E_mean_ci",
        "H_mean2": "H_mean2_ci",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True)

    for ax, (panel, key, ylabel, title) in zip(axes.flat, metric_specs):
        y_vals = [metrics[s][key] for s in beta_strategies]
        ci_key = ci_suffix[key]
        ci_lo  = [metrics[s][ci_key][0] for s in beta_strategies]
        ci_hi  = [metrics[s][ci_key][1] for s in beta_strategies]

        style_axes(ax, grid_axis="both")
        add_panel_label(ax, panel)
        ax.fill_between(beta_vals, ci_lo, ci_hi,
                        alpha=0.18, color=RISK_CURVE_COLOR, zorder=1,
                        label="_ci_band")          # labelled once in legend below
        ax.plot(beta_vals, y_vals, color=RISK_CURVE_COLOR, linewidth=1.8, zorder=3)
        ax.scatter(beta_vals, y_vals,
                   color=[PALETTE[s] for s in beta_strategies],
                   edgecolors="#2F3A4A", linewidths=0.7, s=48, zorder=4)
        ax.axvline(BETA_VALUES[RECOMMENDED_BETA], color="#A0856A",
                   linestyle=":", linewidth=1.1, alpha=0.85)

        for ref_s in ref_strategies:
            ls = "--" if ref_s == "static" else "-."
            ref_key = key.replace("T_mean", "T_mean").replace("T99_mean", "T99_mean") \
                         .replace("E_mean", "E_mean").replace("H_mean2", "H_mean")
            ax.axhline(metrics[ref_s][ref_key], color=PALETTE[ref_s],
                       linestyle=ls, linewidth=1.2, alpha=0.9)

        ax.set_xscale("log")
        ax.set_xticks(beta_vals)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6, fontsize=10.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # use E_q95 formatter for exposure, plain for times
        fmt_key = "E_q95" if "E_" in key else ("H_mean" if "H_" in key else "T_q95")
        ax.yaxis.set_major_formatter(metric_formatter(fmt_key))

    for ax in axes[1]:
        ax.set_xlabel(r"Risk weight $\beta$")

    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color=RISK_CURVE_COLOR, marker="o",
               markerfacecolor=PALETTE["risk_10"], markeredgecolor="#2F3A4A",
               markersize=6, label="Density--Hazard (mean)"),
        Patch(facecolor=RISK_CURVE_COLOR, alpha=0.25, edgecolor="none",
              label="95% bootstrap CI"),
        Line2D([0], [0], color=PALETTE["static"],  linestyle="--", label="Static"),
        Line2D([0], [0], color=PALETTE["density"], linestyle="-.", label="Density-aware"),
        Line2D([0], [0], color="#A0856A", linestyle=":",
               label=r"$\beta=1.0$ (recommended)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0), frameon=True,
               edgecolor="#D0D6DE", fontsize=8.8)
    fig.tight_layout(pad=1.6, h_pad=2.0, w_pad=1.8,
                     rect=[0, 0.07, 1, 1.0])
    save_figure(fig, save_path)


def plot_4x2_violin(df: pd.DataFrame, strategies: list[str], save_path: Path):
    records = []
    for strategy in strategies:
        col = f"{TIME_METRIC}_{strategy}"
        if col not in df.columns:
            continue
        for value in df[col].dropna().to_numpy(dtype=float):
            records.append({"Strategy": STRATEGY_NAMES[strategy], "T_0.95": value, "key": strategy})

    df_long = pd.DataFrame(records)
    if df_long.empty:
        raise ValueError("No valid T_0.95 samples are available for violin plotting.")
    order = [STRATEGY_NAMES[s] for s in strategies if f"{TIME_METRIC}_{s}" in df.columns]
    palette = {STRATEGY_NAMES[s]: PALETTE[s] for s in strategies}

    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    sns.violinplot(
        data=df_long,
        x="Strategy",
        y="T_0.95",
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0.9,
        saturation=1.0,
        ax=ax,
    )
    sns.boxplot(
        data=df_long,
        x="Strategy",
        y="T_0.95",
        order=order,
        width=0.18,
        showcaps=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="#253040", linewidth=0.9, alpha=0.95),
        medianprops=dict(color="#111827", linewidth=1.5),
        whiskerprops=dict(color="#253040", linewidth=0.9),
        capprops=dict(color="#253040", linewidth=0.9),
        ax=ax,
    )

    style_axes(ax, grid_axis="y")
    ax.set_axisbelow(True)
    ax.axvspan(-0.5, 1.5, color=BASELINE_FILL, alpha=0.35, zorder=0)
    ax.axvline(1.5, color=SPINE_COLOR, linestyle=":", linewidth=1.0, alpha=0.9)
    ax.set_xlabel("")
    ax.set_ylabel(r"$T_{0.95}$ (s)")
    ax.set_title(r"Distribution of $T_{0.95}$ across navigation strategies", loc="left", pad=8)
    ax.yaxis.set_major_formatter(metric_formatter("T_q95"))
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    medians = df_long.groupby("Strategy")["T_0.95"].median().reindex(order)
    for xpos, median in enumerate(medians.to_numpy()):
        ax.scatter(xpos, median, s=26, color="#111827", zorder=5)
        ax.text(xpos, median, f"  {median:.0f}", va="center", ha="left", fontsize=8.2, color="#111827")

    y_top = ax.get_ylim()[1]
    ax.text(0.5, y_top * 0.985, "Baselines", ha="center", va="top", fontsize=9.2, color=TEXT_MUTED)
    ax.text(4.0, y_top * 0.985, r"Density-Hazard ($\beta$ sweep)", ha="center", va="top", fontsize=9.2, color=TEXT_MUTED)

    fig.tight_layout()
    save_figure(fig, save_path)


def _legacy_generate_table_4x1(metrics: dict) -> str:
    """
    表格列: Strategy | T_0.95 [CI] | T_0.99 [CI] | E_0.95 | Hard Harm
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline strategy comparison: key metrics under joint uncertainty. "
                 r"Values in brackets are 95\,\% bootstrap confidence intervals.}")
    lines.append(r"\label{tab:baseline_comparison}")
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Strategy} & $T_{0.95}$ (s) & $T_{0.99}$ (s) & $E_{0.95}$ & $\bar{H}$ \\")
    lines.append(r"\midrule")

    for s in BASELINE_STRATEGIES:
        m    = metrics[s]
        name = STRATEGY_NAMES[s]
        t95_str = (f"{m['T_q95']:.1f} "
                   f"[{m['T_q95_ci'][0]:.1f}, {m['T_q95_ci'][1]:.1f}]")
        t99_str = (f"{m['T_q99']:.1f} "
                   f"[{m['T_q99_ci'][0]:.1f}, {m['T_q99_ci'][1]:.1f}]")
        e95_str = f"{m['E_q95']:.0f}"
        h_str   = (f"{m['H_mean']:.2f} "
                   f"[{m['H_mean_ci'][0]:.2f}, {m['H_mean_ci'][1]:.2f}]")
        lines.append(f"{name} & {t95_str} & {t99_str} & {e95_str} & {h_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _legacy_generate_table_4x2(metrics: dict) -> str:
    """
    表格列: β | T_0.95 [CI] | T_0.99 [CI] | E_0.95 | Hard Harm
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sensitivity to risk weight $\beta$: Density-Hazard strategy metrics. "
                 r"Values in brackets are 95\,\% bootstrap confidence intervals.}")
    lines.append(r"\label{tab:beta_sensitivity}")
    lines.append(r"\begin{tabular}{@{}ccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"$\beta$ & $T_{0.95}$ (s) & $T_{0.99}$ (s) & $E_{0.95}$ & $\bar{H}$ \\")
    lines.append(r"\midrule")

    for s in BETA_STRATEGIES:
        m  = metrics[s]
        bv = BETA_VALUES[s]
        t95_str = (f"{m['T_q95']:.1f} "
                   f"[{m['T_q95_ci'][0]:.1f}, {m['T_q95_ci'][1]:.1f}]")
        t99_str = (f"{m['T_q99']:.1f} "
                   f"[{m['T_q99_ci'][0]:.1f}, {m['T_q99_ci'][1]:.1f}]")
        e95_str = f"{m['E_q95']:.0f}"
        h_str   = (f"{m['H_mean']:.2f} "
                   f"[{m['H_mean_ci'][0]:.2f}, {m['H_mean_ci'][1]:.2f}]")
        lines.append(f"{bv} & {t95_str} & {t99_str} & {e95_str} & {h_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# 主程序
# ============================================================================

def _legacy_main():
    print("=" * 70)
    print("Macro-level Validation: Strategy Comparison & β Sensitivity")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"      {len(df)} scenarios loaded")

    # 确认哪些策略列实际存在
    available = [s for s in STRATEGY_KEYS
                 if f"{TIME_METRIC}_{s}" in df.columns]
    missing   = [s for s in STRATEGY_KEYS if s not in available]
    print(f"      Available strategies : {available}")
    if missing:
        print(f"      Missing strategies   : {missing}  (excluded from analysis)")

    # 1b. 离群值过滤
    print("\n[1b/5] Filtering outliers...")
    df = filter_outliers(df, available)
    if len(df) == 0:
        raise RuntimeError("All rows removed after outlier filtering — "
                           "check OUTLIER_METHOD / OUTLIER_IQR_K settings.")

    # 2. 计算所有策略指标
    print("\n[2/5] Computing metrics...")
    metrics = {}
    for s in available:
        metrics[s] = compute_metrics(df, s)
        m = metrics[s]
        print(f"      {STRATEGY_NAMES[s]:20s}: "
              f"T_q95={m['T_q95']:.1f}s  T_q99={m['T_q99']:.1f}s  "
              f"E_q95={m['E_q95']:.0f}  H_mean={m['H_mean']:.2f}")

    # 4. 绘图
    print("\n[3/5] Generating figures...")

    # 4.X.1 比较图（仅 baseline 策略）
    baseline_avail = [s for s in BASELINE_STRATEGIES if s in available]
    if len(baseline_avail) >= 2:
        m_baseline = {s: metrics[s] for s in baseline_avail}
        # 临时修改全局列表用于绘图
        _orig = BASELINE_STRATEGIES[:]
        BASELINE_STRATEGIES.clear()
        BASELINE_STRATEGIES.extend(baseline_avail)
        plot_4x1_comparison(m_baseline, baseline_avail, OUTPUT_DIR / "4x1_strategy_comparison.pdf", sample_size=len(df))
        BASELINE_STRATEGIES.clear()
        BASELINE_STRATEGIES.extend(_orig)
    else:
        print("      [skip] 4.X.1: fewer than 2 baseline strategies available")

    # 4.X.2 β trade-off (需要 static + density + 至少1个 risk_*)
    beta_avail = [s for s in BETA_STRATEGIES if s in available]
    ref_avail  = [s for s in ["static", "density"] if s in available]

    if beta_avail and ref_avail:
        plot_4x2_tradeoff(metrics, OUTPUT_DIR / "4x2_beta_tradeoff.pdf")
        plot_4x2_tradeoff_harm(metrics, OUTPUT_DIR / "4x2_beta_tradeoff_harm.pdf")

    if len(beta_avail) >= 2 and ref_avail:
        _orig_beta = BETA_STRATEGIES[:]
        BETA_STRATEGIES.clear()
        BETA_STRATEGIES.extend(beta_avail)
        plot_4x2_diminishing(metrics, OUTPUT_DIR / "4x2_beta_diminishing.pdf")
        BETA_STRATEGIES.clear()
        BETA_STRATEGIES.extend(_orig_beta)
    else:
        print("      [skip] 4.X.2 diminishing: need ≥2 risk_* strategies")

    # 全策略 violin
    plot_4x2_violin(df, OUTPUT_DIR / "4x2_all_violin.pdf")

    # 5. LaTeX 表格
    print("\n[4/5] Generating LaTeX tables...")
    tex_path = OUTPUT_DIR / "latex_macro.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by strategy_calibration_analysis.py\n\n")
        if len(baseline_avail) >= 2:
            f.write("% === 4.X.1 Baseline Strategy Comparison ===\n")
            f.write(generate_table_4x1({s: metrics[s] for s in baseline_avail}))
            f.write("\n\n")
        if len(beta_avail) >= 2:
            f.write("% === 4.X.2 Beta Sensitivity ===\n")
            f.write(generate_table_4x2({s: metrics[s] for s in beta_avail}))
    print(f"      LaTeX saved → {tex_path.name}")

    # 打印汇总
    print("\n[5/5] Summary")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}/")
    print("  4x1_strategy_comparison.pdf       — 4-panel baseline comparison")
    print("  4x2_beta_tradeoff.pdf             — T_0.95 vs E_0.95 trade-off curve")
    print("  4x2_beta_tradeoff_harm.pdf        — T_0.95 vs Hard Harm trade-off curve")
    print("  4x2_beta_diminishing.pdf          — diminishing returns (4-metric)")
    print("  4x2_all_violin.pdf           — violin distributions (all 7)")
    print("  latex_macro.tex              — ready-to-include LaTeX tables")
    print("=" * 70)


def generate_table_4x1(metrics: dict, strategies: list[str]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline strategy comparison under joint uncertainty. Values in brackets are 95\,\% bootstrap confidence intervals.}")
    lines.append(r"\label{tab:baseline_comparison}")
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Strategy} & $T_{0.95}$ (s) & $T_{0.99}$ (s) & $E_{0.95}$ & $\bar{H}$ \\")
    lines.append(r"\midrule")

    for strategy in strategies:
        m = metrics[strategy]
        name = STRATEGY_NAMES[strategy]
        t95_str = f"{m['T_q95']:.1f} [{m['T_q95_ci'][0]:.1f}, {m['T_q95_ci'][1]:.1f}]"
        t99_str = f"{m['T_q99']:.1f} [{m['T_q99_ci'][0]:.1f}, {m['T_q99_ci'][1]:.1f}]"
        e95_str = f"{m['E_q95']:.0f} [{m['E_q95_ci'][0]:.0f}, {m['E_q95_ci'][1]:.0f}]"
        h_str = f"{m['H_mean']:.2f} [{m['H_mean_ci'][0]:.2f}, {m['H_mean_ci'][1]:.2f}]"
        lines.append(f"{name} & {t95_str} & {t99_str} & {e95_str} & {h_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_table_4x2(metrics: dict, beta_strategies: list[str]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sensitivity to risk weight $\beta$ for the Density-Hazard strategy. Values in brackets are 95\,\% bootstrap confidence intervals.}")
    lines.append(r"\label{tab:beta_sensitivity}")
    lines.append(r"\begin{tabular}{@{}ccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"$\beta$ & $T_{0.95}$ (s) & $T_{0.99}$ (s) & $E_{0.95}$ & $\bar{H}$ \\")
    lines.append(r"\midrule")

    for strategy in beta_strategies:
        m = metrics[strategy]
        beta = format_beta(BETA_VALUES[strategy])
        t95_str = f"{m['T_q95']:.1f} [{m['T_q95_ci'][0]:.1f}, {m['T_q95_ci'][1]:.1f}]"
        t99_str = f"{m['T_q99']:.1f} [{m['T_q99_ci'][0]:.1f}, {m['T_q99_ci'][1]:.1f}]"
        e95_str = f"{m['E_q95']:.0f} [{m['E_q95_ci'][0]:.0f}, {m['E_q95_ci'][1]:.0f}]"
        h_str = f"{m['H_mean']:.2f} [{m['H_mean_ci'][0]:.2f}, {m['H_mean_ci'][1]:.2f}]"
        lines.append(f"{beta} & {t95_str} & {t99_str} & {e95_str} & {h_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    set_publication_style()

    print("=" * 72)
    print("Macro-level Validation: Strategy Comparison and Risk-weight Sensitivity")
    print("=" * 72)

    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"      {len(df)} scenarios loaded")

    available = [s for s in STRATEGY_KEYS if f"{TIME_METRIC}_{s}" in df.columns]
    missing = [s for s in STRATEGY_KEYS if s not in available]
    print(f"      Available strategies : {available}")
    if missing:
        print(f"      Missing strategies   : {missing}  (excluded from analysis)")

    print("\n[1b/5] Filtering outliers...")
    df = filter_outliers(df, available)
    if len(df) == 0:
        raise RuntimeError("All rows removed after outlier filtering; check the outlier settings.")

    print("\n[2/5] Computing metrics...")
    metrics = {}
    for strategy in available:
        metrics[strategy] = compute_metrics(df, strategy)
        m = metrics[strategy]
        print(
            f"      {STRATEGY_NAMES[strategy]:20s}: "
            f"T_q95={m['T_q95']:.1f}s  T_q99={m['T_q99']:.1f}s  "
            f"E_q95={m['E_q95']:.0f}  H_mean={m['H_mean']:.2f}"
        )

    print("\n[3/5] Generating figures...")
    baseline_avail = [s for s in BASELINE_STRATEGIES if s in available]
    beta_avail = [s for s in BETA_STRATEGIES if s in available]
    ref_avail = [s for s in ["static", "density"] if s in available]

    if len(baseline_avail) >= 2:
        plot_4x1_comparison(
            {s: metrics[s] for s in baseline_avail},
            baseline_avail,
            OUTPUT_DIR / "4x1_strategy_comparison.pdf",
            sample_size=len(df),
        )
    else:
        print("      [skip] 4.X.1: fewer than 2 baseline strategies available")

    if beta_avail and ref_avail:
        plot_4x2_tradeoff(metrics, beta_avail, ref_avail, OUTPUT_DIR / "4x2_beta_tradeoff.pdf")
        plot_4x2_tradeoff_harm(metrics, beta_avail, ref_avail, OUTPUT_DIR / "4x2_beta_tradeoff_harm.pdf")
    else:
        print("      [skip] 4.X.2 trade-off: need baseline references and at least one beta strategy")

    if len(beta_avail) >= 2 and ref_avail:
        plot_4x2_diminishing(metrics, beta_avail, ref_avail, OUTPUT_DIR / "4x2_beta_diminishing.pdf")
    else:
        print("      [skip] 4.X.2 diminishing: need at least two beta strategies")

    plot_4x2_violin(df, available, OUTPUT_DIR / "4x2_all_violin.pdf")

    print("\n[4/5] Generating LaTeX tables...")
    tex_path = OUTPUT_DIR / "latex_macro.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by strategy_calibration_analysis.py\n\n")
        if len(baseline_avail) >= 2:
            f.write("% === 4.X.1 Baseline Strategy Comparison ===\n")
            f.write(generate_table_4x1(metrics, baseline_avail))
            f.write("\n\n")
        if len(beta_avail) >= 2:
            f.write("% === 4.X.2 Beta Sensitivity ===\n")
            f.write(generate_table_4x2(metrics, beta_avail))
    print(f"      LaTeX saved -> {tex_path.name}")

    print("\n[5/5] Summary")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIR}/")
    print("  4x1_strategy_comparison.pdf / .png   baseline comparison")
    print("  4x2_beta_tradeoff.pdf / .png         time-exposure trade-off")
    print("  4x2_beta_tradeoff_harm.pdf / .png    time-hard-harm trade-off")
    print("  4x2_beta_diminishing.pdf / .png      beta sensitivity panels")
    print("  4x2_all_violin.pdf / .png            T_0.95 distribution figure")
    print("  latex_macro.tex                      LaTeX-ready summary tables")
    print("=" * 72)


if __name__ == "__main__":
    main()
