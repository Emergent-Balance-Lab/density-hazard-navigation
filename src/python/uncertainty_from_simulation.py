"""
Uncertainty Analysis Script for Real Evacuation Project Outputs
==============================================================

从真实项目目录读取 Monte Carlo 实验结果，并结合 input/project_random_inputs.csv
中的随机输入变量，完成不确定性分析。

支持的三类策略：
- static
- risk_01
- risk_10

项目目录示例：
    ../../results/analysis/static_1/metrics_summary.json
    ../../results/analysis/risk_01_1/metrics_summary.json
    ../../results/analysis/risk_10_1/metrics_summary.json

输入文件示例：
    ../input/project_random_inputs.csv

依赖:
    numpy, pandas, scipy, matplotlib, seaborn
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================================
# 配置参数
# ============================================================================

# 路径
INPUT_CSV = Path("../../data/simulation_inputs/project_random_inputs.csv")
OUTPUT_ROOT = Path("../../results/simulation")              # 每个项目目录所在根目录
RESULTS_DIR = Path("../../results/analysis/uncertainty")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── 手动配置：参与分析的策略 ─────────────────────────────────────────
# key   = 图表/表格中显示的名称
# value = 输出目录前缀（必须与 output/<prefix>_N 目录名一致）
# 删除不存在的策略，添加新策略时照此格式追加即可
STRATEGY_PREFIX = {
    "Static":  "static",
    "Density": "density",
    "Risk-10": "risk_10",
}

# 基准策略的显示名（必须是 STRATEGY_PREFIX 的某个 key），用于配对比较和 CVaR 折减计算
STRATEGY_BASELINE = "Static"

# 每个策略的颜色和线型（自动补全，不够时循环）
STRATEGY_COLORS     = ["#4C72B0", "#C44E52", "#55A868", "#DD8452", "#937860"]
STRATEGY_LINESTYLES = ["--", "-", "-.", ":", (0, (3, 1, 1, 1))]

# ─── 手动指定要对比的样本序号范围（闭区间，整数）─────────────────────
# 修改这两个值来控制分析哪些样本；设为 None 表示不限制（自动扫描全部）
SAMPLE_INDEX_MIN: int = 1
SAMPLE_INDEX_MAX: int = 20   # 例如改为 50 则只取序号 1~50

# ─── 多指标分析控制列表 ───────────────────────────────────────────────
# 在此列表中增删指标即可控制分析范围
# 可用的指标名称（来自 metrics_summary.json）：
#   时间效率：T_0.90 / T_0.95 / T_0.99
#   风险伤害：soft_risk / hard_harm
#   行程统计：distance_mean / route_switching_mean
#   空间层指标：layer7_q95(使用率) / layer8_q95(拥挤度) / layer9_q95(密度)
#               layer14_q95(疏散时间场) / layer17_mean(出口使用率)
#   高风险面积：dens_area_ge_q95_m2 / cong_area_ge_q95_m2 / usage_area_ge_q95_m2
TARGET_METRICS = [
    "T_0.95",
    "soft_risk",
    "hard_harm",
    "distance_mean",
    "route_switching_mean",
    "layer8_q95",
    "layer9_q95",
]

# 各指标的显示名称（用于图表标题和 LaTeX 表格）
METRIC_DISPLAY_NAMES = {
    "T_0.90":                 r"$T_{0.90}$ (s)",
    "T_0.95":                 r"$T_{0.95}$ (s)",
    "T_0.99":                 r"$T_{0.99}$ (s)",
    "soft_risk":              "Soft Risk",
    "hard_harm":              "Hard Harm",
    "distance_mean":          "Mean Travel Distance (m)",
    "route_switching_mean":   "Mean Route Switching",
    "layer7_q95":             "Usage Rate $q_{95}$",
    "layer8_q95":             "Congestion $q_{95}$",
    "layer9_q95":             "Density $q_{95}$",
    "layer14_q95":            "Evacuation Time Field $q_{95}$ (s)",
    "layer17_mean":           "Exit Usage Mean",
    "dens_area_ge_q95_m2":    r"High-Density Area $\geq q_{95}$ (m²)",
    "cong_area_ge_q95_m2":    r"Congested Area $\geq q_{95}$ (m²)",
    "usage_area_ge_q95_m2":   r"High-Usage Area $\geq q_{95}$ (m²)",
}

# 主要时间指标（仅用于可靠性曲线和超越概率）
TARGET_TIME_METRIC = "T_0.95"

# 可靠性曲线阈值范围（会根据真实数据自适应覆盖）
TAU_POINTS = 60

# 超越概率关键阈值（仅对时间指标使用）
CRITICAL_THRESHOLDS = [1200, 1500, 1800]

# Bootstrap
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 参数名称（画图时显示）
PARAM_DISPLAY_NAMES = {
    "num_g": "Population $N$",
    "distribution_mode_id": "Initial distribution",
    "hazard_class_id": "Hazard class",
    "disaster_id": "Ignition location ID",
    "disaster_x": "Ignition $x$",
    "disaster_y": "Ignition $y$",
    "safe_keep_prob": "Safe keep prob.",
    "n_open_exits": "Open exits",
}

# 默认用于敏感性分析的输入列
DEFAULT_PARAM_COLUMNS = [
    "num_g",
    "distribution_mode_id",
    "hazard_class_id",
    "disaster_id",
    "disaster_x",
    "disaster_y",
    "safe_keep_prob",
    "n_open_exits",
]

# ============================================================================
# 工具函数
# ============================================================================

def compute_cvar(data, alpha=0.95):
    """计算 CVaR"""
    data = np.asarray(data, dtype=float).ravel()
    var = np.percentile(data, alpha * 100)
    tail = data[data >= var]
    return float(np.mean(tail)) if len(tail) > 0 else np.nan


def compute_var(data, alpha=0.95):
    """计算 VaR"""
    data = np.asarray(data, dtype=float).ravel()
    return float(np.percentile(data, alpha * 100))


def bootstrap_ci(data, statistic_func, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """Bootstrap 置信区间"""
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    boot_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic_func(sample))

    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def reliability_curve(T, tau_values):
    """经验可靠性函数 R(tau)=P(T<=tau)"""
    T = np.asarray(T, dtype=float)
    return np.array([np.mean(T <= tau) for tau in tau_values], dtype=float)


def reliability_curve_bootstrap(T, tau_values, n_bootstrap=N_BOOTSTRAP):
    """带 bootstrap 置信带的可靠性曲线"""
    T = np.asarray(T, dtype=float)
    n = len(T)

    R_mean = reliability_curve(T, tau_values)
    R_boots = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(T, size=n, replace=True)
        R_boots.append(reliability_curve(sample, tau_values))

    R_boots = np.asarray(R_boots)
    R_lower = np.percentile(R_boots, 2.5, axis=0)
    R_upper = np.percentile(R_boots, 97.5, axis=0)
    return R_mean, R_lower, R_upper


def parse_project_name(project_name: str):
    """
    解析项目名，如：
        static_1
        risk_01_1
        risk_10_1
    返回:
        strategy_key, sample_index
    """
    prefixes = "|".join(re.escape(p) for p in STRATEGY_PREFIX.values())
    m = re.match(rf"^({prefixes})_(\d+)$", project_name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def read_metrics_json(json_path: Path):
    """读取单个 metrics_summary.json"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    time_eff = data.get("time_efficiency", {})
    spatial_eff = data.get("spatial_efficiency", {})
    run_record_stats = data.get("run_record_statistics", {})
    risk_all = spatial_eff.get("risk_all", {})

    out = {
        "project": data.get("project"),
        "T_0.90": time_eff.get("T_0.90"),
        "T_0.95": time_eff.get("T_0.95"),
        "T_0.99": time_eff.get("T_0.99"),
        "total_n": time_eff.get("total_n"),
        "t_end": time_eff.get("t_end"),
        "alive_end": time_eff.get("alive_end"),
        "soft_risk": risk_all.get("soft_risk"),
        "hard_harm": risk_all.get("hard_harm"),
        "distance_mean": ((run_record_stats.get("distance") or {}).get("mean")),
        "route_switching_mean": ((run_record_stats.get("route_switching_count") or {}).get("mean")),
    }

    # 抽取 layer_7 / 8 / 9 / 14 / 17 的常用指标
    layers = spatial_eff.get("layers", [])
    selected_layer_ids = spatial_eff.get("selected_layer_ids", [])
    layer_map = {}

    for lid, layer in zip(selected_layer_ids, layers):
        layer_map[lid] = layer

    def safe_layer_val(layer_id, key):
        layer = layer_map.get(layer_id, {})
        return layer.get(key)

    # 常用空间指标
    out.update({
        "layer7_q95": safe_layer_val(7, "q95"),
        "layer8_q95": safe_layer_val(8, "q95"),
        "layer9_q95": safe_layer_val(9, "q95"),
        "layer14_q95": safe_layer_val(14, "q95"),
        "layer17_mean": safe_layer_val(17, "mean"),
        "dens_area_ge_q95_m2": safe_layer_val(9, "area_ge_q95"),
        "cong_area_ge_q95_m2": safe_layer_val(8, "area_ge_q95"),
        "usage_area_ge_q95_m2": safe_layer_val(7, "area_ge_q95"),
    })

    return out


# ============================================================================
# 数据加载
# ============================================================================

def load_real_data(
    input_csv=INPUT_CSV,
    output_root=OUTPUT_ROOT,
    target_metrics=None,
    param_columns=None,
    sample_index_min=SAMPLE_INDEX_MIN,
    sample_index_max=SAMPLE_INDEX_MAX,
):
    """
    读取真实数据并对齐为多策略配对样本，同时提取多个指标。

    返回：
        merged_df         : 已配对好的总表
        params            : 参数矩阵 (n_samples × n_params)
        all_metric_arrays : {metric_name: {strategy_display_name: 1D ndarray}}
        param_columns     : 实际使用的参数列
    """
    if target_metrics is None:
        target_metrics = TARGET_METRICS
    if param_columns is None:
        param_columns = DEFAULT_PARAM_COLUMNS.copy()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # CSV 无表头，手动指定列名
    _CSV_COLUMNS = [
        "project_name", "sample_index", "run_mode", "global_seed", "sample_seed",
        "num_g", "distribution_mode_id", "distribution_mode_name",
        "hazard_class_id", "hazard_class_name",
        "disaster_id", "disaster_x", "disaster_y",
        "safe_keep_prob", "n_open_exits", "n_total_exits",
        "exit_switches_json", "exit_switch_dict_json", "num_genes",
        "label_values_json",
        "label_0", "label_1", "label_2", "label_3", "label_4",
        "label_6", "label_7", "label_8", "label_9", "label_10",
        "label_11", "label_12", "label_13", "label_14",
    ]

    # 先读一行探测实际列数，再决定是否使用手动列名
    _probe = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=1, header=None)
    _n_cols = _probe.shape[1]

    _probe_with_header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    _has_header = "project_name" in [c.strip() for c in _probe_with_header.columns]

    if _has_header:
        df_input = pd.read_csv(input_csv, encoding="utf-8-sig")
        df_input.columns = df_input.columns.str.strip()
    else:
        # 无表头：按实际列数截取或补齐列名列表
        col_names = _CSV_COLUMNS[:_n_cols] + [f"_extra_{i}" for i in range(_n_cols - len(_CSV_COLUMNS))]
        df_input = pd.read_csv(input_csv, encoding="utf-8-sig", header=None, names=col_names)

    if "project_name" not in df_input.columns:
        raise ValueError(
            f"'project_name' column not found in {input_csv}.\n"
            f"Actual columns: {list(df_input.columns)}"
        )

    # 只保留配置中指定的策略
    valid_prefixes = set(STRATEGY_PREFIX.values())
    df_input["_strategy_prefix"], df_input["_sample_index_from_name"] = zip(
        *df_input["project_name"].map(parse_project_name)
    )
    df_input = df_input[df_input["_strategy_prefix"].isin(valid_prefixes)].copy()

    # 若 sample_index 列存在，优先用 sample_index；否则回退用 project_name 解析出的编号
    if "sample_index" not in df_input.columns:
        df_input["sample_index"] = df_input["_sample_index_from_name"]
    else:
        df_input["sample_index"] = df_input["sample_index"].fillna(df_input["_sample_index_from_name"])

    # 扫描 output_root 下所有 metrics_summary.json
    metric_rows = []
    for project_dir in output_root.iterdir():
        if not project_dir.is_dir():
            continue

        strategy_prefix, sample_idx = parse_project_name(project_dir.name)
        if strategy_prefix not in valid_prefixes:
            continue

        # ── 手动序号过滤 ──────────────────────────────────────────────
        if sample_index_min is not None and sample_idx < sample_index_min:
            continue
        if sample_index_max is not None and sample_idx > sample_index_max:
            continue

        json_path = project_dir / "metrics_summary.json"
        if not json_path.exists():
            continue

        row = read_metrics_json(json_path)
        row["project_name"] = project_dir.name
        row["strategy_prefix"] = strategy_prefix
        row["sample_index"] = sample_idx
        metric_rows.append(row)

    if not metric_rows:
        raise RuntimeError(f"No metrics_summary.json found under: {output_root}")

    df_metrics = pd.DataFrame(metric_rows)

    # 从 input 中拿随机变量，并与 metrics 对齐
    keep_input_cols = ["project_name", "sample_index"] + [
        c for c in param_columns if c in df_input.columns
    ]
    
    df_input_small = df_input[keep_input_cols].copy()

    df_all = pd.merge(
        df_metrics,
        df_input_small,
        on=["project_name", "sample_index"],
        how="left",
        suffixes=("", "_input"),
    )

    # 若 total_n 缺失则用 num_g
    if "num_g" in df_all.columns:
        df_all["population"] = df_all["num_g"]
    else:
        df_all["population"] = df_all["total_n"]

    if "num_g" not in df_all.columns:
        df_all["num_g"] = df_all["total_n"]

    # pivot 值列 = 所有 TARGET_METRICS + 其他常用指标（取交集）
    all_possible_value_cols = list(dict.fromkeys(
        target_metrics + [
            "T_0.90", "T_0.95", "T_0.99",
            "soft_risk", "hard_harm",
            "distance_mean", "route_switching_mean",
            "layer7_q95", "layer8_q95", "layer9_q95", "layer14_q95", "layer17_mean",
            "dens_area_ge_q95_m2", "cong_area_ge_q95_m2", "usage_area_ge_q95_m2",
        ]
    ))
    value_cols = [c for c in all_possible_value_cols if c in df_all.columns]

    df_pivot_src = df_all[["strategy_prefix", "sample_index"] + value_cols].copy()

    # ── 诊断：打印各策略的样本分布 ──────────────────────────────────────
    print(f"     [diag] df_all rows per strategy:")
    for pfx in STRATEGY_PREFIX.values():
        sub = df_all[df_all["strategy_prefix"] == pfx]
        print(f"       {pfx}: {len(sub)} rows, sample_index range: "
              f"{sub['sample_index'].min()} ~ {sub['sample_index'].max()}")

    # pivot：index=sample_index，columns=strategy_prefix，values=各指标
    pivot = df_pivot_src.pivot_table(
        index=["sample_index"],
        columns="strategy_prefix",
        values=value_cols,
        aggfunc="first",
    )

    pivot.columns = [f"{metric}_{strategy}" for metric, strategy in pivot.columns]
    pivot = pivot.reset_index()

    # 把 num_g 从第一个策略行拼回来（用于后续参数列）
    first_pfx = list(STRATEGY_PREFIX.values())[0]
    if "num_g" in df_all.columns:
        num_g_map = (
            df_all[df_all["strategy_prefix"] == first_pfx][["sample_index", "num_g"]]
            .drop_duplicates("sample_index")
        )
        pivot = pd.merge(pivot, num_g_map, on="sample_index", how="left")

    # 取基准策略行上的输入变量，作为每个样本的随机参数
    # num_g 已由 num_g_map 拼入 pivot，此处排除，避免 merge 后产生 num_g_x/num_g_y
    baseline_pfx = STRATEGY_PREFIX[STRATEGY_BASELINE]
    param_cols_for_input = [
        c for c in param_columns
        if c in df_all.columns and c != "num_g"
    ]
    static_cols = list(dict.fromkeys(["sample_index"] + param_cols_for_input))
    df_static_input = (
        df_all[df_all["strategy_prefix"] == baseline_pfx][static_cols]
        .drop_duplicates(subset=["sample_index"])
    )

    merged_df = pd.merge(pivot, df_static_input, on=["sample_index"], how="left")

    # 用 TARGET_METRICS 中第一个可用指标做 dropna（保证三策略都有值）
    primary_metric = next((m for m in target_metrics if m in df_all.columns), None)
    if primary_metric is None:
        raise RuntimeError(f"TARGET_METRICS 中没有任何指标存在于数据中：{target_metrics}")

    need_cols = [f"{primary_metric}_{pfx}" for pfx in STRATEGY_PREFIX.values()]
    missing = [c for c in need_cols if c not in merged_df.columns]
    if missing:
        raise RuntimeError(
            f"以下列在 pivot 后不存在，无法继续分析。\n"
            f"  缺失列: {missing}\n"
            f"  merged_df 实际列: {list(merged_df.columns)}\n"
            f"请检查 STRATEGY_PREFIX 配置与输出目录是否一致。"
        )

    merged_df = merged_df.dropna(subset=need_cols).copy()

    # 去掉重复列名
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()

    # 仅保留真实存在的参数列（num_g 来自 pivot 里的 num_g_map，应已存在）
    param_columns = [c for c in param_columns if c in merged_df.columns]
    if "num_g" not in param_columns and "num_g" in merged_df.columns:
        param_columns = ["num_g"] + param_columns
    params = merged_df[param_columns].to_numpy(dtype=float)

    # 构建多指标数组字典：{metric: {strategy_display_name: 1D ndarray}}
    all_metric_arrays = {}
    for metric in target_metrics:
        metric_arrays = {}
        for display_name, pfx in STRATEGY_PREFIX.items():
            col = f"{metric}_{pfx}"
            if col in merged_df.columns:
                metric_arrays[display_name] = merged_df[col].to_numpy(dtype=float).ravel()
        if metric_arrays:
            all_metric_arrays[metric] = metric_arrays

    return merged_df, params, all_metric_arrays, param_columns


# ============================================================================
# 统计分析
# ============================================================================

def analyze_basic_statistics(T_arrays):
    results = {}

    for name, T in T_arrays.items():
        mean_val = np.mean(T)
        std_val = np.std(T, ddof=1)
        var_95 = compute_var(T)
        cvar_95 = compute_cvar(T)
        cvar_ci = bootstrap_ci(T, compute_cvar)

        results[name] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "cvar_ci_lower": cvar_ci[0],
            "cvar_ci_upper": cvar_ci[1],
        }

    return results


def analyze_exceedance(T_arrays, thresholds=CRITICAL_THRESHOLDS):
    results = {}

    for name, T in T_arrays.items():
        results[name] = {}
        for tau in thresholds:
            exc_prob = np.mean(T > tau)

            def exc_func(x):
                return np.mean(x > tau)

            ci = bootstrap_ci(T, exc_func)

            results[name][tau] = {
                "prob": float(exc_prob),
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }

    return results


def analyze_paired_comparison(T_arrays):
    """将所有非基准策略与 STRATEGY_BASELINE 做配对比较。"""
    results = {}
    T_base = T_arrays[STRATEGY_BASELINE]

    for name, T in T_arrays.items():
        if name == STRATEGY_BASELINE:
            continue
        key = f"{name}_vs_{STRATEGY_BASELINE}"
        delta = T_base - T
        results[key] = {
            "mean_improvement": float(np.mean(delta)),
            "ci": bootstrap_ci(delta, np.mean),
            "pct_better": float(np.mean(delta > 0) * 100),
            "wilcoxon_stat": stats.wilcoxon(T_base, T, alternative="greater"),
        }

    return results


def analyze_sensitivity(params, param_columns, T_arrays):
    results = {}

    for name, T in T_arrays.items():
        correlations = []
        pvalues = []
        for i in range(params.shape[1]):
            rho, p = stats.spearmanr(params[:, i], T)
            correlations.append(float(rho))
            pvalues.append(float(p))
        results[name] = {
            "rho": correlations,
            "pvalue": pvalues,
            "param_columns": param_columns,
        }

    return results


# ============================================================================
# 可视化
# ============================================================================

def get_tau_range(T_arrays):
    all_T = np.concatenate(list(T_arrays.values()))
    lo = max(0, np.floor(np.min(all_T) / 20) * 20 - 40)
    hi = np.ceil(np.max(all_T) / 20) * 20 + 40
    return np.linspace(lo, hi, TAU_POINTS)


def plot_reliability_curves(T_arrays, save_path=None):
    tau_range = get_tau_range(T_arrays)

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(T_arrays.keys())
    colors    = {n: STRATEGY_COLORS[i % len(STRATEGY_COLORS)]     for i, n in enumerate(names)}
    linestyles = {n: STRATEGY_LINESTYLES[i % len(STRATEGY_LINESTYLES)] for i, n in enumerate(names)}

    for name, T in T_arrays.items():
        R_mean, R_lower, R_upper = reliability_curve_bootstrap(T, tau_range)
        ax.plot(
            tau_range, R_mean,
            linestyle=linestyles[name],
            color=colors[name],
            linewidth=2.5,
            label=name,
        )
        ax.fill_between(tau_range, R_lower, R_upper, alpha=0.20, color=colors[name])

    ax.set_xlabel(r"Time threshold $\tau$ (s)", fontsize=12)
    ax.set_ylabel(r"Reliability $R(\tau)=P(T \leq \tau)$", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_title(f"Reliability Curves ({TARGET_TIME_METRIC})", fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()


def plot_cvar_comparison(basic_stats, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = list(STRATEGY_PREFIX.keys())
    cvars = [basic_stats[s]["cvar_95"] for s in strategies]
    errors_lower = [basic_stats[s]["cvar_95"] - basic_stats[s]["cvar_ci_lower"] for s in strategies]
    errors_upper = [basic_stats[s]["cvar_ci_upper"] - basic_stats[s]["cvar_95"] for s in strategies]

    colors = [STRATEGY_COLORS[i % len(STRATEGY_COLORS)] for i in range(len(strategies))]
    bars = ax.bar(
        strategies, cvars,
        yerr=[errors_lower, errors_upper],
        capsize=8,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_ylabel(rf"$\mathrm{{CVaR}}_{{0.95}}({TARGET_TIME_METRIC})$ (s)", fontsize=12)
    ax.set_title("Tail Risk Comparison", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, cvars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max(cvars),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()


def plot_sensitivity_heatmap(sensitivity_results, param_columns, save_path=None):
    fig, ax = plt.subplots(figsize=(8, max(5, 0.55 * len(param_columns))))

    strategies = list(STRATEGY_PREFIX.keys())
    corr_matrix = np.array([sensitivity_results[s]["rho"] for s in strategies]).T

    row_names = [PARAM_DISPLAY_NAMES.get(c, c) for c in param_columns]
    df = pd.DataFrame(
        corr_matrix,
        index=row_names,
        columns=strategies,
    )

    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(f"Sensitivity Analysis: Spearman Correlations vs {TARGET_TIME_METRIC}", fontsize=14)
    ax.set_ylabel("Input Parameter", fontsize=12)
    ax.set_xlabel("Strategy", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()


def plot_improvement_histogram(T_arrays, save_path=None):
    """对所有非基准策略绘制改进量分布直方图（子图并排）。"""
    T_base = T_arrays[STRATEGY_BASELINE]
    compare_names = [n for n in T_arrays if n != STRATEGY_BASELINE]

    fig, axes = plt.subplots(1, max(len(compare_names), 1),
                             figsize=(7 * max(len(compare_names), 1), 5),
                             squeeze=False)

    for ax, name in zip(axes[0], compare_names):
        T = T_arrays[name]
        delta = T_base - T
        color = STRATEGY_COLORS[(list(T_arrays.keys()).index(name)) % len(STRATEGY_COLORS)]

        ax.hist(delta, bins=30, color=color, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(delta), color="black", linestyle="--", linewidth=2,
                   label=f"Mean: {np.mean(delta):.2f}s")
        ax.axvline(0, color="gray", linestyle="-", linewidth=1.5)

        ax.set_xlabel(
            rf"Improvement $\Delta = T_{{\mathrm{{{STRATEGY_BASELINE}}}}} - T_{{\mathrm{{{name}}}}}$ (s)",
            fontsize=12,
        )
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{name} vs {STRATEGY_BASELINE}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        pct_positive = np.mean(delta > 0) * 100
        ax.text(
            0.95, 0.95, f"{pct_positive:.1f}% scenarios improved",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.suptitle("Distribution of Pairwise Improvement", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()


def plot_tornado_diagram(params, param_columns, T_arrays, save_path=None):
    n_strat = len(T_arrays)
    fig, axes = plt.subplots(
        1, n_strat,
        figsize=(7 * n_strat, max(5, 0.45 * len(param_columns))),
        sharey=True,
    )
    if n_strat == 1:
        axes = [axes]

    strategies = list(T_arrays.items())
    colors = [STRATEGY_COLORS[i % len(STRATEGY_COLORS)] for i in range(n_strat)]

    for ax, (name, T), color in zip(axes, strategies, colors):
        sensitivities = []

        for i in range(params.shape[1]):
            rho, _ = stats.spearmanr(params[:, i], T)
            sens = abs(rho) * np.std(T)
            sensitivities.append(float(sens))

        sorted_idx = np.argsort(sensitivities)[::-1]
        sorted_names = [PARAM_DISPLAY_NAMES.get(param_columns[i], param_columns[i]) for i in sorted_idx]
        sorted_sens = [sensitivities[i] for i in sorted_idx]

        ax.barh(range(len(sorted_sens)), sorted_sens, color=color, edgecolor="black")
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Sensitivity (a.u.)", fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

    plt.suptitle("Tornado Diagram: Parameter Sensitivity Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()


# ============================================================================
# LaTeX 表格
# ============================================================================

def generate_latex_table_basic(basic_stats):
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Tail risk comparison across navigation strategies.}
\label{tab:cvar_results}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Strategy} & \textbf{Mean} (s) & $\mathbf{VaR_{0.95}}$ (s) & $\mathbf{CVaR_{0.95}}$ (s) & \textbf{Reduction} \\
\midrule
"""
    static_cvar = basic_stats[STRATEGY_BASELINE]["cvar_95"]

    for name in STRATEGY_PREFIX.keys():
        s = basic_stats[name]
        if name == STRATEGY_BASELINE or static_cvar == 0 or np.isnan(static_cvar):
            reduction_str = "—"
        else:
            reduction = (1 - s["cvar_95"] / static_cvar) * 100
            reduction_str = f"{reduction:.1f}\\%"

        latex += (
            f"{name} & {s['mean']:.4f} & {s['var_95']:.4f} & "
            f"{s['cvar_95']:.4f} [{s['cvar_ci_lower']:.4f}, {s['cvar_ci_upper']:.4f}] & "
            f"{reduction_str} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_latex_table_exceedance(exc_results):
    taus = list(CRITICAL_THRESHOLDS)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Exceedance probabilities $P(T > \tau^*)$ at critical thresholds.}
\label{tab:exceedance}
\begin{tabular}{@{}l""" + "c" * len(taus) + r"""@{}}
\toprule
\textbf{Strategy}
"""
    for tau in taus:
        latex += f" & $\\tau^*={tau}$ s"
    latex += r""" \\
\midrule
"""

    for name in STRATEGY_PREFIX.keys():
        row = name
        for tau in taus:
            e = exc_results[name][tau]
            row += f" & {e['prob']:.3f} [{e['ci_lower']:.3f}, {e['ci_upper']:.3f}]"
        latex += row + r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_latex_table_sensitivity(sensitivity_results, param_columns):
    strat_names = list(STRATEGY_PREFIX.keys())
    col_spec = "c" * len(strat_names)
    header_cols = " & ".join(f"\\textbf{{{n}}}" for n in strat_names)

    latex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Spearman correlation coefficients between uncertain inputs and evacuation time.}}
\label{{tab:sensitivity}}
\begin{{tabular}}{{@{{}}l{col_spec}@{{}}}}
\toprule
\textbf{{Parameter}} & {header_cols} \\
\midrule
"""
    base_rho = sensitivity_results[STRATEGY_BASELINE]["rho"]

    for i, col in enumerate(param_columns):
        label = PARAM_DISPLAY_NAMES.get(col, col)
        vals = []
        for name in strat_names:
            rho = sensitivity_results[name]["rho"][i]
            if name != STRATEGY_BASELINE and abs(rho) < abs(base_rho[i]) * 0.7:
                vals.append(f"\\textbf{{{rho:.2f}}}")
            else:
                vals.append(f"{rho:.2f}")
        latex += label + " & " + " & ".join(vals) + " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 72)
    print("Uncertainty Analysis for Real Evacuation Outputs")
    print("=" * 72)

    print("\n[1/6] Loading real data...")
    merged_df, params, all_metric_arrays, param_columns = load_real_data(
        input_csv=INPUT_CSV,
        output_root=OUTPUT_ROOT,
        target_metrics=TARGET_METRICS,
    )

    print(f"     Paired samples loaded: {len(merged_df)}")
    print(f"     Strategies: {list(STRATEGY_PREFIX.keys())}")
    print(f"     Metrics: {list(all_metric_arrays.keys())}")
    print(f"     Parameter columns: {param_columns}")
    for metric, T_arrays in all_metric_arrays.items():
        first_arr = next(iter(T_arrays.values()))
        print(f"       {metric}: {len(first_arr)} samples per strategy")

    merged_df.to_csv(RESULTS_DIR / "paired_samples.csv", index=False)
    print(f"     Paired sample table saved to: {RESULTS_DIR / 'paired_samples.csv'}")

    # ── 逐指标分析 ────────────────────────────────────────────────────────
    all_results = {}

    latex_all = "% Auto-generated LaTeX tables\n\n"

    for metric, T_arrays in all_metric_arrays.items():
        metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
        metric_dir = RESULTS_DIR / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*72}")
        print(f"Metric: {metric}  ({metric_label})")
        print(f"{'='*72}")

        print("\n  [2/6] Basic statistics...")
        basic_stats = analyze_basic_statistics(T_arrays)
        for name, s in basic_stats.items():
            print(f"     {name:10s}: Mean={s['mean']:.4f}, "
                  f"CVaR95={s['cvar_95']:.4f} "
                  f"[{s['cvar_ci_lower']:.4f}, {s['cvar_ci_upper']:.4f}]")

        print("  [3/6] Exceedance probabilities...")
        # 只对时间指标做超越概率（非时间指标阈值无意义）
        is_time_metric = metric in ("T_0.90", "T_0.95", "T_0.99")
        exc_results = analyze_exceedance(T_arrays) if is_time_metric else {}

        print("  [4/6] Paired comparison...")
        paired_results = analyze_paired_comparison(T_arrays)
        for key, val in paired_results.items():
            print(f"     {key}: Mean Δ={val['mean_improvement']:.4f} "
                  f"[{val['ci'][0]:.4f}, {val['ci'][1]:.4f}], "
                  f"better {val['pct_better']:.1f}%, "
                  f"p={val['wilcoxon_stat'].pvalue:.3e}")

        print("  [5/6] Sensitivity...")
        sensitivity_results = analyze_sensitivity(params, param_columns, T_arrays)

        print("  [6/6] Figures...")
        if is_time_metric:
            plot_reliability_curves(T_arrays, metric_dir / "reliability_curves.pdf")
        plot_cvar_comparison(basic_stats, metric_dir / "cvar_comparison.pdf")
        plot_sensitivity_heatmap(sensitivity_results, param_columns, metric_dir / "sensitivity_heatmap.pdf")
        plot_improvement_histogram(T_arrays, metric_dir / "improvement_histogram.pdf")
        plot_tornado_diagram(params, param_columns, T_arrays, metric_dir / "tornado_diagram.pdf")

        latex_basic = generate_latex_table_basic(basic_stats)
        latex_sens  = generate_latex_table_sensitivity(sensitivity_results, param_columns)
        latex_exc   = generate_latex_table_exceedance(exc_results) if is_time_metric else ""

        latex_all += f"% === {metric} ===\n"
        latex_all += latex_basic + "\n"
        if latex_exc:
            latex_all += latex_exc + "\n"
        latex_all += latex_sens + "\n\n"

        all_results[metric] = {
            "basic_stats": basic_stats,
            "exc_results": exc_results,
            "paired_results": paired_results,
            "sensitivity_results": sensitivity_results,
        }

    with open(RESULTS_DIR / "latex_tables.tex", "w", encoding="utf-8") as f:
        f.write(latex_all)

    print(f"\n✓ All results saved to {RESULTS_DIR}/")
    print(f"  - paired_samples.csv")
    print(f"  - latex_tables.tex")
    for metric in all_metric_arrays:
        print(f"  - {metric}/  (cvar_comparison, sensitivity_heatmap, improvement_histogram, tornado_diagram"
              + (", reliability_curves" if metric in ("T_0.90","T_0.95","T_0.99") else "") + ")")

    return all_results, merged_df


if __name__ == "__main__":
    results = main()