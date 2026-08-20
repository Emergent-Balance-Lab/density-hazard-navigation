import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Iterable, List

def _first_crossing_time_linear(time_arr, value_arr, threshold):
    """
    Find earliest time when value_arr <= threshold.
    Uses linear interpolation between adjacent samples.
    Returns float time, or None if never crosses.
    """
    if len(time_arr) == 0:
        return None

    if value_arr[0] <= threshold:
        return float(time_arr[0])

    for i in range(1, len(time_arr)):
        v0, v1 = value_arr[i - 1], value_arr[i]
        if v1 <= threshold < v0:
            t0, t1 = float(time_arr[i - 1]), float(time_arr[i])
            denom = (v1 - v0)
            if abs(denom) < 1e-12:
                return t1
            alpha = (threshold - v0) / denom
            return t0 + alpha * (t1 - t0)

        if v1 <= threshold:
            return float(time_arr[i])

    return None

from pathlib import Path

def read_total_crowd(project_name: str, base_dir: str = "../../data/simulation_inputs") -> int:
    """
    Read data/simulation_inputs/{project_name}/crowd.txt and return total crowd number (int).

    Expected first line example:
        Crowd 50000 1
    Second line might be like:
        Male 1 30
    """
    crowd_path = Path(base_dir) / project_name / "crowd.txt"
    if not crowd_path.exists():
        raise FileNotFoundError(f"crowd.txt not found: {crowd_path}")

    with crowd_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise ValueError(f"crowd.txt first line is empty: {crowd_path}")

    parts = first_line.split()
    # 最常见格式: ["Crowd", "50000", "1"]
    if len(parts) < 2 or parts[0].lower() != "crowd":
        raise ValueError(f"Unexpected first line format: '{first_line}' in {crowd_path}")

    try:
        total = int(float(parts[1]))  # 兼容 "50000" 或 "50000.0"
    except Exception as e:
        raise ValueError(f"Cannot parse crowd number from: '{first_line}' in {crowd_path}") from e

    return total


def compute_time_efficiency_indices(
    project_name: str,
    *,
    base_output_dir: str = "../../results/simulation",
    filename: str = "alive_series.csv",
    quantiles = (0.90, 0.95, 0.99),   # <- include 0.99 instead of Tmax
    ) -> dict:
    """
    Read ../../results/analysis/{project_name}/alive_series.csv and compute completion times:
      - T_p for p in quantiles (default: 0.90, 0.95, 0.99)

    alive(t) = not-yet-evacuated count.
    Completion p means alive(t) <= (1-p)*total_n.
    """

    total_n = read_total_crowd(project_name, base_dir = "../../data/simulation_inputs")   

    csv_path = os.path.join(base_output_dir, project_name, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"alive series file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"time", "alive"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got: {list(df.columns)}")

    df = df.sort_values("time").reset_index(drop=True)

    time_arr = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    alive_arr = pd.to_numeric(df["alive"], errors="coerce").to_numpy()

    mask = (~pd.isna(time_arr)) & (~pd.isna(alive_arr))
    time_arr = time_arr[mask]
    alive_arr = alive_arr[mask]

    if len(time_arr) == 0:
        raise ValueError("alive_series.csv has no valid numeric rows.")

    results = {
        "project": project_name,
        "total_n": int(total_n),
        "t_start": float(time_arr[0]),
        "t_end": float(time_arr[-1]),
        "alive_start": float(alive_arr[0]),
        "alive_end": float(alive_arr[-1]),
    }

    for p in quantiles:
        thr_alive = (1.0 - float(p)) * total_n
        t_p = _first_crossing_time_linear(time_arr, alive_arr, thr_alive)   # compute T_p
        # results[f"T_{p:.2f}"] = t_p                                         # None if never reached
        results[f"T_{p:.2f}"] = None if t_p is None else round(float(t_p), 1)
    return results

def _read_meta_basic(meta_path: Path) -> Dict[str, float]:
    keys_needed = {"width", "height", "cellSize", "x_min", "y_min"}
    info: Dict[str, float] = {}

    if not meta_path.exists():
        raise FileNotFoundError(f"meta file not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        for _ in range(5):  # only first 5 lines
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k in keys_needed:
                info[k] = float(v)

    missing = keys_needed - set(info.keys())
    if missing:
        raise ValueError(f"meta.txt missing keys in first 5 lines: {sorted(missing)}; got {info}")

    info["width"] = int(info["width"])
    info["height"] = int(info["height"])
    return info

def compute_run_record_statistics(
    project_name: str,
    *,
    base_output_dir: str = "../../results/simulation",
    filename: str = "run_record.csv",
    quantiles = (0.90, 0.95, 0.99),
) -> dict:
    """
    Read ../../results/simulation/{project_name}/run_record.csv and compute statistics for:
      - distance
      - route_switching_count

    For each metric, compute:
      - mean
      - median
      - q90, q95, q99
    """

    csv_path = os.path.join(base_output_dir, project_name, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"run_record file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"distance", "route_switching_count"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got: {list(df.columns)}")

    distance_arr = pd.to_numeric(df["distance"], errors="coerce").to_numpy()
    route_arr = pd.to_numeric(df["route_switching_count"], errors="coerce").to_numpy()

    distance_arr = distance_arr[~pd.isna(distance_arr)]
    route_arr = route_arr[~pd.isna(route_arr)]

    if len(distance_arr) == 0:
        raise ValueError("run_record.csv has no valid numeric values in 'distance'.")
    if len(route_arr) == 0:
        raise ValueError("run_record.csv has no valid numeric values in 'route_switching_count'.")

    def _summarize(arr: np.ndarray) -> dict:
        out = {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
        }
        for q in quantiles:
            out[f"q{int(q * 100)}"] = round(float(np.quantile(arr, q)), 4)
        return out

    results = {
        "project": project_name,
        "n_records": int(len(df)),
        "distance": _summarize(distance_arr),
        "route_switching_count": _summarize(route_arr),
    }

    return results

def compute_grid_value_indices_selected_layers(
    project_name: str,
    *,
    base_output_dir: str = "../../results/simulation",
    bin_subdir: str = "bin",
    meta_filename: str = "meta.txt",
    layer_prefix: str = "layer_",
    layer_ids: Optional[Iterable[int]] = None,      # e.g. [7,8,9]
    dtype: Any = np.float32,
    quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),          # q90/q95/q99
    max_multipliers: Tuple[float, ...] = (0.90, 0.95, 0.99),    # 0.9*max etc.
    ignore_zero: bool = True,   # <-- NEW: remove value==0 cells
    ignore_nan: bool = True,
    ignore_inf: bool = True,
    round_digits: Optional[int] = 6,
) -> Dict[str, Any]:
    # -------- read meta (only first 5 lines) --------
    bin_dir = Path(base_output_dir) / project_name / bin_subdir
    meta_path = bin_dir / meta_filename
    if not bin_dir.exists():
        raise FileNotFoundError(f"bin directory not found: {bin_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta file not found: {meta_path}")

    # keys_needed = {"width", "height", "cellSize", "x_min", "y_min", "soft_risk", "hard_harm"}
    keys_needed = {
        "width", "height", "cellSize", "x_min", "y_min",
        "src_width", "src_height", "src_cellSize",
        "channels", "update_grid_num",
        "soft_risk", "hard_harm",
    }
    basic: Dict[str, float] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for _ in range(15):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k in keys_needed:
                basic[k] = float(v)

    missing = keys_needed - set(basic.keys())
    if missing:
        raise ValueError(f"meta.txt missing keys in first 5 lines: {sorted(missing)}; got {basic}")

    W, H = int(basic["width"]), int(basic["height"])
    cellSize = float(basic["cellSize"])
    cell_area = cellSize * cellSize
    N = W * H

    src_W = int(basic["src_width"])
    src_H = int(basic["src_height"])
    src_cell = float(basic["src_cellSize"])
    src_area = src_cell * src_cell
    src_N = src_W * src_H

    channels = int(basic["channels"])
    update_grid_num = int(basic["update_grid_num"])

    soft_risk = float(basic["soft_risk"])
    hard_harm = int(basic["hard_harm"])

    # -------- decide which bin files to load --------
    if layer_ids is None:
        bin_files = sorted(bin_dir.glob("*.bin"))
        if not bin_files:
            raise FileNotFoundError(f"no .bin files found in: {bin_dir}")
        selected_layer_ids = None
    else:
        layer_ids = [int(x) for x in layer_ids]
        selected_layer_ids = layer_ids
        bin_files = []
        missing_files = []
        for lid in layer_ids:
            fp = bin_dir / f"{layer_prefix}{lid}.bin"
            if fp.exists():
                bin_files.append(fp)
            else:
                missing_files.append(fp.name)
        if missing_files:
            raise FileNotFoundError(f"missing bin files in {bin_dir}: {missing_files}")

    def _r(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        return round(float(x), round_digits) if isinstance(round_digits, int) else float(x)

    out: Dict[str, Any] = {
        "project": project_name,
        "grid": {
            "width": W,
            "height": H,
            "cellSize": cellSize,
            "cell_area": cell_area,
            "x_min": float(basic["x_min"]),
            "y_min": float(basic["y_min"]),
            "num_cells": N,
        },
        "src_grid": {   # NEW: meta里的高分辨率网格信息
            "src_width": src_W,
            "src_height": src_H,
            "src_cellSize": src_cell,
            "src_cell_area": src_area,
            "src_num_cells": src_N,
            "channels": channels,
            "update_grid_num": update_grid_num,
        },

        

        # NEW: risk_all 汇总层（总量 + 归一化）
        "risk_all": {
            # keep raw totals
            "soft_risk": soft_risk,
            "hard_harm": hard_harm,

            # # coarse grid normalization
            # "soft_risk_per_cell": _r(soft_risk / N),
            # "hard_harm_per_cell": _r(hard_harm / N),
            # "soft_risk_per_m2": _r(soft_risk / (N * cell_area)),
            # "hard_harm_per_m2": _r(hard_harm / (N * cell_area)),

            # # src grid normalization (更“物理一致”，因为soft_risk/hard_harm通常按src计算/累加)
            # "soft_risk_per_src_cell": _r(soft_risk / src_N),
            # "hard_harm_per_src_cell": _r(hard_harm / src_N),
            # "soft_risk_per_src_m2": _r(soft_risk / (src_N * src_area)),
            # "hard_harm_per_src_m2": _r(hard_harm / (src_N * src_area)),

            # # ratio between grids (should be ~10 for your example)
            # "scale_factor_cell": _r((cellSize / src_cell) if src_cell > 0 else None),
            # "scale_factor_area": _r(((cell_area / src_area) if src_area > 0 else None)),
        },

        "selected_layer_ids": selected_layer_ids,
        "num_bin_files": len(bin_files),
        "layers": [],
    }

    for fp in bin_files:
        arr = np.fromfile(fp, dtype=dtype)
        if arr.size != N:
            raise ValueError(
                f"bin size mismatch: {fp.name} has {arr.size} values, expected {N} (= {W}*{H}). "
                f"Check meta width/height or dtype."
            )

        # --- build valid mask (IMPORTANT: remove zeros first) ---
        mask = np.ones(arr.shape, dtype=bool)

        if ignore_zero:
            mask &= (arr != 0)

        if ignore_nan and np.issubdtype(arr.dtype, np.floating):
            mask &= ~np.isnan(arr)

        if ignore_inf and np.issubdtype(arr.dtype, np.floating):
            mask &= ~np.isinf(arr)

        valid = arr[mask]

        layer_stat: Dict[str, Any] = {"file": fp.name, "dtype": str(np.dtype(dtype))}
        layer_stat["valid_n"] = int(valid.size)
        layer_stat["zero_removed"] = bool(ignore_zero)
        layer_stat["zero_n_total"] = int(np.sum(arr == 0))

        if valid.size == 0:
            layer_stat.update({"min": None, "max": None, "mean": None, "std": None})
            # thresholds
            for q in quantiles:
                k = f"q{int(round(q*100))}"
                layer_stat[k] = None
                layer_stat[f"count_ge_{k}"] = None
                layer_stat[f"freq_ge_{k}"] = None
                layer_stat[f"area_ge_{k}"] = None
            for m in max_multipliers:
                k = f"max_mul_{int(round(m*100))}"
                layer_stat[k] = None
                layer_stat[f"count_ge_{k}"] = None
                layer_stat[f"freq_ge_{k}"] = None
                layer_stat[f"area_ge_{k}"] = None
            out["layers"].append(layer_stat)
            continue

        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        vmean = float(np.mean(valid))
        vstd = float(np.std(valid))

        layer_stat["min"] = _r(vmin)
        layer_stat["max"] = _r(vmax)
        layer_stat["mean"] = _r(vmean)
        layer_stat["std"] = _r(vstd)

        # --- (A) quantile thresholds: q90/q95/q99 (on valid only) ---
        qs = np.quantile(valid, quantiles)
        for q, thr in zip(quantiles, qs):
            key = f"q{int(round(q*100))}"
            thr = float(thr)
            c = int(np.sum(valid >= thr))  # high-value region within valid set
            layer_stat[key] = _r(thr)
            layer_stat[f"count_ge_{key}"] = c
            layer_stat[f"freq_ge_{key}"] = _r(c / valid.size)
            layer_stat[f"area_ge_{key}"] = _r(c * cell_area)

        # --- (B) amplitude thresholds: 0.9*max etc (max computed on valid) ---
        for m in max_multipliers:
            key = f"max_mul_{int(round(m*100))}"
            thr = float(m) * vmax
            c = int(np.sum(valid >= thr))
            layer_stat[key] = _r(thr)
            layer_stat[f"count_ge_{key}"] = c
            layer_stat[f"freq_ge_{key}"] = _r(c / valid.size)
            layer_stat[f"area_ge_{key}"] = _r(c * cell_area)

        out["layers"].append(layer_stat)
    return out

def analyze_and_save_projects(
    project_names: List[str],
    *,
    base_output_dir: str = "../../results/simulation",
    save_filename: str = "metrics_summary.json",
    # time
    alive_filename: str = "alive_series.csv",
    time_quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),
    # space
    layer_ids: Optional[Iterable[int]] = (7, 8, 9),
    grid_quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),
    max_multipliers: Tuple[float, ...] = (0.90, 0.95, 0.99),
    dtype=np.float32,
    ignore_zero: bool = True,
    # run record
    run_record_filename: str = "run_record.csv",
    run_record_quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),
) -> Dict[str, Any]:
    """
    Input: project_names (list of names)
    Output: For each project, compute + save JSON to:
        ../../results/simulation/{project_name}/metrics_summary.json

    Returns a dict keyed by project_name.
    """
    all_results: Dict[str, Any] = {}

    for name in project_names:
        # 1) time metrics
        time_metrics = compute_time_efficiency_indices(
            name,
            base_output_dir=base_output_dir,
            filename=alive_filename,
            quantiles=time_quantiles,
        )

        # 2) spatial metrics
        spatial_metrics = compute_grid_value_indices_selected_layers(
            name,
            base_output_dir=base_output_dir,
            layer_ids=list(layer_ids) if layer_ids is not None else None,
            quantiles=grid_quantiles,
            max_multipliers=max_multipliers,
            dtype=dtype,
            ignore_zero=ignore_zero,
        )
        
        # 3) run_record statistics
        run_record_metrics = compute_run_record_statistics(
            name,
            base_output_dir=base_output_dir,
            filename=run_record_filename,
            quantiles=run_record_quantiles,
        )

        # 4) final json
        summary = {
            "project": name,
            "time_efficiency": time_metrics,
            "spatial_efficiency": spatial_metrics,
            "run_record_statistics": run_record_metrics,
        }
        
        # 4) save
        out_dir = Path(base_output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / save_filename
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        all_results[name] = summary
    return all_results

def analyze_and_save_projects_330(
    project_names: List[str],
    *,
    base_output_dir: str = "../../results/simulation",
    save_filename: str = "metrics_summary.json",
    # time
    alive_filename: str = "alive_series.csv",
    time_quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),
    # space
    layer_ids: Optional[Iterable[int]] = (7, 8, 9),
    grid_quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),
    max_multipliers: Tuple[float, ...] = (0.90, 0.95, 0.99),
    dtype = np.float32,
    ignore_zero: bool = True,
) -> Dict[str, Any]:
    """
    Input: project_names (list of names)
    Output: For each project, compute + save JSON to:
        ../../results/simulation/{project_name}/metrics_summary.json

    Returns a dict keyed by project_name.
    """
    all_results: Dict[str, Any] = {}

    for name in project_names:
        # 1) time metrics
        time_metrics = compute_time_efficiency_indices(
            name,
            base_output_dir=base_output_dir,
            filename=alive_filename,
            quantiles=time_quantiles,
        )

        # 2) spatial metrics
        spatial_metrics = compute_grid_value_indices_selected_layers(
            name,
            base_output_dir=base_output_dir,
            layer_ids=list(layer_ids) if layer_ids is not None else None,
            quantiles=grid_quantiles,
            max_multipliers=max_multipliers,
            dtype=dtype,
            ignore_zero=ignore_zero,
        )

        # 3) final json
        summary = {
            "project": name,
            "time_efficiency": time_metrics,
            "spatial_efficiency": spatial_metrics,
        }

        # 4) save
        out_dir = Path(base_output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / save_filename
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        all_results[name] = summary
    return all_results


if __name__ == "__main__":

    # #time/effciency
    # out = compute_time_efficiency_indices("shipai_1")
    # print(out)
    # stats = compute_grid_value_indices_selected_layers(
    #     "shipai_1",
    #     layer_ids = [7, 8, 9],
    # )
    # print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    projects = ["risk_10_3", "risk_10_4", "risk_10_5"]  # 你直接改这里

    # results = analyze_and_save_projects(
    #     projects,
    #     layer_ids = [7, 8, 9],
    #     save_filename="metrics_summary.json",
    # )
    
    # 4) 分析并保存结果
    analyze_and_save_projects(
        projects,
        # layer_ids=[7, 8, 9, 14, 15, 16, 17],
        layer_ids=[7, 8, 9, 14, 15, 16, 17],
        save_filename="metrics_summary.json",
    )
    
    # 可选：打印一个总览
    # print(json.dumps(list(results.keys()), ensure_ascii=False, indent=2))





