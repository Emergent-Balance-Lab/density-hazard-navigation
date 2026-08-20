
from __future__ import annotations
from simulation_io import *
from metric_extraction import *
import subprocess
"""
为三类标准通道场景自动生成 project 文件夹：
- straight  笔直通道
- l_shape   L 形通道
- t_shape   T 形通道

本版本按你这次的新要求修改：
1. boundary.txt 不再直接使用原始通道边界，而是对“可通行区域”向外四周扩宽 2 m。
2. building.txt 不再复制模板，也不再读取别的 building 文件。
   而是把“外扩 2 m 之后新增的那一圈区域”离散成很多小矩形障碍物，直接写入 building.txt。
3. building.txt 中每个障碍格式为：
   Building 点数 5 面积
   x1 y1
   x2 y2
   ...
   其中层高固定为 5；点数按 polygon 顶点数写，矩形通常为 4。
4. crowd.txt / safearea.txt 仍自动生成；其他静态文件继续从模板复制。

新增：
5. 仿真结束并生成 metrics_summary.json 后，自动读取每个 project 的
   _scene_info.txt + metrics_summary.json / matrics_summary.json
   计算平均流量 q = total_n / (time * width)
6. 提供统一 draw(...) 入口，支持：
   - 按场景分类的折线图
   - 按“流量-道路宽度”关系的散点图
   - 三类场景合并散点图
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional
import math
import random
import shutil
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

Point = Tuple[float, float]


# ============================================================
# 参数区
# ============================================================

@dataclass
class ScenarioSpec:
    scene_name: str        # straight / l_shape / t_shape
    idx: int               # 1,2,3...
    width_m: float         # 通道宽度 0.5 ~ 5.0 m
    density_min: float = 5
    density_max: float = 5

    # straight
    straight_length: float = 60.0

    # l_shape
    l_main_length: float = 30.0
    l_branch_length: float = 30.0

    # t_shape
    t_stem_length: float = 30.0
    t_cross_half_length: float = 30.0

    # 安全区深度（出口外延）
    safe_depth: float = 2.0

    # boundary 外扩宽度（本次固定按你的要求设为 2 m）
    boundary_expand_m: float = 2.0

    # 将外扩环带离散为多少尺寸的小矩形障碍
    obstacle_cell_size: float = 0.25

    # 输出时整体平移，避免负坐标
    offset_x: float = 35.0
    offset_y: float = 30.0

    # 有效出口数量：straight/l_shape=1, t_shape=2（双端出口）
    n_exits: int = 1


# ============================================================
# 基础工具
# ============================================================

def fmt_point(p: Point) -> str:
    return f"{p[0]:.3f} {p[1]:.3f}"

def write_name_txt(project_names: Sequence[str], name_txt_path: Path) -> None:
    lines = [str(len(project_names))]
    lines.extend(project_names)
    name_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def shift_geom(geom, dx: float, dy: float):
    from shapely.affinity import translate
    return translate(geom, xoff=dx, yoff=dy)


def flatten_polygons(geom) -> List[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(flatten_polygons(g))
        return out
    return []


def polygon_exterior_points(poly: Polygon) -> List[Point]:
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(x), float(y)) for x, y in coords]


# ============================================================
# 场景几何：先构造“原始可行走区域”
# ============================================================

def build_walkable_straight(spec: ScenarioSpec) -> tuple[Polygon, list[Point], float]:
    w = spec.width_m
    L = spec.straight_length
    d = spec.safe_depth

    walk = box(0.0, -w / 2, L + d, w / 2)
    walk = shift_geom(walk, spec.offset_x, spec.offset_y)

    safe_pt = (L + d / 2 + spec.offset_x, spec.offset_y)
    return walk, [safe_pt], float(walk.area)

def build_walkable_l_shape(spec: ScenarioSpec) -> tuple[Polygon, list[Point], float]:
    w = spec.width_m
    L1 = spec.l_main_length
    L2 = spec.l_branch_length
    d = spec.safe_depth

    horiz = box(0.0, -w / 2, L1 + w / 2, w / 2)
    vert = box(L1 - w / 2, w / 2, L1 + w / 2, L2 + d)
    walk = unary_union([horiz, vert])
    walk = shift_geom(walk, spec.offset_x, spec.offset_y)

    safe_pt = (L1 + spec.offset_x, L2 + d / 2 + spec.offset_y)
    return walk, [safe_pt], float(walk.area)

def build_walkable_t_shape(spec: ScenarioSpec) -> tuple[Polygon, list[Point], float]:
    """
    T 形汇流通道：横臂两端封闭，人员从横臂向茎部汇流，
    唯一出口在茎部下方（单出口汇流场景）。

    坐标示意（未平移）：

         |←  2*half  →|
         ┌─────────────┐  ← y = stem + w/2
         │  cross arm  │
         └──┬───────┬──┘  ← y = stem - w/2
            │ stem  │
            │       │
            └───┬───┘  ← y = 0
                │ exit │  ← 向下延伸 safe_depth
                └──────┘  ← y = -d
    """
    w = spec.width_m
    stem = spec.t_stem_length
    half = spec.t_cross_half_length
    d = spec.safe_depth

    # 横臂：对称，两端封闭（不延伸出口）
    cross_rect = box(-half, stem - w / 2, half, stem + w / 2)
    # 茎部：向上连接横臂，向下延伸 safe_depth 作为出口缓冲区
    stem_rect = box(-w / 2, -d, w / 2, stem)
    walk = unary_union([stem_rect, cross_rect])
    walk = shift_geom(walk, spec.offset_x, spec.offset_y)

    # 唯一出口：茎部下方中心
    safe_pts = [(spec.offset_x, -d / 2 + spec.offset_y)]
    return walk, safe_pts, float(walk.area)

def build_scene_geometry(spec: ScenarioSpec):
    if spec.scene_name == "straight":
        walk, safe_pts, walk_area = build_walkable_straight(spec)
    elif spec.scene_name == "l_shape":
        walk, safe_pts, walk_area = build_walkable_l_shape(spec)
    elif spec.scene_name == "t_shape":
        walk, safe_pts, walk_area = build_walkable_t_shape(spec)
    else:
        raise ValueError(f"Unsupported scene_name: {spec.scene_name}")

    rho = random.uniform(spec.density_min, spec.density_max)
    people = max(1, int(round(walk_area * rho)))

    outer_boundary = walk.buffer(spec.boundary_expand_m, join_style=2)
    obstacle_region = outer_boundary.difference(walk)
    obstacle_polys = raster_fill_region_with_rectangles(obstacle_region, spec.obstacle_cell_size)

    return walk, outer_boundary, obstacle_polys, safe_pts, people, rho

# ============================================================
# 将外扩 ring 区域离散成很多小矩形障碍物
# ============================================================

def frange(start: float, stop: float, step: float):
    x = start
    while x < stop - 1e-12:
        yield x
        x += step

def raster_fill_region_with_rectangles(region, cell_size: float = 0.25) -> List[Polygon]:
    if region.is_empty:
        return []

    minx, miny, maxx, maxy = region.bounds
    start_x = math.floor(minx / cell_size) * cell_size
    start_y = math.floor(miny / cell_size) * cell_size
    end_x = math.ceil(maxx / cell_size) * cell_size
    end_y = math.ceil(maxy / cell_size) * cell_size

    polys: List[Polygon] = []
    area_eps = 1e-9

    for x in frange(start_x, end_x, cell_size):
        for y in frange(start_y, end_y, cell_size):
            cell = box(x, y, x + cell_size, y + cell_size)
            inter = region.intersection(cell)

            if inter.is_empty:
                continue

            for g in flatten_polygons(inter):
                if g.area > area_eps:
                    polys.append(g)

    return polys

# ============================================================
# 写文件
# ============================================================

# def write_boundary(boundary_path: Path, 
#                    boundary_poly: Polygon) -> None:
#     pts = polygon_exterior_points(boundary_poly)
#     lines = ["1", f"Boundary {len(pts)}"]
#     lines.extend(fmt_point(p) for p in pts)
#     boundary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
def write_boundary(
    boundary_path: Path,
    boundary_poly: Polygon,
    cell_size: float = 0.1,
    canvas_buffer: float = 10.0,
) -> None:
    pts = polygon_exterior_points(boundary_poly)

    _, _, maxx, maxy = boundary_poly.bounds
    canvas_width = maxx + canvas_buffer
    canvas_length = maxy + canvas_buffer
    lines = [
        "1",
        f"Canvas {canvas_width:.3f} {canvas_length:.3f}",
        f"Cell_size {cell_size:.1f}",
        f"Boundary {len(pts)}",
    ]
    lines.extend(fmt_point(p) for p in pts)
    boundary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_safearea(safearea_path: Path, safe_points: Sequence[Point], radius: float = 3.0, weight: float = 1.0) -> None:
    lines = [str(len(safe_points))]
    for p in safe_points:
        lines.append(f"SafeArea 1 {radius:.1f} {weight:.1f}")
        lines.append(fmt_point(p))
        lines.append("")
    safearea_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def update_crowd_total_from_template(crowd_template: Path, crowd_out: Path, total_people: int) -> None:
    lines = crowd_template.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("crowd template is empty")

    first = lines[0].strip().split()
    if len(first) >= 3 and first[0].lower() == "crowd":
        first[1] = str(total_people)
        lines[0] = " ".join(first)
    else:
        lines = [f"Crowd {total_people} 1", "Male 1 30"]

    crowd_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_buildings(building_path: Path, building_polys: Sequence[Polygon], floor_height: float = 5.0) -> None:
    lines = [str(len(building_polys))]
    for poly in building_polys:
        pts = polygon_exterior_points(poly)
        area = float(poly.area)
        lines.append(f"Building {len(pts)} {floor_height:.0f} {area:.3f}")
        lines.extend(fmt_point(p) for p in pts)
    building_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# 模板复制
# ============================================================

def copy_template(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)

_STATIC_FILE_CACHE: dict[str, bytes] = {}

def copy_static_templates(template_dir: Path, out_dir: Path) -> None:
    static_files = [
        "addwidth.txt",
        "disasterpara.txt",
        "disastersource.txt",
        "population_distribution.txt",
        "road.txt",
        "startpoint.txt",
    ]
    for name in static_files:
        src = template_dir / name
        # 首次读取时缓存到内存，后续项目直接写内存内容，避免重复磁盘读取
        if str(src) not in _STATIC_FILE_CACHE:
            _STATIC_FILE_CACHE[str(src)] = src.read_bytes()
        (out_dir / name).write_bytes(_STATIC_FILE_CACHE[str(src)])


# ============================================================
# 主生成逻辑
# ============================================================

def scene_folder_name(scene_name: str, idx: int) -> str:
    return f"{scene_name}_{idx:03d}"

def generate_one_project(template_dir: Path, output_root: Path, spec: ScenarioSpec, verbose: bool = True) -> tuple[Path, str]:
    walk, boundary_poly, building_polys, safe_pts, people, rho = build_scene_geometry(spec)

    project_name = scene_folder_name(spec.scene_name, spec.idx)
    out_dir = output_root / project_name
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_static_templates(template_dir, out_dir)
    write_boundary(out_dir / "boundary.txt", boundary_poly)
    write_buildings(out_dir / "building.txt", building_polys, floor_height=5.0)
    write_safearea(out_dir / "safearea.txt", safe_pts, radius=3.0, weight=1.0)
    update_crowd_total_from_template(template_dir / "crowd.txt", out_dir / "crowd.txt", people)

    meta_lines = [
        f"scene_name = {spec.scene_name}",
        f"scene_index = {spec.idx}",
        f"width_m = {spec.width_m:.3f}",
        f"n_exits = {spec.n_exits}",
        f"density_sampled = {rho:.4f}",
        f"people = {people}",
        f"safearea_count = {len(safe_pts)}",
        f"boundary_points = {len(polygon_exterior_points(boundary_poly))}",
        f"building_count = {len(building_polys)}",
        f"walk_area = {walk.area:.3f}",
        f"boundary_area = {boundary_poly.area:.3f}",
        f"obstacle_area = {sum(p.area for p in building_polys):.3f}",
        f"boundary_expand_m = {spec.boundary_expand_m:.3f}",
        f"obstacle_cell_size = {spec.obstacle_cell_size:.3f}",
    ]
    (out_dir / "_scene_info.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    if verbose:
        print(
            f"[OK] {out_dir} | width={spec.width_m:.2f} m | "
            f"density={rho:.2f} p/m² | people={people} | buildings={len(building_polys)}"
        )

    return out_dir, project_name


def generate_projects(
    template_dir: str | Path,
    output_root: str | Path,
    widths: Iterable[float] = (1.0, 2.0, 3.0, 4.0, 5.0),
    per_width_repeats: int = 1,
    seed: int = 42,
) -> list[str]:
    random.seed(seed)
    template_dir = Path(template_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    required = [
        "addwidth.txt",
        "crowd.txt",
        "disasterpara.txt",
        "disastersource.txt",
        "population_distribution.txt",
        "road.txt",
        "startpoint.txt",
    ]
    missing = [x for x in required if not (template_dir / x).exists()]
    
    if missing:
        raise FileNotFoundError(f"Missing template files: {missing}")

    project_names: List[str] = []

    for scene_name in ("straight", "l_shape", "t_shape"):
        idx = 1
        for w in widths:
            for _ in range(per_width_repeats):
                spec = ScenarioSpec(
                    scene_name=scene_name,
                    idx=idx,
                    width_m=float(w),
                    n_exits=1,
                )
                _, project_name = generate_one_project(template_dir, output_root, spec)
                project_names.append(project_name)
                idx += 1

    write_name_txt(project_names, output_root / "name.txt")
    print(f"[OK] name.txt written: {output_root / 'name.txt'}")
    return project_names


# ============================================================
# 流量分析与绘图
# ============================================================

FLOW_METRICS = ("flow_T_0.90", "flow_T_0.95", "flow_T_0.99", "flow_t_end", "flow_max")
TIME_KEYS = ("T_0.90", "T_0.95", "T_0.99", "t_end_seconds")
T_END_TIME_UNIT_SECONDS = 1.0   # 每个仿真步长对应的物理时间（秒）
SCENE_ORDER = ("straight", "l_shape", "t_shape")

# flow_max 截取区间：跳过前 START_PCT 预热段 + 截掉后 (1-END_PCT) 尾段
FLOW_MAX_START_PCT: float = 0.05   # 前 5% 认为是预热，跳过
FLOW_MAX_END_PCT:   float = 0.90   # 90% 之后认为是稀疏尾段，截掉

# 学术绘图标签
SCENE_LABELS: dict[str, str] = {
    "straight": "Straight",
    "l_shape":  "L-shape",
    "t_shape":  "T-shape",
}
METRIC_LABELS: dict[str, str] = {
    "flow_T_0.90": r"$q_s$ at $T_{90}$",
    "flow_T_0.95": r"$q_s$ at $T_{95}$",
    "flow_T_0.99": r"$q_s$ at $T_{99}$",
    "flow_t_end":  r"$q_s$ at $T_\mathrm{end}$",
    "flow_max":    r"$q_s$ stable max ($5\%$–$90\%$)",
}
Y_LABEL = r"Specific flow $q_s$ [pers/(s·m)]"
X_LABEL = "Corridor width $w$ (m)"

def compute_flow_max_from_series(
    alive_csv: Path,
    total_n: float,
    start_pct: float = FLOW_MAX_START_PCT,
    end_pct: float   = FLOW_MAX_END_PCT,
) -> float:
    """
    从 alive_series.csv 截取"稳定疏散段"，计算该段的平均流出速率。

    截取规则：
      - 起点 T_start：alive 第一次下穿 (1 - start_pct) * total_n
                       即已有 start_pct 比例的人离开（跳过预热段）
      - 终点 T_end  ：alive 第一次下穿 (1 - end_pct)   * total_n
                       即已有 end_pct   比例的人离开（截掉稀疏尾段）

    flow_max = (end_pct - start_pct) * total_n / (T_end - T_start)  [pers/s]

    若文件不存在、数据不足或区间无效则返回 np.nan。
    """
    if not alive_csv.exists() or total_n <= 0 or np.isnan(total_n):
        return np.nan

    try:
        df = pd.read_csv(alive_csv)
    except Exception:
        return np.nan

    if not {"time", "alive"}.issubset(df.columns):
        return np.nan

    df = df.sort_values("time").reset_index(drop=True)
    t_arr = pd.to_numeric(df["time"],  errors="coerce").to_numpy(dtype=float)
    a_arr = pd.to_numeric(df["alive"], errors="coerce").to_numpy(dtype=float)

    valid = ~(np.isnan(t_arr) | np.isnan(a_arr))
    t_arr, a_arr = t_arr[valid], a_arr[valid]

    if len(t_arr) < 2:
        return np.nan

    thr_start = (1.0 - start_pct) * total_n   # alive 下穿此值 → 进入稳定段
    thr_end   = (1.0 - end_pct)   * total_n   # alive 下穿此值 → 离开稳定段

    t_start = _first_crossing(t_arr, a_arr, thr_start)
    t_end   = _first_crossing(t_arr, a_arr, thr_end)

    if t_start is None or t_end is None or t_end <= t_start:
        return np.nan

    n_out = (end_pct - start_pct) * total_n
    return float(n_out / (t_end - t_start))


def _first_crossing(t_arr, a_arr, threshold: float) -> Optional[float]:
    """线性插值求 alive 序列第一次下穿 threshold 的时刻。"""
    if a_arr[0] <= threshold:
        return float(t_arr[0])
    for i in range(1, len(a_arr)):
        if a_arr[i] <= threshold:
            denom = a_arr[i] - a_arr[i - 1]
            if abs(denom) < 1e-12:
                return float(t_arr[i])
            alpha = (threshold - a_arr[i - 1]) / denom
            return float(t_arr[i - 1] + alpha * (t_arr[i] - t_arr[i - 1]))
    return None


def parse_scene_info(scene_info_path: Path) -> dict:
    data = {}
    if not scene_info_path.exists():
        return data
    for line in scene_info_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data

def load_metrics_file(project_dir: Path) -> Optional[dict]:
    for name in ("metrics_summary.json", "matrics_summary.json"):
        fp = project_dir / name
        if fp.exists():
            with fp.open("r", encoding="utf-8") as f:
                return json.load(f)
    return None

def infer_scene_from_project_name(project_name: str) -> Optional[str]:
    pn = project_name.lower()
    if pn.startswith("straight"):
        return "straight"
    if pn.startswith("l_shape") or pn.startswith("l-") or pn.startswith("lshape"):
        return "l_shape"
    if pn.startswith("t_shape") or pn.startswith("t-") or pn.startswith("tshape"):
        return "t_shape"
    return None

def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)

def safe_div(num: float, den: float) -> float:
    if den is None or np.isnan(den) or den <= 0:
        return np.nan
    return num / den

def collect_flow_dataframe(
    output_root: str | Path,
    project_names: Optional[Sequence[str]] = None,
    input_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    output_root : C++ 仿真输出目录（存放 metrics_summary.json / alive_series.csv）
    input_root  : 项目配置目录（存放 _scene_info.txt / name.txt）；
                  若为 None 则与 output_root 相同（兼容旧行为）。
    """
    output_root = Path(output_root)
    input_root  = Path(input_root) if input_root is not None else output_root

    if project_names is None:
        name_txt = input_root / "name.txt"
        if name_txt.exists():
            raw_lines = name_txt.read_text(encoding="utf-8").splitlines()
            project_names = [x.strip() for x in raw_lines[1:] if x.strip()]
        else:
            project_names = sorted([p.name for p in output_root.iterdir() if p.is_dir()])

    rows = []

    for project_name in project_names:
        input_dir  = input_root  / project_name
        output_dir = output_root / project_name

        scene_meta   = parse_scene_info(input_dir / "_scene_info.txt")
        metrics_data = load_metrics_file(output_dir)
        if metrics_data is None:
            continue

        time_eff = metrics_data.get("time_efficiency", metrics_data)
        scene_name = scene_meta.get("scene_name", infer_scene_from_project_name(project_name))
        width_m = safe_float(scene_meta.get("width_m", np.nan))
        n_exits = int(safe_float(scene_meta.get("n_exits", 1)))
        effective_width = width_m * n_exits   # T 形双出口时有效宽度 = 2W
        total_n = safe_float(time_eff.get("total_n", scene_meta.get("people", np.nan)))

        row = {
            "project": project_name,
            "scene_name": scene_name,
            "width_m": width_m,
            "n_exits": n_exits,
            "total_n": total_n,
            # 原始时间值（仿真步长，1步=T_END_TIME_UNIT_SECONDS秒）
            "T_0.90": safe_float(time_eff.get("T_0.90", np.nan)),
            "T_0.95": safe_float(time_eff.get("T_0.95", np.nan)),
            "T_0.99": safe_float(time_eff.get("T_0.99", np.nan)),
            "t_end": safe_float(time_eff.get("t_end", np.nan)),
            "alive_start": safe_float(time_eff.get("alive_start", np.nan)),
            "alive_end": safe_float(time_eff.get("alive_end", np.nan)),
        }

        # 统一换算为秒，再计算比流量 q_s = N / (T_s × W_eff) [pers/(s·m)]
        t_factor = T_END_TIME_UNIT_SECONDS
        row["t_end_seconds"] = safe_float(row["t_end"] * t_factor)

        # row["flow_T_0.90"] = safe_div(total_n, row["T_0.90"] * t_factor * effective_width)
        # row["flow_T_0.95"] = safe_div(total_n, row["T_0.95"] * t_factor * effective_width)
        # row["flow_T_0.99"] = safe_div(total_n, row["T_0.99"] * t_factor * effective_width)
        # row["flow_t_end"]  = safe_div(total_n, row["t_end_seconds"]       * effective_width)
        
        row["flow_T_0.90"] = safe_div(total_n, row["T_0.90"] * t_factor )
        row["flow_T_0.95"] = safe_div(total_n, row["T_0.95"] * t_factor )
        row["flow_T_0.99"] = safe_div(total_n, row["T_0.99"] * t_factor )
        row["flow_t_end"]  = safe_div(total_n, row["t_end_seconds"]       )

        # 稳定疏散段流量：跳过前 FLOW_MAX_START_PCT 预热 + 截掉尾段
        alive_csv = output_dir / "alive_series.csv"
        row["flow_max"] = compute_flow_max_from_series(
            alive_csv, total_n,
            start_pct=FLOW_MAX_START_PCT,
            end_pct=FLOW_MAX_END_PCT,
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["scene_name"] = pd.Categorical(df["scene_name"], categories=list(SCENE_ORDER), ordered=True)
        df = df.sort_values(["scene_name", "width_m", "project"]).reset_index(drop=True)
    return df

def build_grouped_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(["scene_name", "width_m"], observed=True)[list(FLOW_METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped

def _plot_scene_line(group_df: pd.DataFrame, scene_name: str, save_path: Path) -> None:
    if group_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = group_df["width_m"].to_numpy(dtype=float)

    for metric in FLOW_METRICS:
        mean_col = f"{metric}_mean"
        if mean_col not in group_df.columns:
            continue
        y = group_df[mean_col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label=METRIC_LABELS.get(metric, metric))

    ax.set_title(f"{SCENE_LABELS.get(scene_name, scene_name)}: mean specific flow vs corridor width")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)

def _plot_scene_scatter(
    raw_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    scene_name: str,
    scatter_metric: str,
    save_path: Path,
    show_mean_line: bool,
    show_std_errorbar: bool,
) -> None:
    if raw_df.empty:
        return

    scene_label = SCENE_LABELS.get(scene_name, scene_name)
    metric_label = METRIC_LABELS.get(scatter_metric, scatter_metric)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(raw_df["width_m"], raw_df[scatter_metric], alpha=0.75, label=f"{scene_label} samples")

    if show_mean_line and not grouped_df.empty:
        x = grouped_df["width_m"].to_numpy(dtype=float)
        y = grouped_df[f"{scatter_metric}_mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=f"{scene_label} mean")

        if show_std_errorbar and f"{scatter_metric}_std" in grouped_df.columns:
            yerr = grouped_df[f"{scatter_metric}_std"].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=4, alpha=0.8)

    ax.set_title(f"{scene_label}: {metric_label} vs corridor width")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)

def _plot_combined_scatter(
    df: pd.DataFrame,
    grouped_stats: pd.DataFrame,
    scatter_metric: str,
    save_path: Path,
    show_mean_line: bool,
    show_std_errorbar: bool,
) -> None:
    if df.empty:
        return

    metric_label = METRIC_LABELS.get(scatter_metric, scatter_metric)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for scene_name in SCENE_ORDER:
        scene_label = SCENE_LABELS.get(scene_name, scene_name)
        raw_sub = df[df["scene_name"] == scene_name]
        if raw_sub.empty:
            continue
        ax.scatter(raw_sub["width_m"], raw_sub[scatter_metric], alpha=0.65, label=f"{scene_label} samples")

        if show_mean_line:
            grp_sub = grouped_stats[grouped_stats["scene_name"] == scene_name]
            if not grp_sub.empty:
                x = grp_sub["width_m"].to_numpy(dtype=float)
                y = grp_sub[f"{scatter_metric}_mean"].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", linewidth=2.0, label=f"{scene_label} mean")

                if show_std_errorbar and f"{scatter_metric}_std" in grp_sub.columns:
                    yerr = grp_sub[f"{scatter_metric}_std"].fillna(0.0).to_numpy(dtype=float)
                    ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=4, alpha=0.8)

    ax.set_title(f"Comparison of {metric_label} vs corridor width")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

def draw(
    output_root: str | Path,
    input_root: Optional[str | Path] = None,
    project_names: Optional[Sequence[str]] = None,
    summary_csv: str | Path = "../../results/analysis/avg_flow_summary.csv",
    grouped_csv: str | Path = "../../results/analysis/avg_flow_grouped_stats.csv",
    figure_dir: str | Path = "../../results/analysis/avg_flow_figures",
    enable_scene_line_plots: bool = True,
    enable_width_scatter: bool = True,
    enable_combined_width_scatter: bool = True,
    scatter_metric: str = "flow_t_end",
    show_mean_line_in_scatter: bool = True,
    show_std_errorbar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    统一绘图入口：
    output_root : 仿真输出目录（metrics_summary.json 所在位置）
    input_root  : 项目配置目录（_scene_info.txt 所在位置）；None 时同 output_root
    """
    if scatter_metric not in FLOW_METRICS:
        raise ValueError(f"scatter_metric must be one of {FLOW_METRICS}, got {scatter_metric}")

    df = collect_flow_dataframe(
        output_root=output_root,
        project_names=project_names,
        input_root=input_root,
    )
    grouped_stats = build_grouped_stats(df)

    summary_csv = Path(summary_csv)
    grouped_csv = Path(grouped_csv)
    figure_dir = Path(figure_dir)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped_csv.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    grouped_stats.to_csv(grouped_csv, index=False, encoding="utf-8-sig")

    if enable_scene_line_plots and not grouped_stats.empty:
        for scene_name in SCENE_ORDER:
            grp_sub = grouped_stats[grouped_stats["scene_name"] == scene_name]
            if grp_sub.empty:
                continue
            save_path = figure_dir / f"avg_flow_{scene_name}.png"
            _plot_scene_line(grp_sub, scene_name, save_path)

    if enable_width_scatter and not df.empty:
        for scene_name in SCENE_ORDER:
            raw_sub = df[df["scene_name"] == scene_name]
            grp_sub = grouped_stats[grouped_stats["scene_name"] == scene_name]
            if raw_sub.empty:
                continue
            save_path = figure_dir / f"scatter_{scatter_metric}_{scene_name}.png"
            _plot_scene_scatter(
                raw_df=raw_sub,
                grouped_df=grp_sub,
                scene_name=scene_name,
                scatter_metric=scatter_metric,
                save_path=save_path,
                show_mean_line=show_mean_line_in_scatter,
                show_std_errorbar=show_std_errorbar,
            )

    if enable_combined_width_scatter and not df.empty:
        save_path = figure_dir / f"scatter_{scatter_metric}_combined.png"
        _plot_combined_scatter(
            df=df,
            grouped_stats=grouped_stats,
            scatter_metric=scatter_metric,
            save_path=save_path,
            show_mean_line=show_mean_line_in_scatter,
            show_std_errorbar=show_std_errorbar,
        )

    print(f"[OK] summary csv saved to: {summary_csv}")
    print(f"[OK] grouped csv saved to: {grouped_csv}")
    print(f"[OK] figures saved under:   {figure_dir}")

    return df, grouped_stats


# ============================================================
# 流量–宽度验证分析（独立分析入口，不重新运行仿真）
# ============================================================

# Directly comparable empirical capacity range for a straight corridor.
# The bend and T-junction studies use fixed corridor widths and therefore
# support qualitative geometric checks rather than capacity--width bands.
C_EMP_STRAIGHT_RANGE: tuple[float, float] = (1.6, 2.2)  # [pers/(s·m)]

# 绘图配色
SCENE_COLORS: dict[str, str] = {
    "straight": "#1f77b4",   # 蓝
    "l_shape":  "#ff7f0e",   # 橙
    "t_shape":  "#2ca02c",   # 绿
}
SCENE_MARKERS: dict[str, str] = {
    "straight": "o",
    "l_shape":  "s",
    "t_shape":  "^",
}


def remove_stuck_runs(
    df: pd.DataFrame,
    t_end_col: str = "t_end_seconds",
    group_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    在每个 (scene_name, width_m) 组内，删除仿真卡死（用时极长）的异常值。

    判定规则（任一触发）：
      1. IQR 规则：t_end > Q3 + 1.5 × IQR（当 IQR = 0 时退化为 t_end > Q3）
      2. 绝对倍数：t_end > 3 × 组内中位数（捕捉极端卡死，如 t=4999）
      3. NaN：t_end 缺失（仿真完全失败）

    仅删除偏高（用时过长）的样本，保留用时偏短（疏散顺畅）的样本。
    返回 (clean_df, removed_df)。
    """
    if group_cols is None:
        group_cols = ["scene_name", "width_m"]

    if t_end_col not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns)

    outlier_indices: list = []

    for _, grp in df.groupby(group_cols, observed=True):
        vals = grp[t_end_col]
        nan_idx = vals[vals.isna()].index.tolist()
        outlier_indices.extend(nan_idx)

        valid = vals.dropna()
        if valid.empty:
            continue

        q1  = float(valid.quantile(0.25))
        q3  = float(valid.quantile(0.75))
        iqr = q3 - q1

        # 当组内 IQR=0（所有值相同）时使用 Q3 作为上界，捕捉任何偏高值；
        # 当组内 IQR>0 时使用标准 1.5×IQR 规则。
        # 仅标记偏高值（用时过长 = 疏散失效），不剔除偏低值（疏散顺畅）。
        if iqr > 1e-9:
            upper = q3 + 1.5 * iqr
        else:
            upper = q3   # IQR=0：任何高于共同值的样本都视为异常

        bad = valid[valid > upper].index.tolist()
        outlier_indices.extend(bad)

    outlier_indices_set = set(outlier_indices)
    mask = df.index.isin(outlier_indices_set)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def fit_ols_slope(b_vals: np.ndarray, q_vals: np.ndarray) -> float:
    """
    最小二乘拟合 q = C × b（过原点）。
    返回斜率 C [pers/(s·m)]。
    """
    b = np.asarray(b_vals, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    valid = ~(np.isnan(b) | np.isnan(q))
    b, q = b[valid], q[valid]
    if len(b) == 0:
        return np.nan
    return float(np.dot(b, q) / np.dot(b, b))


def compute_validation_stats(
    df_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    按 (scene_name, width_m) 分组，计算 flow_max 的均值/标准差，
    并衍生每组比流量 J_spec = flow_max_mean / width_m。
    返回 DataFrame，包含列：
      scene_name, width_m, n, q_mean, q_std, J_spec
    """
    rows = []
    for (scene, w), grp in df_clean.groupby(["scene_name", "width_m"], observed=True):
        vals = grp["flow_max"].dropna()
        if vals.empty:
            continue
        q_mean = float(vals.mean())
        q_std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append({
            "scene_name": scene,
            "width_m": float(w),
            "n": len(vals),
            "q_mean": q_mean,
            "q_std": q_std,
            "J_spec": q_mean / float(w),
        })
    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["scene_name"] = pd.Categorical(
            stats_df["scene_name"], categories=list(SCENE_ORDER), ordered=True
        )
        stats_df = stats_df.sort_values(["scene_name", "width_m"]).reset_index(drop=True)
    return stats_df


def plot_flow_width_validation(
    df_clean: pd.DataFrame,
    stats_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    save_path: Path,
    b_ref: np.ndarray = np.linspace(0.8, 5.5, 100),
) -> dict[str, float]:
    """
    绘制"最大流量 vs 通道宽度"验证图（发表级）。

    - 散点：各次仿真样本（剔除异常值后）
    - 实线：各场景均值回归线（过原点 OLS）
    - 阴影带：有直接容量--宽度证据的直通道经验范围
    - 叉号：已被剔除的异常值（透明显示）

    返回各场景拟合斜率字典 {scene_name: C_sim}。
    """
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    fitted_slopes: dict[str, float] = {}

    # Only the straight-corridor literature provides a directly comparable
    # capacity--width range.  Do not turn fixed-width bend/T-junction
    # fundamental diagrams into unsupported capacity bands.
    corridor_low, corridor_high = C_EMP_STRAIGHT_RANGE
    ax.fill_between(
        b_ref,
        corridor_low * b_ref,
        corridor_high * b_ref,
        color="#8ecae6",
        alpha=0.30,
        linewidth=0,
        label=rf"Published straight-corridor range: $C={corridor_low:.1f}$--${corridor_high:.1f}$",
        zorder=0,
    )
    ax.plot(b_ref, corridor_low * b_ref, "--", color="#2f80b7", linewidth=0.9, zorder=1)
    ax.plot(b_ref, corridor_high * b_ref, "--", color="#2f80b7", linewidth=0.9, zorder=1)

    # 各场景数据
    for scene in SCENE_ORDER:
        color   = SCENE_COLORS[scene]
        marker  = SCENE_MARKERS[scene]
        label   = "T-shape, two-arm total" if scene == "t_shape" else SCENE_LABELS[scene]

        raw_sub  = df_clean[df_clean["scene_name"] == scene]
        stat_sub = stats_df[stats_df["scene_name"] == scene]

        if raw_sub.empty:
            continue

        # 散点（原始样本，去重后仍然可能有多点重叠）
        ax.scatter(raw_sub["width_m"], raw_sub["flow_max"],
                   color=color, marker=marker, s=30, alpha=0.55, zorder=3)

        # 均值 ± σ 误差棒
        if not stat_sub.empty:
            xm = stat_sub["width_m"].to_numpy(float)
            ym = stat_sub["q_mean"].to_numpy(float)
            ye = stat_sub["q_std"].fillna(0.0).to_numpy(float)
            ax.errorbar(xm, ym, yerr=ye, fmt=marker, color=color,
                        markersize=7, linewidth=0, elinewidth=1.5,
                        capsize=4, zorder=4, label=f"{label} (mean ± σ)")

            # OLS 拟合线（过原点）
            C_sim = fit_ols_slope(xm, ym)
            fitted_slopes[scene] = C_sim
            if not np.isnan(C_sim):
                ax.plot(b_ref, C_sim * b_ref, "-", color=color,
                        linewidth=1.8, alpha=0.85, zorder=2)

    # 标记已剔除的异常值（叉号，灰色透明）
    if removed_df is not None and not removed_df.empty:
        for scene in SCENE_ORDER:
            sub = removed_df[removed_df["scene_name"] == scene]
            if sub.empty:
                continue
            ax.scatter(sub["width_m"], sub["flow_max"],
                       marker="x", s=50, color="gray", alpha=0.5, zorder=2,
                       linewidths=1.2)
        # 只加一次图例项
        ax.scatter([], [], marker="x", s=50, color="gray", alpha=0.6,
                   linewidths=1.2, label="Flagged high-duration run")

    ax.set_xlabel(r"Corridor width $b$ (m)", fontsize=12)
    ax.set_ylabel(r"Max. flow rate $J_{\max}$ (pers/s)", fontsize=12)
    ax.set_xlim(0.6, 5.7)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] validation figure saved → {save_path}")
    return fitted_slopes


def print_validation_table(
    stats_df: pd.DataFrame,
    fitted_slopes: dict[str, float],
    removed_df: pd.DataFrame,
) -> None:
    """
    打印流量–宽度验证汇总表，方便填写论文数值。

    对 T 形通道额外计算“单臂等效比流量” C_arm = C_t / 2。
    固定宽度的转角与 T 形实验不在此输出中伪装成容量--宽度范围。
    """
    emp_ranges = {
        "straight": C_EMP_STRAIGHT_RANGE,
        "l_shape":  None,
        "t_shape":  None,
    }
    n_removed = {} if removed_df.empty else removed_df.groupby("scene_name", observed=True).size().to_dict()

    print("\n" + "=" * 70)
    print("  Flow–Width Validation Summary  (q = C · b, OLS through origin)")
    print("=" * 70)

    # 主表：直通 / L 形 / T 形总流量斜率
    print(f"\n  {'Scenario':<18} {'C_sim [pers/(s·m)]':>22} {'C_emp range':>13} {'Err low/high%':>15} {'n_rm':>6}")
    print("  " + "-" * 80)
    for scene in SCENE_ORDER:
        C_sim   = fitted_slopes.get(scene, np.nan)
        nr      = n_removed.get(scene, 0)
        label   = SCENE_LABELS.get(scene, scene)
        emp_range = emp_ranges[scene]
        if emp_range is None:
            emp_str = "not matched"
            err_str = "n/a"
        else:
            low, high = emp_range
            rel_low = (C_sim - low) / low * 100 if not np.isnan(C_sim) else np.nan
            rel_high = (C_sim - high) / high * 100 if not np.isnan(C_sim) else np.nan
            emp_str = f"{low:.1f}--{high:.1f}"
            err_str = f"{rel_low:+.1f}/{rel_high:+.1f}"
        print(f"  {label:<18} {C_sim:>22.3f} {emp_str:>13} {err_str:>15} {nr:>6}")

    # T 形：额外行：单臂等效
    C_t_total = fitted_slopes.get("t_shape", np.nan)
    if not np.isnan(C_t_total):
        C_arm = C_t_total / 2.0
        print(f"  {'T-shape (per arm)':<18} {C_arm:>22.3f} {'not matched':>13} {'n/a':>15}")

    # L vs straight 相对差
    C_s = fitted_slopes.get("straight", np.nan)
    C_l = fitted_slopes.get("l_shape",  np.nan)
    if not (np.isnan(C_s) or np.isnan(C_l)):
        diff_pct = (C_l - C_s) / C_s * 100
        print(f"\n  L-shape vs Straight capacity difference: {diff_pct:+.1f}%")
        print(f"  (empirical finding: no significant difference expected)")

    print("=" * 70)

    # 每组比流量明细
    print("\n  Per-width specific flow J_spec = q_mean / b  [pers/(s·m)]:")
    print(f"  {'Scene':<14} {'b(m)':>5} {'J_spec':>9} {'J_arm(T/2)':>11} {'n_valid':>8}")
    print("  " + "-" * 52)
    for _, row in stats_df.iterrows():
        scene = row["scene_name"]
        j_arm_str = f"{row['J_spec'] / 2:.3f}" if scene == "t_shape" else "     —"
        print(f"  {scene:<14} {row['width_m']:>5.1f} {row['J_spec']:>9.3f} {j_arm_str:>11} {int(row['n']):>8}")
    print()


def analyze_flow_width(
    output_root: str | Path = "../../results/simulation",
    input_root:  str | Path = "../../data/simulation_inputs",
    summary_csv: str | Path = "../../results/analysis/avg_flow_summary.csv",
    figure_path: str | Path = "../../results/analysis/avg_flow_figures/flow_width_validation.pdf",
    project_names: Optional[Sequence[str]] = None,
    reload_from_summary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    独立分析入口：直接加载已有仿真结果，完成流量–宽度验证分析。

    reload_from_summary=True  → 优先从 summary_csv（已有 CSV）加载，速度快
    reload_from_summary=False → 从 output_root 重新读取 JSON，结果最新

    返回 (df_clean, stats_df, fitted_slopes)
    """
    summary_csv = Path(summary_csv)
    if reload_from_summary and summary_csv.exists():
        df = pd.read_csv(summary_csv)
        if "scene_name" not in df.columns:
            # 回退到 JSON 加载
            df = collect_flow_dataframe(output_root, project_names, input_root)
    else:
        df = collect_flow_dataframe(output_root, project_names, input_root)

    if df.empty:
        print("[WARN] No data loaded; check output_root / summary_csv.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # 确保 scene_name 是有序 Categorical（便于分组排序）
    if "scene_name" in df.columns:
        df["scene_name"] = pd.Categorical(
            df["scene_name"], categories=list(SCENE_ORDER), ordered=True
        )

    # 过滤出三类标准场景
    df = df[df["scene_name"].isin(SCENE_ORDER)].copy()

    # 删除仿真卡死的异常值
    df_clean, removed_df = remove_stuck_runs(df, t_end_col="t_end_seconds")

    if not removed_df.empty:
        print(f"[INFO] Removed {len(removed_df)} stuck-run outlier(s):")
        for _, r in removed_df.iterrows():
            print(f"       {r['project']}  scene={r['scene_name']}  "
                  f"w={r['width_m']:.1f}m  t_end={r['t_end_seconds']:.0f}s")

    # 计算每组均值 / 比流量
    stats_df = compute_validation_stats(df_clean)

    # OLS 拟合 + 绘图
    fitted_slopes = plot_flow_width_validation(
        df_clean, stats_df, removed_df, save_path=figure_path
    )

    # 打印汇总表
    print_validation_table(stats_df, fitted_slopes, removed_df)

    return df_clean, stats_df, fitted_slopes


if __name__ == "__main__":
    TEMPLATE_DIR  = "../../data/simulation_inputs/STL_basic"
    INPUT_ROOT    = "../../data/simulation_inputs"    # _scene_info.txt 写到这里
    OUTPUT_ROOT   = "../../results/simulation"   # metrics_summary.json 写到这里

    ROOT = Path(__file__).resolve().parent
    REPO_ROOT = ROOT.parents[1]
    SRC_CPP = REPO_ROOT / "src" / "cpp"
    BUILD_CPP = REPO_ROOT / "build" / "cpp"

    WIDTHS = [1.0, 2.0, 3.0, 4.0, 5.0]
    PER_WIDTH_REPEATS = 5

    EXE = "STL"

    # ── 模式选择 ──────────────────────────────────────────────
    # ANALYSIS_ONLY = True  → 跳过仿真，直接分析已有结果
    # ANALYSIS_ONLY = False → 完整流程（生成项目 → 运行 C++ → 分析）
    ANALYSIS_ONLY = True

    if ANALYSIS_ONLY:
        # 仅加载已有结果并生成验证图
        analyze_flow_width(
            output_root=OUTPUT_ROOT,
            input_root=INPUT_ROOT,
            summary_csv="../../results/analysis/avg_flow_summary.csv",
            figure_path="../../results/analysis/avg_flow_figures/flow_width_validation.pdf",
            reload_from_summary=True,
        )
    else:
        # 绘图开关
        ENABLE_SCENE_LINE_PLOTS = True
        ENABLE_WIDTH_SCATTER = True
        ENABLE_COMBINED_WIDTH_SCATTER = True
        SCATTER_METRIC = "flow_t_end"
        SHOW_MEAN_LINE_IN_SCATTER = True
        SHOW_STD_ERRORBAR = True

        list_1 = generate_projects(
            template_dir=TEMPLATE_DIR,
            output_root=INPUT_ROOT,
            widths=WIDTHS,
            per_width_repeats=PER_WIDTH_REPEATS,
            seed=42,
        )

        # 2) 运行 C++：cwd=src_cpp，确保它能用 ./input/name.txt
        subprocess.run(
            [str(BUILD_CPP / EXE)],
            cwd=str(SRC_CPP),
            check=True
        )

        # 3) 转换输出的 bin 为 tif
        for project_name in list_1:
            convert_bin_to_tif_for_project(project_name)

        # 4) 分析并保存结果
        analyze_and_save_projects(
            list_1,
            layer_ids=[7, 8, 9, 14, 15, 16, 17],
            save_filename="metrics_summary.json",
        )

        # 5) 统一绘图入口
        draw(
            output_root=OUTPUT_ROOT,
            input_root=INPUT_ROOT,
            project_names=list_1,
            summary_csv="../../results/analysis/avg_flow_summary.csv",
            grouped_csv="../../results/analysis/avg_flow_grouped_stats.csv",
            figure_dir="../../results/analysis/avg_flow_figures",
            enable_scene_line_plots=ENABLE_SCENE_LINE_PLOTS,
            enable_width_scatter=ENABLE_WIDTH_SCATTER,
            enable_combined_width_scatter=ENABLE_COMBINED_WIDTH_SCATTER,
            scatter_metric=SCATTER_METRIC,
            show_mean_line_in_scatter=SHOW_MEAN_LINE_IN_SCATTER,
            show_std_errorbar=SHOW_STD_ERRORBAR,
        )

        # 6) 额外运行流量–宽度验证分析
        analyze_flow_width(
            output_root=OUTPUT_ROOT,
            input_root=INPUT_ROOT,
            summary_csv="../../results/analysis/avg_flow_summary.csv",
            figure_path="../../results/analysis/avg_flow_figures/flow_width_validation.pdf",
            project_names=list_1,
            reload_from_summary=False,
        )
