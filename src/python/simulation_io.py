import os
import math
import numpy as np
import rasterio
import random
from rasterio.transform import from_origin

from pathlib import Path
from typing import List, Iterable, Optional, Sequence, Tuple

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

# from __future__ import annotations
from dataclasses import dataclass, asdict, replace, field
import argparse
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ================= 全局配置 =================

# 源坐标系：WGS 1984 World Mercator (单位：米)
INPUT_CRS = 'EPSG:3395'
# 栅格坐标系：保留原始米制投影进行栅格化，确保精度
RASTER_CRS = INPUT_CRS

# 栅格分辨率 (Resolution)：每个像元代表的实际米数。
RASTER_RESOLUTION_METERS = 0.1

# 这两个会在函数里通过 startpoint.txt 赋值
START_POINT_X = 0.0
START_POINT_Y = 0.0

# @dataclass
# class HazardParams:
#     v_open: float = 0.06
#     v_build: float = 0.03
#     block_building: int = 0

#     Rmax: float = 10.0
#     tau_pre: float = 500.0
#     tau_rise: float = 20.0
#     R_block: float | None = None  # if None -> auto Rmax*0.99

#     alpha_pre: float = 0.1
#     INF: float = 1e30

@dataclass
class HazardParams:
    # v_open: float = 0.12
    # v_build: float = 0.06
    
    # LogNormal parameters for spread speed
    v_open_mu: float = 0.12
    v_open_sigma: float = 0.3

    v_build_mu: float = 0.06
    v_build_sigma: float = 0.02

    block_building: int = 0

    Rmax: float = 10.0
    tau_pre: float = 500.0
    tau_rise: float = 120.0
    R_block: float = 9.0          #| None = None  # if None -> auto Rmax*0.99

    alpha_pre: float = 0.3
    INF: float = 1e30
    
    v_open = random.lognormvariate(v_open_mu, v_open_sigma)
    v_build = random.lognormvariate(v_build_mu, v_build_sigma)

    # # sampled values
    # v_open: float = field(init=False)
    # v_build: float = field(init=False)

    # def __post_init__(self):
    #     self.v_open = random.lognormvariate(self.v_open_mu, self.v_open_sigma)
    #     self.v_build = random.lognormvariate(self.v_build_mu, self.v_build_sigma)

def _auto_rblock(hp: HazardParams) -> HazardParams:
    if hp.R_block is None:
        hp = replace(hp, R_block=hp.Rmax * 0.99)
    return hp

# ================= 工具函数 =================
def read_start_point(txt_path: str):
    """读取起点坐标 (World Mercator 米制单位)"""
    with open(txt_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        x, y = map(float, line.split())
    return x, y

def read_metadata(folder):
    """读取 C++ 生成的 meta.txt 文件（自动跳过非数字行）"""
    meta_path = os.path.join(folder, "meta.txt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到元数据文件: {meta_path}")
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过空行 或 注释行
            if not line or line.startswith("#"):
                continue
            # 必须包含 "="
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            # 包含 {} 的行跳过
            if "{" in val or "}" in val:
                continue
            # 尝试解析数字
            try:
                if "." in val:
                    meta[key] = float(val)
                else:
                    meta[key] = int(val)
            except ValueError:
                continue
    return meta


def downsample_grid(data, original_res, target_res, mode = 'mean'):
    """
    精度调节 (空间降采样)
    data: 2D numpy array (Height, Width)
    original_res: 原始格子大小 (如 0.2)
    target_res: 目标格子大小 (如 1.0)
    mode: 'mean' (密度/速度) 或 'sum' (人数计数)
    """
    factor = target_res / original_res
    
    # 如果倍数接近 1，直接返回
    if abs(factor - 1.0) < 1e-5:
        return data

    factor = int(round(factor))
    if factor <= 1:
        print(f"警告: 目标精度 {target_res} 小于或等于原始精度，不做处理。")
        return data

    h, w = data.shape
    new_h = h // factor
    new_w = w // factor

    # 裁切 (丢弃边缘无法整除的部分)
    crop_h = new_h * factor
    crop_w = new_w * factor
    cropped = data[:crop_h, :crop_w]

    # 升维
    reshaped = cropped.reshape(new_h, factor, new_w, factor)

    # 聚合
    if mode == 'mean':
        result = reshaped.mean(axis=(1, 3))
    elif mode == 'sum':
        result = reshaped.sum(axis=(1, 3))
    else:
        raise ValueError("Mode must be 'mean' or 'sum'")

    # print(f"   -> 降采样: 原始{data.shape} -> 因子{factor}x -> 结果{result.shape}")
    return result


def process_layer(bin_file, meta, output_file, target_resolution=None, agg_mode='mean'):
    """处理单个图层：读取 -> 翻转 -> 降采样 -> 保存 GeoTIFF"""
    global START_POINT_X, START_POINT_Y

    # 1. 读取二进制 (C++ float 对应 np.float32)
    raw_data = np.fromfile(bin_file, dtype=np.float32)
    
    w, h = meta['width'], meta['height']
    
    if raw_data.size != w * h:
        print(f"❌ 错误: 文件大小不匹配 {bin_file}")
        return

    # 2. 重塑为 2D 并垂直翻转
    data = raw_data.reshape((h, w))
    data = np.flipud(data)

    # 3. 精度调节
    current_res = meta['cellSize']
    final_res = current_res
    
    if target_resolution and target_resolution > current_res:
        data = downsample_grid(data, current_res, target_resolution, mode=agg_mode)
        final_res = target_resolution

    # 4. 计算新的地理变换参数 (Transform)
    local_x_min = meta['x_min']
    local_y_min = meta['y_min']
    
    world_x_min = START_POINT_X + local_x_min
    world_y_min_base = START_POINT_Y + local_y_min
    
    # 注意: y_max 使用原始高度和原始分辨率
    world_y_max = world_y_min_base + (h * current_res)
    
    transform = from_origin(world_x_min, world_y_max, final_res, final_res)

    # 5. 写入 GeoTIFF
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    out_dtype = np.float32
    
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=out_dtype,
        crs=RASTER_CRS,
        transform=transform,
        compress='lzw',
        nodata=-9999
    ) as dst:
        dst.write(data.astype(out_dtype), 1)
    # print(f"✅ 已保存: {output_file} (Res: {final_res}m, CRS: {RASTER_CRS})")


# ================= 核心封装函数 =================

def convert_bin_to_tif_for_project(project_name: str):
    """
    根据项目名（例如 'shipai_0'）：
    - 自动读取 ./input/<project_name>/startpoint.txt
    - 自动从 ./output/<project_name>/bin 读取 layer_*.bin
    - 输出到 ./output/<project_name>/tif/*.tif
    其他参数固定。
    """
    global START_POINT_X, START_POINT_Y

    # 1. 路径设置
    input_bin_folder   = f"../../results/simulation/{project_name}/bin"
    output_tif_folder  = f"../../results/simulation/{project_name}/tif"
    startpoint_txt     = f"../../data/simulation_inputs/{project_name}/startpoint.txt"

    os.makedirs(output_tif_folder, exist_ok=True)

    # 2. 读取起点坐标
    START_POINT_X, START_POINT_Y = read_start_point(startpoint_txt)
    # print(f"起点坐标加载成功: ({START_POINT_X}, {START_POINT_Y}) from {startpoint_txt}")

    # 3. 读取元数据
    meta_info = read_metadata(input_bin_folder)
    # print(f"元数据加载成功: 原始尺寸 {meta_info['width']}x{meta_info['height']}, 精度 {meta_info['cellSize']}m")

    # 4. 定义要处理的 layer 列表（你原来的 tasks 不变）
    tasks = [
        ("layer_0.bin", "0_boundary.tif",                 RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_1.bin", "1_building.tif",                 RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_2.bin", "2_roadnetwork.tif",              RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_3.bin", "3_roadpotential.tif",            RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_4.bin", "4_roaddistance.tif",             RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_5.bin", "5_roadpotentialgradient_X.tif",  RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_6.bin", "6_roadpotentialgradient_Y.tif",  RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_7.bin", "7_cumuusagefrequency.tif",       RASTER_RESOLUTION_METERS, 'sum'),
        ("layer_8.bin", "8_cumucongestiontime.tif",       RASTER_RESOLUTION_METERS, 'sum'),
        ("layer_9.bin", "9_cumucrowddensity.tif",         RASTER_RESOLUTION_METERS, 'sum'),
        ("layer_10.bin", "10_updateRoads.tif",             RASTER_RESOLUTION_METERS, 'sum'),
        ("layer_11.bin", "11_dynamic_density.tif",         RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_12.bin", "12_dynamic_cost.tif",            RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_13.bin", "13_disaster_source.tif",         RASTER_RESOLUTION_METERS, 'sum'),
        ("layer_14.bin", "14_disaster_arrival_time.tif",   RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_15.bin", "15_disaster_intensity.tif",      RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_16.bin", "16_disaster_hard_mask.tif",      RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_17.bin", "17_disaster_mask_road.tif",      RASTER_RESOLUTION_METERS, 'mean'),
        ("layer_18.bin", "18_population_distribution.tif",    RASTER_RESOLUTION_METERS, 'mean'),
    ]

    # 5. 循环处理各个图层
    for bin_name, tif_name, target_res, mode in tasks:
        bin_path = os.path.join(input_bin_folder, bin_name)
        tif_path = os.path.join(output_tif_folder, tif_name)
        
        if os.path.exists(bin_path):
            # print(f"正在处理: {bin_name} -> {tif_name} ({mode.upper()}, {target_res}m)")
            process_layer(bin_path, meta_info, tif_path, target_res, mode)
        # else:
        #     print(f"跳过: 找不到文件 {bin_path}")

    # print(f"[bin to tif] convert bin to tif for {project_name} done!")

# ================= 测试入口 =================
import pandas as pd
import matplotlib.pyplot as plt

def plot_and_analyze(project_name,
                     state_done=0,
                     dist_thresh=300.0,
                     time_thresh=400.0,
                     bins_time=40,
                     bins_dist=40,
                     out_png=None):
    # --- Load ---
    # "../../results/simulation/shipai_2250/run_record.csv"
    out_png="../../results/figures/" + project_name + "_hist_time_distance.png"
    csv_path = "../../results/simulation/" + project_name + "/run_record.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # --- Clean + filter completed ---
    df["state"] = pd.to_numeric(df["state"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    done = df[df["state"] == state_done].dropna(subset=["time", "distance"]).copy()
    done = done[(done["time"] >= 0) & (done["distance"] >= 0)]

    n = len(done)
    if n == 0:
        raise ValueError(f"No valid completed rows found (state={state_done}).")

    # --- Ratios ---
    p_dist = (done["distance"] <= dist_thresh).mean()
    p_time = (done["time"] <= time_thresh).mean()
    p_both = ((done["distance"] <= dist_thresh) & (done["time"] <= time_thresh)).mean()

    print(f"[hist_time_distance] Completed samples (state={state_done}): {n}")
    print(f"Share with distance <= {dist_thresh:.0f} m: {p_dist*100:.2f}%")
    print(f"Share with time <= {time_thresh:.0f} s: {p_time*100:.2f}%")
    print(f"Share with BOTH (distance <= {dist_thresh:.0f} m AND time <= {time_thresh:.0f} s): {p_both*100:.2f}%")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Time histogram
    ax1.hist(done["time"].to_numpy(), bins=bins_time)
    # ax1.axvline(time_thresh, linewidth=2, color="k")
    ax1.set_title("Evacuation Time Histogram (Completed)")
    ax1.set_xlabel("Evacuation Time (s)")
    ax1.set_ylabel("Number of People")
    ax1.grid(True, alpha=0.25)

    # Distance histogram
    ax2.hist(done["distance"].to_numpy(), bins=bins_dist)
    # ax2.axvline(dist_thresh, linewidth=2, color="k")
    ax2.set_title("Evacuation Distance Histogram (Completed)")
    ax2.set_xlabel("Evacuation Distance (m)")
    ax2.set_ylabel("Number of People")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print("Saved:", out_png)

    # plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def uniform_sample(df, max_points=5000):
    """
    对 DataFrame 进行均匀采样（等间隔采样）
    max_points: 最多保留多少个点
    """
    n = len(df)
    if n <= max_points:
        return df

    step = max(1, n // max_points)
    return df.iloc[::step].reset_index(drop=True)

def Draw_fundenmental_paradigm(project_name, csv_id):
    # --- 自动检测演示数据逻辑 (可选) ---
    # 如果该路径下没有文件，生成一个模拟文件以确保代码可运行

    file_path = "../../results/simulation/" + project_name + "/csv/" + str(csv_id) + ".csv"
    output_path = "../../results/figures/" + project_name + "_" + str(csv_id) + "_fundamental_diagram.png"

    if not os.path.exists(file_path):
        print(f"提示: 未找到文件 {file_path}，正在生成模拟数据用于演示...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 创建模拟数据: ID, State, Speed, Flow, Density
        # 基本交通流关系: Flow = Speed * Density
        mock_density = np.linspace(5, 120, 200)
        mock_speed = 120 * (1 - mock_density / 140) + np.random.normal(0, 5, 200) # 简单的线性跟驰模型 + 噪声
        mock_speed = np.maximum(mock_speed, 0) # 速度不能小于0
        mock_flow = mock_speed * mock_density
        
        mock_df = pd.DataFrame({
            'ID': range(1, 201),
            'State': [1]*190 + [0]*10, # 混入一些 State=0
            'Speed': mock_speed,
            'Flow': mock_flow,
            'Density': mock_density
        })
        mock_df.to_csv(file_path, index=False)
    # --------------------------------

    # 2. 读取并预处理数据 (保留你原本的逻辑)
    cleaned_data = []
    header = []

    try:
        # 尝试打开文件读取
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取所有行，去除空白字符
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            raise ValueError("File is empty")
        # Handle header
        header = lines[0].split(',')
        num_columns = len(header)

        # 处理数据行
        for line in lines[1:]:
            parts = line.split(',')
            # 确保行长度一致，不足的补 None，防止 pandas 报错
            if len(parts) < num_columns:
                parts += [None] * (num_columns - len(parts))
            # 如果行太长，截断（虽然这种情况少见）
            elif len(parts) > num_columns:
                parts = parts[:num_columns]
            cleaned_data.append(parts)
        print(f"Successfully read file: {file_path}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        cleaned_data = []
    except Exception as e:
        print(f"Error reading file: {e}")
        cleaned_data = []

    # Create DataFrame
    df = pd.DataFrame(cleaned_data, columns=header)

    # 3. Data Cleaning
    if not df.empty:
        # Convert required columns to numeric, coerce errors to NaN
        # 检查列是否存在，防止文件列名不匹配
        required_cols = ['State', 'Speed', 'Flow', 'Density']
        available_cols = [col for col in required_cols if col in df.columns]

        for col in available_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 过滤 State 为 0 的行
        # 注意：dropna 会移除转换失败产生的 NaN
        if 'State' in df.columns:
            df_clean = df[df['State'] != 0].dropna(subset=available_cols)
        else:
            df_clean = df.dropna(subset=available_cols)
    else:
        df_clean = pd.DataFrame()

    # 4. Visualization
    if not df_clean.empty and 'Density' in df_clean.columns:

        # ---------- 1️⃣ 只保留 State == 1 ----------
        df_valid = df_clean[df_clean['State'] == 1].copy()

        if df_valid.empty:
            raise ValueError("No data with State == 1 after cleaning.")

        # ---------- 2️⃣ 均匀采样 ----------
        df_sampled = uniform_sample(df_valid, max_points=6000)

        print(f"After uniform sampling: {len(df_sampled)} points")

        # ---------- 3️⃣ 绘图设置 ----------
        # plt.rcParams['font.sans-serif'] = [
        #     'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'
        # ]
        # plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('ggplot')

        # ---------- 4️⃣ 并排绘制两个子图 ----------
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- 左图：Speed vs Density ---
        axes[0].scatter(
            df_sampled['Density'],
            df_sampled['Speed'],
            c='tab:blue',
            marker='x',
            s=15,
            linewidths=0.8,
            alpha=0.9
        )

        # median_speed = df_sampled['Speed'].median()

        # Density 分成 10 个等宽区间
        bins = np.linspace(df_sampled['Density'].min(),
                        df_sampled['Density'].max(), 30)

        df_sampled['Density_bin'] = pd.cut(df_sampled['Density'], bins)

        # 每个密度区间的中位数
        median_speed_by_bin = df_sampled.groupby('Density_bin')['Speed'].median()

        # 区间中心
        bin_centers = [interval.mid for interval in median_speed_by_bin.index]

        axes[0].plot(
        bin_centers,
        median_speed_by_bin.values,
        color='black',
        linewidth=2,
        label='Median Speed (binned)')


        axes[0].set_title('Speed–Density Relationship')
        axes[0].set_xlabel('Density')
        axes[0].set_ylabel('Speed')
        axes[0].grid(True)
        axes[0].legend()

        # --- 右图：Flow vs Density ---
        axes[1].scatter(
            df_sampled['Density'],
            df_sampled['Flow'],
            c='tab:blue',
            marker='x',
            s=15,
            linewidths=0.8,
            alpha=0.9
        )

        median_flow_by_bin = df_sampled.groupby('Density_bin')['Flow'].median()

        axes[1].plot(
        bin_centers,
        median_flow_by_bin.values,
        color='black',
        linewidth=2,
        label='Median Flow (binned)'
        )

        axes[1].set_title('Flow–Density Relationship')
        axes[1].set_xlabel('Density')
        axes[1].set_ylabel('Flow')
        axes[1].grid(True)
        axes[1].legend()

        # ---------- 5️⃣ 保存 + 展示 ----------
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Figure saved to: {os.path.abspath(output_path)}")
        # plt.show()
    else:
        print("\nNo valid data available for plotting.")


import os
import random
import sys
import json
import geopandas as gpd
from pathlib import Path
from typing import Tuple, List, Optional, Union


class ShapefileProcessor:
    def __init__(
        self,
        project_name: str,
        input_root: str = "../example",
        output_root: str = "output",
        target_crs: str = "EPSG:3395",
        basic_input_file: str = "shipai_0",
        file_config: Optional[dict] = None,
    ):
        """
        初始化处理器
        :param project_name: 项目名称 (例如 "Shipai916")
        :param input_root: 兼容老版本的输入根目录（如果没用 JSON，可以继续使用）
        :param output_root: 输出结果的根目录
        :param target_crs: 目标投影坐标系
        :param file_config: JSON 中的 files 字段，用于指定 5 个 shp 的完整路径
        """
        self.project_name = project_name

        self.hazard_params = HazardParams()

        # 兼容老逻辑：默认输入目录
        basic_shp =  basic_input_file
        
        self.input_dir = Path(input_root) / basic_shp

        # 输出目录
        self.output_dir = Path(output_root) / project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_point: Tuple[float, float] = (0.0, 0.0)

        # 目标投影坐标系：World Mercator（单位：米）
        self.target_crs = target_crs

        # JSON 传入的 5 个文件路径
        # 结构要求：
        # {
        #   "startpoint": "...",
        #   "building": "...",
        #   "road": "...",
        #   "safearea": "...",
        #   "boundary": "..."
        # }
        self.file_config = file_config or {}

        # print(f"[*] 初始化完成，输出目录: {self.output_dir}")

    def normalize_coord(self, x: float, y: float) -> Tuple[float, float]:
        """将绝对坐标转换为相对坐标（相对于 start_point），单位：米"""
        return x - self.start_point[0], y - self.start_point[1]

    def _write_coords_to_file(self, file_handle, coords: List[Tuple[float, ...]]):
        """
        辅助函数：将坐标列表写入文件
        修复: 兼容 (x, y, z) 3D坐标，只取前两个值
        """
        for coord in coords:
            # 确保只取前两个值 (x, y)，忽略 z 轴或其他多余值
            x, y = coord[0], coord[1]
            nx, ny = self.normalize_coord(x, y)
            file_handle.write(f"{nx:.3f} {ny:.3f}\n")
        file_handle.write("\n")  # 每个要素后空一行

    # ========================
    # 统一读+投影函数
    # ========================
    def _read_and_project(self, shp_path: Path) -> gpd.GeoDataFrame:
        """
        读取 shp 并投影到目标坐标系（EPSG:3395，单位：m）
        假定原始是 WGS84 (EPSG:4326)，如果未设置 CRS，则强制认为是 4326。
        """
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            print(f"[!] 警告: {shp_path} 未设置 CRS，将假定为 EPSG:4326 (WGS84)")
            gdf = gdf.set_crs(epsg=4326)

        if gdf.crs.to_string() != self.target_crs:
            original_crs = gdf.crs
            gdf = gdf.to_crs(self.target_crs)
            # print(f"[+] 已将 {shp_path.name} 从 {original_crs} 投影为 {self.target_crs}")
        # else:
            # print(f"[*] {shp_path.name} 已经是 {self.target_crs}，无需投影")

        return gdf

    # ========================
    # 各层处理函数（改为支持传入路径）
    # ========================

    def process_start_point(self, filename: str = "startpoint.shp") -> None:
        """
        处理起始点文件，设定全局参考坐标
        注意：这里会先把点投影到目标 CRS，再作为起点（单位：米）

        优先使用 JSON 中的路径：self.file_config["startpoint"]
        否则退回到 input_root/project_name/filename
        """
        json_path = self.file_config.get("startpoint")
        if json_path:
            shp_path = Path(json_path)
        else:
            shp_path = self.input_dir / filename

        txt_path = self.output_dir / "startpoint.txt"

        if not shp_path.exists():
            print(f"[!] 警告: 起始点文件不存在 {shp_path}，将使用默认原点 (0,0)")
            self.start_point = (0.0, 0.0)
            return

        try:
            gdf = self._read_and_project(shp_path)
            if len(gdf) > 0 and gdf.geometry.iloc[0].geom_type == "Point":
                point = gdf.geometry.iloc[0]
                # 这里 start_point 已经是米单位的投影坐标
                self.start_point = (point.x, point.y)
                # print(f"[+] 提取起始点（投影到 {self.target_crs} 后）: {self.start_point}")

                # 写入 txt 的也是投影后的坐标（m）
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"{point.x:.3f} {point.y:.3f}\n")
            else:
                print("[!] 错误: startpoint.shp 中无数据或类型不为 Point")
        except Exception as e:
            print(f"[!] 处理起始点时发生异常: {e}")

    def process_buildings(
        self,
        filename: str = "building.shp",
        output_name: str = "building.txt",
    ) -> None:
        """
        处理建筑数据 (Polygon/MultiPolygon)
        格式: Building [点数] [层数] [面积]
        """
        self._process_polygon_layer(
            filename,
            output_name,
            type_label="Building",
            attr_getter=self._get_building_attrs,
            json_key="building",
        )


    def process_add_widths(
        self,
        filename: str = "add_width.shp",
        output_name: str = "addwidth.txt",
        values: list = None
    ) -> None:
        """
        处理建筑数据 (Polygon/MultiPolygon)
        格式: Building [点数] [层数] [面积]
        """
        self._process_polygon_layer(
            filename,
            output_name,
            type_label="addwidth",
            attr_getter=self._get_building_attrs,
            json_key="addwidth",
            label_values = values
        )

    def _get_building_attrs(self, row, geom) -> str:
        """
        获取建筑特定的属性格式字符串
        返回格式: "层数 面积"
        注意：此时 geom.area 是 m^2
        """
        # 尝试获取 Floor 或 floor 字段，默认值为 5
        floor = row.get("Floor", row.get("floor", 5))
        area = geom.area
        return f"{floor} {area:.3f}"

    # def process_boundaries(
    #     self,
    #     filename: str = "boundary.shp",
    #     output_name: str = "boundary.txt",
    # ) -> None:
    #     """
    #     处理边界数据 (Polygon/MultiPolygon)
    #     格式: Boundary [点数]
    #     """
    #     self._process_polygon_layer(
    #         filename,
    #         output_name,
    #         type_label="Boundary",
    #         attr_getter=lambda row, geom: "",
    #         json_key="boundary",
    #     )
    
    def process_boundaries(
        self,
        filename: str = "boundary.shp",
        output_name: str = "boundary.txt",
        cell_size: float = 0.1,
        canvas_buffer: float = 10.0,
    ) -> None:
        """
        输出格式：

        1
        Canvas width length
        Cell_size 0.1
        Boundary n
        x y
        ...
        """

        # 路径
        if "boundary" in self.file_config:
            shp_path = Path(self.file_config["boundary"])
        else:
            shp_path = self.input_dir / filename

        output_path = self.output_dir / output_name

        if not shp_path.exists():
            print(f"[!] boundary 不存在: {shp_path}")
            return

        try:
            gdf = self._read_and_project(shp_path)

            # 🔥 合并所有 polygon（关键！！！）
            geom = gdf.unary_union

            # 保证是 Polygon
            if geom.geom_type == "MultiPolygon":
                # 取最大区域
                geom = max(geom.geoms, key=lambda g: g.area)

            # 坐标
            coords = list(geom.exterior.coords[:-1])

            # 👉 计算 canvas
            _, _, maxx, maxy = geom.bounds
            canvas_width = (maxx - self.start_point[0]) + canvas_buffer
            canvas_length = (maxy - self.start_point[1]) + canvas_buffer

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("1\n")
                f.write(f"Canvas {canvas_width:.3f} {canvas_length:.3f}\n")
                f.write(f"Cell_size {cell_size:.3f}\n")
                f.write(f"Boundary {len(coords)}\n")

                self._write_coords_to_file(f, coords)

            # print(f"[✓] Boundary 写入完成: {output_path}")

        except Exception as e:
            print(f"[!] boundary 处理失败: {e}")

    def _process_polygon_layer(
        self,
        filename: str,
        output_name: str,
        type_label: str,
        attr_getter,
        json_key: Optional[str] = None,
        label_values: Optional[list] = None,  # ⭐ 新增可选参数：额外 value 列表
    ):
        """
        通用的多边形处理逻辑 (用于 Building 和 Boundary)

        :param json_key: JSON 中对应的 key，如 "building" / "boundary"
        :param label_values: 每个要素对应一个数字，用来覆盖
                            'type_label 点数 extra_info...' 中 extra_info 的第一个数字
                            例如:
                            原本: Building 4 5 116.326
                            覆盖后: Building 4 X 116.326   (X 来自 label_values)
        """

        # 优先使用 JSON 提供的路径
        if json_key and json_key in self.file_config:
            shp_path = Path(self.file_config[json_key])
        else:
            shp_path = self.input_dir / filename

        output_path = self.output_dir / output_name

        if not shp_path.exists():
            print(f"[!] 跳过: 文件不存在 {shp_path}")
            return

        try:
            gdf = self._read_and_project(shp_path)

            # ⭐ 检查长度是否匹配
            if label_values is not None and len(label_values) != len(gdf):
                raise ValueError(
                    f"label_values 长度({len(label_values)}) != 要素数量({len(gdf)})"
                )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"{len(gdf)}\n")  # 写入总要素数

                count = 0
                # 用 enumerate 确保顺序跟 label_values 对应
                for feat_idx, (_, row) in enumerate(gdf.iterrows()):
                    geom = row["geometry"]

                    # 原本的 extra_info，比如 "5 116.326" 或者 "5"
                    extra_info = attr_getter(row, geom)

                    # 统一转成字符串处理
                    extra_str = ""
                    if extra_info is not None and extra_info != "":
                        extra_info_str = str(extra_info).strip()
                        if extra_info_str:
                            parts = extra_info_str.split()

                            # ⭐ 如果有 label_values，就覆盖 parts[0]
                            if label_values is not None and len(parts) > 0:
                                parts[0] = str(label_values[feat_idx])

                            new_extra_info_str = " ".join(parts)
                            extra_str = f" {new_extra_info_str}"

                    # ⭐ 如果 extra_info 原本为空，但你给了 label_values
                    elif label_values is not None:
                        extra_str = f" {label_values[feat_idx]}"

                    geoms_to_process = []
                    if geom.geom_type == "Polygon":
                        geoms_to_process.append(geom)
                    elif geom.geom_type == "MultiPolygon":
                        geoms_to_process.extend(geom.geoms)

                    for poly in geoms_to_process:
                        coords = list(poly.exterior.coords[:-1])

                        # 点数还是用真实点数
                        n_points = len(coords)

                        # 输出: type_label 点数 + 覆盖后的 extra_info
                        # 例如: Building 4 3 116.326
                        f.write(f"{type_label} {n_points}{extra_str}\n")

                        self._write_coords_to_file(f, coords)
                        count += 1

            # print(f"[+] 成功写入 {output_path} / {output_name} (处理要素: {count})")

        except Exception as e:
            print(f"[!] 处理 {filename} 时出错: {e}")

    def process_roads(
        self,
        filename: str = "road.shp",
        output_name: str = "road.txt",
    ) -> None:
        """
        处理道路数据 (LineString/MultiLineString)
        格式: Road [点数] [宽度]
        """
        # 优先使用 JSON 提供的路径
        if "road" in self.file_config:
            shp_path = Path(self.file_config["road"])
        else:
            shp_path = self.input_dir / filename

        output_path = self.output_dir / output_name

        if not shp_path.exists():
            print(f"[!] 跳过: 文件不存在 {shp_path}")
            return

        try:
            gdf = self._read_and_project(shp_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"{len(gdf)}\n")

                for _, row in gdf.iterrows():
                    geom = row["geometry"]
                    width = row.get("W", 3)  # 默认宽度 3 (单位：米)

                    geoms_to_process = []
                    if geom.geom_type == "LineString":
                        geoms_to_process.append(geom)
                    elif geom.geom_type == "MultiLineString":
                        geoms_to_process.extend(geom.geoms)

                    for line in geoms_to_process:
                        coords = list(line.coords)
                        f.write(f"Road {len(coords)} {width}\n")
                        self._write_coords_to_file(f, coords)

            # print(f"[+] 成功写入 {output_path}/{output_name}")
        except Exception as e:
            print(f"[!] 处理 {filename} 时出错: {e}")

    from pathlib import Path
    import random


    def process_safe_areas(
        self,
        filename: str = "safearea.shp",
        output_name: str = "safearea.txt",
        keep_prob: float = 1.0,
    ):
        """
        处理安全区域 (主要针对 Point / MultiPoint)

        新增功能：
        - 保存出口开关状态
        self.safe_exit_switches   -> [1,0,1,...]
        self.safe_exit_switch_dict -> {0:1,1:0,...}
        self.n_open_exits
        self.n_total_exits

        返回：
        - exit_switches: List[int]
        - exit_switch_dict: Dict[int, int]
        """

        if "safearea" in self.file_config:
            shp_path = Path(self.file_config["safearea"])
        else:
            shp_path = self.input_dir / filename

        output_path = self.output_dir / output_name

        if not shp_path.exists():
            print(f"[!] 跳过: 文件不存在 {shp_path}")
            self.safe_exit_switches = []
            self.safe_exit_switch_dict = {}
            self.n_open_exits = 0
            self.n_total_exits = 0
            return [], {}

        try:
            gdf = self._read_and_project(shp_path)

            kept_entries = []
            exit_switches = []
            exit_switch_dict = {}

            exit_idx = 0  # 从 0 开始编号，便于后面记录 {0,1,...}

            for _, row in gdf.iterrows():
                geom = row["geometry"]
                width = row.get("Width", 3.0)
                value = row.get("Value", 1.0)

                # ---- 单点出口 ----
                if geom.geom_type == "Point":
                    keep_flag = 1 if random.random() < keep_prob else 0
                    exit_switches.append(keep_flag)
                    exit_switch_dict[exit_idx] = keep_flag

                    if keep_flag == 1:
                        kept_entries.append({
                            "coords": [(geom.x, geom.y)],
                            "width": width,
                            "value": value,
                            "exit_idx": exit_idx,
                        })

                    exit_idx += 1

                # ---- 多点出口 ----
                elif geom.geom_type == "MultiPoint":
                    for pt in geom.geoms:
                        keep_flag = 1 if random.random() < keep_prob else 0
                        exit_switches.append(keep_flag)
                        exit_switch_dict[exit_idx] = keep_flag

                        if keep_flag == 1:
                            kept_entries.append({
                                "coords": [(pt.x, pt.y)],
                                "width": width,
                                "value": value,
                                "exit_idx": exit_idx,
                            })

                        exit_idx += 1

            # 保存到 self
            self.safe_exit_switches = exit_switches
            self.safe_exit_switch_dict = exit_switch_dict
            self.n_open_exits = int(sum(exit_switches))
            self.n_total_exits = int(len(exit_switches))

            # 写 safearea.txt
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"{len(kept_entries)}\n")
                for item in kept_entries:
                    coords = item["coords"]
                    width = item["width"]
                    value = item["value"]

                    f.write(f"SafeArea {len(coords)} {width:.1f} {value:.1f}\n")
                    self._write_coords_to_file(f, coords)

            return exit_switches, exit_switch_dict

        except Exception as e:
            print(f"[!] 处理 {filename} 时出错: {e}")
            self.safe_exit_switches = []
            self.safe_exit_switch_dict = {}
            self.n_open_exits = 0
            self.n_total_exits = 0
            return [], {}

    def write_crowd_config(self, num):
        """
        在项目路径下生成 crowd.txt 文件。
        """
        # 注意：根据你的逻辑，输入文件似乎是在 INPUT_PATH 下
        config_path = os.path.join("../../data/simulation_inputs", self.project_name, "crowd.txt")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"Crowd {num} 1\n")  # 修正为 num
            f.write("Male 1 30\n")
        # print(f"Add crowd: {config_path}")
        
        
    def random_point_in_polygon(self, poly, max_attempts=10000):
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(max_attempts):
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            p = Point(x, y)
            if poly.contains(p):
                return p
        raise RuntimeError("无法在 polygon 内采样到随机点")

    def random_line_in_polygon(self, poly, min_length_ratio=0.15, max_attempts=1000):
        minx, miny, maxx, maxy = poly.bounds
        diag = math.hypot(maxx - minx, maxy - miny)
        min_len = diag * min_length_ratio

        for _ in range(max_attempts):
            p1 = self.random_point_in_polygon(poly)
            p2 = self.random_point_in_polygon(poly)
            line = LineString([p1, p2])
            if line.length >= min_len:
                return line
        raise RuntimeError("无法在 polygon 内生成足够长的随机线段")

    def normalize_weights(self, ws):
        s = sum(ws)
        return [w / s for w in ws]
    



    def generate_population_distribution_from_boundary_2(
        self,
        boundary_shp: str = "boundary.shp",
        output_name: str = "population_distribution.txt",
        mode_id: int = 1,
        seed=None,
        uniform_base: float = 1.0,
        center_sigma_ratio: float = 0.18,
        center_weight: float = 1.0,
        ring_radius_ratio: float = 0.50,
        ring_sigma_ratio: float = 0.08,
        ring_weight: float = 1.0,
        ring_point_count: int = 8,
        ring_angle_offset_deg: float = 0.0,
        ):

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        mode_map = {
            1: "uniform",
            2: "center",
            3: "ring",
        }

        if mode_id not in mode_map:
            raise ValueError(f"Unsupported mode_id={mode_id}, must be 1/2/3")

        mode = mode_map[mode_id]

        if "boundary" in self.file_config:
            shp_path = Path(self.file_config["boundary"])
        else:
            shp_path = self.input_dir / boundary_shp

        if "population_distribution" in self.file_config:
            output_path = Path(self.file_config["population_distribution"])
        else:
            output_path = self.output_dir / output_name

        output_path.parent.mkdir(parents=True, exist_ok=True)

        gdf = self._read_and_project(shp_path)
        if gdf is None or gdf.empty:
            raise ValueError(f"boundary 文件为空或读取失败: {shp_path}")

        geom_types = gdf.geometry.geom_type
        boundary_gdf = gdf[geom_types.isin(["Polygon", "MultiPolygon"])].copy()
        if boundary_gdf.empty:
            raise ValueError("boundary.shp 中没有 Polygon / MultiPolygon 要素")

        geom = boundary_gdf.geometry.union_all()
        if geom.is_empty:
            raise ValueError("boundary 几何为空")

        if not isinstance(geom, (Polygon, MultiPolygon)):
            raise ValueError("boundary union 之后不是 Polygon / MultiPolygon")

        minx, miny, maxx, maxy = geom.bounds
        W = maxx - minx
        H = maxy - miny
        D = math.hypot(W, H)

        if D <= 0:
            raise ValueError("boundary 范围异常，无法计算有效尺度")

        x0, y0 = self.start_point

        def _safe_center_point(poly_geom):
            c = poly_geom.centroid
            if poly_geom.contains(c):
                return c
            return poly_geom.representative_point()

        def _extract_boundary_coords(poly_geom):
            coords = []
            if isinstance(poly_geom, Polygon):
                coords.extend(list(poly_geom.exterior.coords))
            elif isinstance(poly_geom, MultiPolygon):
                for g in poly_geom.geoms:
                    coords.extend(list(g.exterior.coords))
            return coords

        def _effective_radius(poly_geom, center_pt):
            coords = _extract_boundary_coords(poly_geom)
            if not coords:
                return 0.25 * min(W, H)

            cx, cy = center_pt.x, center_pt.y
            dists = [math.hypot(x - cx, y - cy) for x, y in coords]
            dists = np.asarray(dists, dtype=float)

            r_eff = float(np.percentile(dists, 75))
            r_eff = max(r_eff, 0.15 * min(W, H))
            r_eff = min(r_eff, 0.50 * D)
            return r_eff

        def _point_inside_or_shrink(poly_geom, center_pt, angle_rad, target_radius, shrink_steps=24):
            cx, cy = center_pt.x, center_pt.y
            for t in np.linspace(1.0, 0.1, shrink_steps):
                rr = target_radius * float(t)
                px = cx + rr * math.cos(angle_rad)
                py = cy + rr * math.sin(angle_rad)
                p = Point(px, py)
                if poly_geom.contains(p):
                    return p
            return center_pt

        enable_uniform = 0
        enable_gradient = 0
        enable_point = 0
        enable_line = 0

        w_uniform = 0.0
        w_gradient = 0.0
        w_point = 0.0
        w_line = 0.0

        gradient_direction = "NORTH_SOUTH"
        gradient_alpha = 0.0

        point_sources = []
        point_count = 0

        line_sources = []
        line_source_sigma = 0.0
        line_source_weight = 0.0

        if mode == "uniform":
            enable_uniform = 1
            w_uniform = 1.0

        elif mode == "center":
            enable_point = 1
            w_point = 1.0

            cpt = _safe_center_point(geom)
            sigma = max(0.01, center_sigma_ratio * D)

            point_sources.append((
                round(cpt.x - x0, 2),
                round(cpt.y - y0, 2),
                round(sigma, 2),
                round(center_weight, 2),
            ))
            point_count = 1

        elif mode == "ring":
            enable_point = 1
            w_point = 1.0

            cpt = _safe_center_point(geom)
            r_eff = _effective_radius(geom, cpt)
            ring_radius = ring_radius_ratio * r_eff
            sigma = max(0.01, ring_sigma_ratio * D)

            angle0 = math.radians(ring_angle_offset_deg)
            n = max(4, int(ring_point_count))

            for i in range(n):
                theta = angle0 + 2.0 * math.pi * i / n
                p = _point_inside_or_shrink(geom, cpt, theta, ring_radius)
                point_sources.append((
                    round(p.x - x0, 2),
                    round(p.y - y0, 2),
                    round(sigma, 2),
                    round(ring_weight, 2),
                ))

            point_count = len(point_sources)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# ==================================================\n")
            f.write("# Population macro-distribution configuration\n")
            f.write("# Auto-generated from boundary.shp\n")
            f.write("# ==================================================\n\n")

            f.write(f"# MODE_ID = {mode_id}\n")
            f.write(f"# MODE = {mode}\n")
            f.write(f"# BOUNDARY_WIDTH  = {W:.3f}\n")
            f.write(f"# BOUNDARY_HEIGHT = {H:.3f}\n")
            f.write(f"# BOUNDARY_DIAG   = {D:.3f}\n\n")

            f.write("# ---------- Global switch ----------\n")
            f.write(f"ENABLE_UNIFORM      = {enable_uniform}\n")
            f.write(f"ENABLE_GRADIENT     = {enable_gradient}\n")
            f.write(f"ENABLE_POINT_SOURCE = {enable_point}\n")
            f.write(f"ENABLE_LINE_SOURCE  = {enable_line}\n\n")

            f.write("# ---------- Mixture weights ----------\n")
            f.write(f"W_UNIFORM      = {w_uniform:.4f}\n")
            f.write(f"W_GRADIENT     = {w_gradient:.4f}\n")
            f.write(f"W_POINT_SOURCE = {w_point:.4f}\n")
            f.write(f"W_LINE_SOURCE  = {w_line:.4f}\n\n")

            f.write("# ---------- Uniform ----------\n")
            f.write(f"UNIFORM_BASE = {uniform_base:.2f}\n\n")

            f.write("# ---------- Gradient ----------\n")
            f.write(f"GRADIENT_DIRECTION = {gradient_direction}\n")
            f.write(f"GRADIENT_ALPHA = {gradient_alpha:.2f}\n\n")

            f.write("# ---------- Point sources ----------\n")
            f.write(f"POINT_SOURCE_COUNT = {point_count}\n")
            for i, (x, y, sigma, weight) in enumerate(point_sources):
                f.write(f"POINT_SOURCE_{i} = {x:.2f}, {y:.2f}, {sigma:.2f}, {weight:.2f}\n")
            f.write("\n")

            f.write("# ---------- Line sources ----------\n")
            f.write(f"LINE_SOURCE_SIGMA = {line_source_sigma:.2f}\n")
            f.write(f"LINE_SOURCE_WEIGHT = {line_source_weight:.2f}\n")
            for i, (x1, y1, x2, y2) in enumerate(line_sources):
                f.write(f"LINE_{i} = {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

    def generate_population_distribution_from_boundary(
        self,
        boundary_shp: str = "boundary.shp",
        output_name: str = "population_distribution.txt",
        seed=None,
    ) -> None:
        """
        根据 boundary.shp 随机生成 population_distribution.txt

        特点：
        1. 自动适应空间尺度
        2. 随机启用 4 类分布中的至少 2 类
        3. 点源、线源在 boundary 内随机生成
        4. 输出坐标统一转成相对 self.start_point 的坐标
        """

        # 1) 输入输出路径
        if "boundary" in self.file_config:
            shp_path = Path(self.file_config["boundary"])
        else:
            shp_path = self.input_dir / boundary_shp

        if "population_distribution" in self.file_config:
            output_path = Path(self.file_config["population_distribution"])
        else:
            output_path = self.output_dir / output_name

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            random.seed(seed)

        # 2) 读取边界
        gdf = self._read_and_project(shp_path)
        if gdf is None or gdf.empty:
            raise ValueError(f"boundary 文件为空或读取失败: {shp_path}")

        geom_types = gdf.geometry.geom_type
        boundary_gdf = gdf[geom_types.isin(["Polygon", "MultiPolygon"])].copy()
        if boundary_gdf.empty:
            raise ValueError("boundary.shp 中没有 Polygon / MultiPolygon 要素")

        geom = boundary_gdf.geometry.union_all()
        if geom.is_empty:
            raise ValueError("boundary 几何为空")

        # 3) 统一用原始坐标计算空间尺度
        minx, miny, maxx, maxy = geom.bounds
        W = maxx - minx
        H = maxy - miny
        D = math.hypot(W, H)

        if D <= 0:
            raise ValueError("boundary 范围异常，无法计算有效尺度")

        # 相对坐标基准
        x0, y0 = self.start_point

        # 4) 启用开关：至少保留两类
        enable_uniform = random.choice([0, 1])
        enable_gradient = random.choice([0, 1])
        enable_point = random.choice([0, 1])
        enable_line = random.choice([0, 1])

        enabled = [enable_uniform, enable_gradient, enable_point, enable_line]
        if sum(enabled) < 2:
            idxs = random.sample(range(4), 2)
            enabled = [1 if i in idxs else 0 for i in range(4)]

        enable_uniform, enable_gradient, enable_point, enable_line = enabled

        # 5) 权重随机生成并归一化
        raw_weights = [random.uniform(0.2, 1.5) if flag else 0.0 for flag in enabled]
        positive_weights = [w for w in raw_weights if w > 0]
        norm_weights = self.normalize_weights(positive_weights)

        w_uniform = w_gradient = w_point = w_line = 0.0
        k = 0
        if enable_uniform:
            w_uniform = norm_weights[k]
            k += 1
        if enable_gradient:
            w_gradient = norm_weights[k]
            k += 1
        if enable_point:
            w_point = norm_weights[k]
            k += 1
        if enable_line:
            w_line = norm_weights[k]
            k += 1

        # 6) Uniform
        uniform_base = round(random.uniform(0.8, 1.2), 2)

        # 7) Gradient
        gradient_direction = random.choice(["NORTH_SOUTH", "EAST_WEST"])
        gradient_alpha = round(random.uniform(0.4, 1.8) * random.choice([-1, 1]), 2)

        # 8) Point sources
        point_sources = []
        point_count = 0
        if enable_point:
            point_count = random.randint(1, 5)
            for _ in range(point_count):
                p = self.random_point_in_polygon(geom)
                sigma = round(random.uniform(0.05 * D, 0.18 * D), 2)
                weight = round(random.uniform(0.5, 1.5), 2)

                # 输出为相对坐标
                point_sources.append((
                    round(p.x - x0, 2),
                    round(p.y - y0, 2),
                    sigma,
                    weight
                ))

        # 9) Line sources
        line_sources = []
        line_source_sigma = 0.0
        line_source_weight = 0.0

        if enable_line:
            line_count = random.randint(1, 4)
            for _ in range(line_count):
                line = self.random_line_in_polygon(geom, min_length_ratio=0.15)
                x1, y1 = line.coords[0]
                x2, y2 = line.coords[-1]

                # 输出为相对坐标
                line_sources.append((
                    round(x1 - x0, 2),
                    round(y1 - y0, 2),
                    round(x2 - x0, 2),
                    round(y2 - y0, 2),
                ))

            line_source_sigma = round(random.uniform(0.08 * D, 0.25 * D), 2)
            line_source_weight = round(random.uniform(0.8, 2.0), 2)

        # 10) 写入 txt
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# ==================================================\n")
            f.write("# Population macro-distribution configuration\n")
            f.write("# Auto-generated from boundary.shp\n")
            f.write("# ==================================================\n\n")

            f.write("# ---------- Global switch ----------\n")
            f.write(f"ENABLE_UNIFORM      = {enable_uniform}\n")
            f.write(f"ENABLE_GRADIENT     = {enable_gradient}\n")
            f.write(f"ENABLE_POINT_SOURCE = {enable_point}\n")
            f.write(f"ENABLE_LINE_SOURCE  = {enable_line}\n\n")

            f.write("# ---------- Mixture weights (normalized) ----------\n")
            f.write(f"W_UNIFORM      = {w_uniform:.4f}\n")
            f.write(f"W_GRADIENT     = {w_gradient:.4f}\n")
            f.write(f"W_POINT_SOURCE = {w_point:.4f}\n")
            f.write(f"W_LINE_SOURCE  = {w_line:.4f}\n\n")

            f.write("# ==================================================\n")
            f.write("# Uniform distribution\n")
            f.write("# ==================================================\n")
            f.write(f"UNIFORM_BASE = {uniform_base:.2f}\n\n")

            f.write("# ==================================================\n")
            f.write("# Gradient distribution\n")
            f.write("# ==================================================\n")
            f.write(f"GRADIENT_DIRECTION = {gradient_direction}\n")
            f.write(f"GRADIENT_ALPHA = {gradient_alpha:.2f}\n\n")

            f.write("# ==================================================\n")
            f.write("# Point-source distribution\n")
            f.write("# ==================================================\n")
            f.write(f"POINT_SOURCE_COUNT = {point_count}\n")
            for i, (x, y, sigma, weight) in enumerate(point_sources):
                f.write(f"POINT_SOURCE_{i} = {x:.2f}, {y:.2f}, {sigma:.2f}, {weight:.2f}\n")
            f.write("\n")

            f.write("# ==================================================\n")
            f.write("# Line-source distribution\n")
            f.write("# ==================================================\n")
            f.write(f"LINE_SOURCE_SIGMA = {line_source_sigma:.2f}\n")
            f.write(f"LINE_SOURCE_WEIGHT = {line_source_weight:.2f}\n")
            for i, (x1, y1, x2, y2) in enumerate(line_sources):
                f.write(f"LINE_{i} = {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        # print(f"[+] 已生成 population distribution 配置文件: {output_path}")
    
    from pathlib import Path
    from typing import Optional, Sequence, List, Tuple, Dict
    import random


    def process_disaster_points_by_mode(
        self,
        shp_map: Optional[Dict[int, str]] = None,
        hazard_class_id: Optional[int] = None,
        disaster_id: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        write_with_index: bool = True,
    ) -> Tuple[int, str, int, List[Tuple[float, float]]]:
        """
        自动从 3 个灾害 shp 文件中选择一种模式，再在对应 shp 中选择具体点，
        并写出 disastersource.txt。

        流程：
        1) 先选择 hazard_class_id（1/2/3）
        2) 自动读取对应 shp 文件
        3) 自动统计该 shp 中 Point 数量
        4) 若未指定 disaster_id，则随机选 1 个点（1-based）
        5) 自动减去 self.start_point
        6) 写出 disastersource.txt
        7) 返回 hazard_class_id, hazard_class_name, disaster_id, coords

        参数
        ----
        shp_map : dict[int, str] | None
            三类灾害 shp 文件映射，例如：
            {
                1: "disastersource_near_exit.shp",
                2: "disastersource_near_mainroad.shp",
                3: "disastersource_far.shp",
            }
            若为 None，则使用默认文件名

        hazard_class_id : int | None
            若提供，则固定使用该类（1/2/3）
            否则自动随机选择

        disaster_id : int | None
            若提供，则固定选择该点（1-based）
            否则自动随机选择

        seed : int | None
            随机种子

        write_with_index : bool
            输出 txt 时是否写索引

        返回
        ----
        hazard_class_id : int
        hazard_class_name : str
        disaster_id : int          # 1-based
        coords : List[(x, y)]      # 已减去 self.start_point
        """

        if seed is not None:
            random.seed(seed)

        if shp_map is None:
            shp_map = {
                1: "disastersource_near_exit.shp",
                2: "disastersource_near_mainroad.shp",
                3: "disastersource_far.shp",
            }

        class_name_map = {
            1: "near_exit",
            2: "near_main_road",
            3: "far_from_exit_and_main_road",
        }

        # 1) 先选灾害类别
        if hazard_class_id is None:
            hazard_class_id = random.randint(1, 3)

        if hazard_class_id not in shp_map:
            raise ValueError(f"hazard_class_id={hazard_class_id} 不在 shp_map 中")

        hazard_class_name = class_name_map.get(hazard_class_id, f"class_{hazard_class_id}")

        # 2) 路径解析
        shp_name = shp_map[hazard_class_id]
        shp_path = Path(shp_name)
        if not shp_path.is_absolute():
            shp_path = self.input_dir / shp_name

        txt_path = self.output_dir / "disastersource.txt"

        # 同时更新 file_config，便于后续追踪
        self.file_config["disastersource"] = str(shp_path)

        # 3) 文件存在性检查
        if not shp_path.exists():
            raise FileNotFoundError(f"灾害源文件不存在: {shp_path}")

        try:
            # 4) 读取 + 投影
            gdf = self._read_and_project(shp_path)

            if gdf is None or len(gdf) == 0:
                raise ValueError(f"shp 读取为空: {shp_path}")

            geom_types = gdf.geometry.geom_type
            points_gdf = gdf[geom_types == "Point"].copy()

            if len(points_gdf) == 0:
                raise TypeError(f"shp 中没有 Point 类型要素: {shp_path}")

            points_gdf = points_gdf.reset_index(drop=True)
            total = len(points_gdf)

            # 5) 自动选择具体点（1-based）
            if disaster_id is None:
                disaster_id = random.randint(1, total)
            else:
                disaster_id = int(disaster_id)
                if disaster_id < 1 or disaster_id > total:
                    raise ValueError(f"disaster_id={disaster_id} 越界，有效范围 [1, {total}]")

            # 转成 0-based 选取
            idx0 = disaster_id - 1
            selected = points_gdf.iloc[[idx0]]

            # 6) 提取坐标，并减去 self.start_point
            coords: List[Tuple[float, float]] = []
            for geom in selected.geometry:
                coords.append((
                    float(geom.x) - self.start_point[0],
                    float(geom.y) - self.start_point[1]
                ))

            if len(coords) == 0:
                raise ValueError("选中的灾害点数量为 0")

            self.disaster_points = coords

            # 7) 写 txt
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"{len(coords)}\n")
                for (x, y) in coords:
                    if write_with_index:
                        f.write(f"{idx0} {x:.3f} {y:.3f}\n")
                    else:
                        f.write(f"{x:.3f} {y:.3f}\n")

            return hazard_class_id, hazard_class_name, disaster_id, coords

        except Exception as e:
            print(f"[!] process_disaster_points_by_mode 处理异常: {e}")
            self.disaster_points = [(300.0, 400.0)]
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("1\n")
                if write_with_index:
                    f.write("0 300.000 400.000\n")
                else:
                    f.write("300.000 400.000\n")
            return hazard_class_id, hazard_class_name, 1, self.disaster_points

    def process_boundary_random_point(
        self,
        filename: str = "boundary.shp",
        output_name: str = "disastersource.txt",
        *,
        seed: Optional[int] = None,
        write_with_index: bool = True,
        max_attempts: int = 10000,
    ) -> None:
        """
        读取 boundary.shp，在 Polygon / MultiPolygon 内随机选取一个点并写入 txt。

        输出
        ----
        - self.disaster_points: List[Tuple[float, float]]
        - self.start_point: Tuple[float, float]
        - output_dir/output_name
        """

        if seed is not None:
            random.seed(seed)

        json_path = self.file_config.get("boundary")
        shp_path = Path(json_path) if json_path else (self.input_dir / filename)
        txt_path = self.output_dir / output_name

        if not shp_path.exists():
            print(f"[!] 警告: boundary 文件不存在 {shp_path}，将使用默认原点 (300,400)")
            self.disaster_points = [(300.0, 400.0)]
            self.start_point = self.disaster_points[0]

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("1\n")
                if write_with_index:
                    f.write("0 300.000 400.000\n")
                else:
                    f.write("300.000 400.000\n")
            return

        try:
            gdf = self._read_and_project(shp_path)

            if gdf is None or len(gdf) == 0:
                raise ValueError("boundary shp 读取为空或无要素。")

            geom_types = gdf.geometry.geom_type
            boundary_gdf = gdf[geom_types.isin(["Polygon", "MultiPolygon"])].copy()

            if len(boundary_gdf) == 0:
                raise TypeError("boundary.shp 中没有 Polygon 或 MultiPolygon 要素。")

            boundary_gdf = boundary_gdf.reset_index(drop=True)
            merged_geom = boundary_gdf.geometry.union_all()

            if merged_geom.is_empty:
                raise ValueError("boundary 合并后为空几何。")

            # 在原始坐标系中采样
            minx, miny, maxx, maxy = merged_geom.bounds

            selected_point = None
            for _ in range(max_attempts):
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                pt = Point(x, y)

                # contains 不包含边界点；covers 更稳一些
                if merged_geom.covers(pt):
                    selected_point = pt
                    break

            if selected_point is None:
                raise RuntimeError(
                    f"随机采样失败：在 {max_attempts} 次尝试内未能在 boundary 内找到有效点。"
                )

            # 转成相对 self.start_point 的坐标再写出
            sx, sy = self.start_point
            coords = [(float(selected_point.x) - sx, float(selected_point.y) - sy)]

            self.disaster_points = coords
            # 注意：这里不要再把 start_point 改成新采样点
            # self.start_point 保持原来的基准点不变

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("1\n")
                if write_with_index:
                    f.write(f"0 {coords[0][0]:.3f} {coords[0][1]:.3f}\n")
                else:
                    f.write(f"{coords[0][0]:.3f} {coords[0][1]:.3f}\n")

        except Exception as e:
            print(f"[!] 处理 boundary 随机点时发生异常: {e}")
            self.disaster_points = [(300.0, 400.0)]

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("1\n")
                if write_with_index:
                    f.write("0 300.000 400.000\n")
                else:
                    f.write("300.000 400.000\n")

 
    
    def preset(self, mode: str) -> HazardParams:
        """
        Define three presets.
        You can tune these numbers any time.
        """
        mode = mode.lower().strip()
        if mode == "fire":
            # Fire: usually block buildings (hard obstacle), fast rise
            hp = HazardParams(
                v_open=0.10,          # m/s
                v_build=0.00,         # not used since block_building=1
                block_building=1,
                Rmax=10.0,
                tau_pre=0.0,          # fire often has little "pre-exposure"
                tau_rise=10.0,
                alpha_pre=0.0,
                INF=1e30,
            )
        elif mode == "flood":
            # Flood: can block buildings, slower rise than fire
            hp = HazardParams(
                v_open=0.06,
                v_build=0.00,
                block_building=1,
                Rmax=10.0,
                tau_pre=0.0,
                tau_rise=30.0,
                alpha_pre=0.0,
                INF=1e30,
            )
        elif mode == "smoke":
            # Smoke: allow "penetration" through buildings with v_build, has pre-exposure
            hp = HazardParams(
                v_open=0.06,
                v_build=0.03,
                block_building=0,
                Rmax=10.0,
                tau_pre=500.0,
                tau_rise=20.0,
                alpha_pre=0.1,
                INF=1e30,
            )
        elif mode == "custom":
            # Custom: placeholder for user-defined parameters
            hp = HazardParams(
                v_open=0.10,
                v_build=0.05,
                block_building=0,
                Rmax=10.0,
                tau_pre=500.0,
                tau_rise=120.0,
                alpha_pre=0.30,
                INF=1e30,
            )
        elif mode == "custom2":
            # Custom: placeholder for user-defined parameters
            hp = HazardParams(
                v_open = 0.10,
                v_build = 0.05,
                block_building = 0,
                Rmax = 10.0,
                tau_pre = 500.0,
                tau_rise = 120.0,
                alpha_pre = 0.3,
                INF = 1e30,
            )
        elif mode == "dis_exp_1":
            # Custom: placeholder for user-defined parameters
            hp = HazardParams(
                v_open = 0.05,
                v_build = 0.03,
                block_building = 0,
                Rmax = 10.0,
                tau_pre = 500.0,
                tau_rise = 120.0,
                alpha_pre = 0.3,
                INF = 1e30,
            )
        elif mode == "dis_exp_2":
            # Custom: placeholder for user-defined parameters
            hp = HazardParams(
                v_open = 0.10,
                v_build = 0.05,
                block_building = 0,
                Rmax = 10.0,
                tau_pre = 500.0,
                tau_rise = 120.0,
                alpha_pre = 0.3,
                INF = 1e30,
            )
        elif mode == "dis_exp_3":
            # Custom: placeholder for user-defined parameters
            hp = HazardParams(
                v_open = 0.50,
                v_build = 0.25,
                block_building = 0,
                Rmax = 10.0,
                tau_pre = 500.0,
                tau_rise = 120.0,
                alpha_pre = 0.3,
                INF = 1e30,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from fire/flood/smoke.")
        # Auto R_block if not set
        if hp.R_block is None:
            hp.R_block = hp.Rmax * 0.99
        return hp
    
    # =========================
    # NEW: preset_para() with 15 fixed cases
    # =========================
    def preset_para(self, case: str, base_mode: str = "custom2") -> HazardParams:
        """
        15 fixed parameter cases for sensitivity study.

        Design: 3 groups × 5 levels = 15
        A01-A05: sweep tau_rise (post-arrival rise), others fixed
        B01-B05: sweep tau_pre  (pre-arrival exposure), others fixed
        C01-C05: sweep alpha_pre (front intensity), others fixed

        You can change the level lists to match your study.
        """
        base = self.preset(base_mode)

        # ---- choose 5 levels each (from your sweep ranges) ----
        speed_levels = [0.01, 0.05, 0.10, 0.50, 1.0]   # pick 5 from (20,80,120,180,240,300)
        tau_rise_levels = [20.0, 80.0, 120.0, 180.0, 300.0]   # pick 5 from (20,80,120,180,240,300)
        tau_pre_levels  = [100.0, 300.0, 500.0, 700.0, 900.0] # your list had 4; add 900 as a high-end (edit if you dislike)
        alpha_levels    = [0.10, 0.30, 0.50, 0.70, 0.90]      # your list had 4; add 0.90 as high-end (edit if you dislike)

        case = case.strip().upper()

        # Group A: tau_rise sweep
        if case in {"A01","A02","A03","A04","A05"}:
            idx = int(case[-2:]) - 1
            hp = replace(base, tau_rise=float(tau_rise_levels[idx]))
            return _auto_rblock(hp)

        # Group B: tau_pre sweep
        if case in {"B01","B02","B03","B04","B05"}:
            idx = int(case[-2:]) - 1
            hp = replace(base, tau_pre=float(tau_pre_levels[idx]))
            return _auto_rblock(hp)

        # Group C: alpha_pre sweep
        if case in {"C01","C02","C03","C04","C05"}:
            idx = int(case[-2:]) - 1
            hp = replace(base, alpha_pre=float(alpha_levels[idx]))
            return _auto_rblock(hp)
        
        # Group s: v_open, v_build sweep
        if case in {"S01","S02","S03","S04","S05"}:
            idx = int(case[-2:]) - 1
            hp = replace(base, v_open = float(speed_levels[idx]))
            hp = replace(base, v_build = float(speed_levels[idx]/2))
            return _auto_rblock(hp)

        raise ValueError("Unknown case. Use A01-A05, B01-B05, C01-C05.")


    def list_preset_para(self, base_mode: str = "custom2") -> List[Tuple[str, HazardParams]]:
        """Return all 15 cases with their HazardParams."""
        cases = ([f"A{i:02d}" for i in range(1,6)] 
        + [f"B{i:02d}" for i in range(1,6)] 
        + [f"C{i:02d}" for i in range(1,6)]
        +[f"S{i:02d}" for i in range(1,6)])

        return [(c, self.preset_para(c, base_mode=base_mode)) for c in cases]

    def to_txt(self, hp: HazardParams, mode: str) -> str:
        lines = []
        lines.append("# -----------------------------")
        lines.append("# Hazard parameters")
        lines.append("# Format: key = value")
        lines.append("# Comments: start with #")
        lines.append("# Units: v_* in m/s, time in seconds")
        lines.append(f"# Preset mode: {mode}")
        lines.append("# -----------------------------\n")

        # Keep a stable key order
        lines.append(f"v_open = {hp.v_open}")
        lines.append(f"v_build = {hp.v_build}")
        lines.append(f"block_building = {hp.block_building}\n")

        lines.append(f"Rmax = {hp.Rmax}")
        lines.append(f"tau_pre = {hp.tau_pre}")
        lines.append(f"tau_rise = {hp.tau_rise}")
        lines.append(f"R_block = {hp.R_block}\n")

        lines.append(f"alpha_pre = {hp.alpha_pre}\n")

        lines.append(f"INF = {hp.INF}")
        lines.append("")
        return "\n".join(lines)
    
    def hp_to_txt(self) -> str:
        # hp = self.hp
        # mode = self.mode
        lines = []
        lines.append("# -----------------------------")
        lines.append("# Hazard parameters")
        lines.append("# Format: key = value")
        lines.append("# Comments: start with #")
        lines.append("# Units: v_* in m/s, time in seconds")
        lines.append("# -----------------------------\n")

        lines.append("v_open = 0.12")
        lines.append("v_build = 0.06")
        lines.append("block_building = 0\n")

        lines.append("Rmax = 10.0")
        lines.append("tau_pre = 500.0")
        lines.append("tau_rise = 120.0")
        lines.append("R_block = 6.0\n")

        lines.append("alpha_pre = 0.3\n")

        lines.append("INF = 1e30")
        lines.append("")

        return "\n".join(lines)


    def write_disasterpara_overwrite(self, mode: str) -> str:
        """
        Always overwrite disasterpara.txt (create parent dirs if needed).
        Returns the written file path (string).
        """
        hp = self.preset(mode)

        base_dir = os.path.dirname(os.path.abspath(__file__))  # src_py/
        hazardparas_path = os.path.abspath(
            os.path.join(base_dir, "../../data/simulation_inputs", self.project_name, "disasterpara.txt")
        )

        os.makedirs(os.path.dirname(hazardparas_path), exist_ok=True)

        content = self.to_txt(hp, mode)
        with open(hazardparas_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

        return hazardparas_path
    
    def write_disasterpara_overwrite_hp(self, mode: str, hp_mode: HazardParams) -> str:
        """
        Always overwrite disasterpara.txt (create parent dirs if needed).
        Returns the written file path (string).
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))                 # src_py/
        hazardparas_path = os.path.abspath(
            os.path.join(base_dir, "../../data/simulation_inputs", self.project_name, "disasterpara.txt")
        )

        os.makedirs(os.path.dirname(hazardparas_path), exist_ok=True)

        content = self.to_txt(hp_mode, hp_mode)
        with open(hazardparas_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return hazardparas_path
    
    def write_disaster_parameters(self) -> str:
        """
        Always overwrite disasterpara.txt (create parent dirs if needed).
        Returns the written file path (string).
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))  # src_py/
        hazardparas_path = os.path.abspath(
            os.path.join(base_dir, "../../data/simulation_inputs", self.project_name, "disasterpara.txt")
        )

        os.makedirs(os.path.dirname(hazardparas_path), exist_ok=True)

        content = self.hp_to_txt()
        with open(hazardparas_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return hazardparas_path

    def run_all(self, label_values, disaster_indices, num, hazartype):
        """执行所有处理步骤"""
        # print(f"=== 开始处理项目: {self.project_name} ===")
        # label_values = [random.randint(2, 3) for _ in range(14)]
        self.process_start_point()
        self.process_buildings()
        self.process_roads()
        self.process_safe_areas()
        self.process_boundaries()
        self.process_add_widths(values=label_values)
        self.write_crowd_config(num)

        # get disaster points, three modes totally
        # self.process_disaster_points(point_indices=[0, 3, 10], n=3)
        # self.process_disaster_points(point_indices = disaster_indices, strict_n = False)
        self.write_disasterpara_overwrite(mode = hazartype)
        self.process_disaster_points(n=1)

        


        # print("Input Process | ")

def Generate_TXT(PROJECT_NAME, disaster_indices = [13], num = 50000, hazard_mode = "custom"):
    """
    Docstring for Generate_TXT
    genarate the input txt files needed for simulation
    :param PROJECT_NAME: name of the project
    :param disaster_indices: list of indices for disaster source points
    :param hazard_mode: type of hazard, default is "fire", can be "fire", "flood", or "smoke"
    """

    # PROJECT_NAME = "shipai_2"
    # label_values = [random.randint(2, 3) for _ in range(14)]

    INPUT_ROOT = "../../data/shp"
    OUTPUT_ROOT = "../../data/simulation_inputs"

    DISCRETE_CHOICES = np.array([0.0]) # [新增] 仅允许这三个离散值
    NUM_GENES = 15         # 基因数量 = 路段数量
    label_values = np.random.choice(DISCRETE_CHOICES, NUM_GENES)
    processor = ShapefileProcessor(
        PROJECT_NAME,
        INPUT_ROOT,
        OUTPUT_ROOT,
    )

    processor.run_all(label_values, disaster_indices, num, hazard_mode)

def read_startpoint_first_line(txt_path: str) -> tuple[float, float]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Empty startpoint file: {txt_path}")
    parts = [p for p in line.replace(",", " ").split() if p]
    if len(parts) < 2:
        raise ValueError(f"First line must contain two numbers, got: {line}")
    return float(parts[0]), float(parts[1])


def export_points_shp_by_project(
    project_name: str,
    step: int | str,
    *,
    crs: str = "EPSG:3395",
    base_csv_dir: str = "../../results/simulation",
    base_input_dir: str = "../../data/simulation_inputs",
    base_out_dir: str = "../../results/analysis/shp",
    x_col: str = "Px",
    y_col: str = "Py",
    state_col: str = "State",
    drop_state_value: int = 0,      # ignore State=0
    keep_all_fields: bool = True,
) -> str:
    """
    Only need (project_name, step). Paths are constructed internally:
      CSV:        ../../results/simulation/{project_name}/csv/{step}.csv
      startpoint: ../../data/simulation_inputs/{project_name}/startpoint.txt
      SHP:        ../../results/analysis/shp/{project_name}_{step}.shp
    """
    step_str = str(step)

    csv_path = os.path.join(base_csv_dir, project_name, "csv", f"{step_str}.csv")
    startpoint_txt = os.path.join(base_input_dir, project_name, "startpoint.txt")
    shp_path = os.path.join(base_out_dir, f"{project_name}_{step_str}.shp")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(startpoint_txt):
        raise FileNotFoundError(f"startpoint.txt not found: {startpoint_txt}")

    os.makedirs(base_out_dir, exist_ok=True)

    x0, y0 = read_startpoint_first_line(startpoint_txt)

    df = pd.read_csv(csv_path)
    for col in (x_col, y_col, state_col):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # ignore State=0
    df = df[df[state_col] != drop_state_value].copy()

    # numeric coords + drop invalid
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    # add startpoint offset (meters)
    df[x_col] = df[x_col] + x0
    df[y_col] = df[y_col] + y0

    if not keep_all_fields:
        keep_cols = [c for c in ["ID", state_col, x_col, y_col] if c in df.columns]
        df = df[keep_cols].copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df[x_col].to_numpy(), df[y_col].to_numpy())],
        crs=crs,
    )

    gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    return shp_path


# ===== Usage =====
# shp_file = export_points_shp_by_project("Shipai", 3000)
# print("Saved:", shp_file)

def write_crowd_config(project_name: str):
    """
    在项目路径下生成 crowd.txt 文件。
    """
    # 注意：根据你的逻辑，输入文件似乎是在 INPUT_PATH 下
    config_path = os.path.join("../../data/simulation_inputs", project_name, "crowd.txt")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("Crowd 50000 1\n")  # 修正为 50000
        f.write("Male 1 30\n")
    # print(f"Add crowd: {config_path}")

def write_project_names_txt(project_names: List[str], txt_path: str) -> str:
    """
    Write project names to a txt file.
    Format:
      line1: number of names (int)
      line2..: one project name per line
    Returns the written file path as string.
    """
    p = Path(txt_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [str(len(project_names))] + project_names
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)

from pathlib import Path

def reset_project_log_csv(log_csv: str = "../../data/simulation_inputs/project_log.csv") -> str:
    """
    Clear the log file content (keep the file). If parent dirs don't exist, create them.
    After reset, the file will contain only:
      0
    """
    p = Path(log_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("0\n", encoding="utf-8")
    return str(p)

def update_project_log_csv(project_list: Iterable[str],
                           csv_path: str = "../../data/simulation_inputs/project_log.csv") -> List[str]:
    """
    维护一个“首行=数量、后续每行=project_name”的CSV日志文件（单列）。
    - 文件不存在：创建
    - 文件存在：读取已有项目，追加新项目（去重），并更新第一行数量
    返回：更新后的完整项目列表（按写入顺序）
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[str] = []

    if csv_path.exists():
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        # 允许第一行不是数字（容错）
        start = 1 if (len(lines) > 0 and lines[0].strip().isdigit()) else 0
        for ln in lines[start:]:
            name = ln.strip().strip(",")  # 容错：即使有人写成 "xxx,"
            if name:
                existing.append(name)

    # 去重追加：保持原有顺序 + 新项目按输入顺序追加
    seen = set(existing)
    updated = list(existing)
    for name in project_list:
        name = str(name).strip()
        if name and name not in seen:
            updated.append(name)
            seen.add(name)

    # 写回：第一行数量，后面每行一个名字（单列CSV）
    out_lines = [str(len(updated))] + updated
    csv_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return updated

def Create_Project_Group(start_idx: int, end_idx: int, num_g = 50000, 
                         hazard_mode_g = "custom", run_mode = "static",
                         log_csv: str="../../data/simulation_inputs/project_log.csv", 
                         out_txt: str = "../../data/simulation_inputs/name.txt") -> List[str]:
    """
    Create a list of project names from start_idx to end_idx (inclusive),
    call Generate_TXT for each project, then write all names to out_txt.
    Example:
      start_idx=1, end_idx=3
      -> ["shipai_1", "shipai_2", "shipai_3"]
      out_txt content:
        3
        shipai_1
        shipai_2
        shipai_3
    """

    project_list: List[str] = []
    for i in range(start_idx, end_idx + 1):
        project_name = f"{run_mode}_{num_g}_{i}"
        Generate_TXT(project_name, [13], num = num_g, hazard_mode = hazard_mode_g)  # 保留你的逻辑
        project_list.append(project_name)
        

    write_project_names_txt(project_list, out_txt)

    # ✅ 追加记录到CSV日志（不断更新）
    update_project_log_csv(project_list, log_csv)

    return project_list


def Create_DisasterParas_Group(start_idx: int, end_idx: int, num_g = 50000,
                         run_mode = "main_density_disaster_30",
                         log_csv: str="../../data/simulation_inputs/project_log.csv",
                         out_txt: str = "../../data/simulation_inputs/name.txt") -> List[str]:
    """
    Create a list of project names from start_idx to end_idx (inclusive),
    call Generate_TXT for each project, then write all names to out_txt.
    Example:
      start_idx=1, end_idx=3
      -> ["shipai_1", "shipai_2", "shipai_3"]
      out_txt content:
        3
        shipai_1
        shipai_2
        shipai_3
    """

    INPUT_ROOT = "../../data/shp"
    OUTPUT_ROOT = "../../data/simulation_inputs"
    # ids = range(0, 61)
    disaster_id = 41
    project_list: List[str] = []


    disaster_lst = ShapefileProcessor("1").list_preset_para()

    for key, value in disaster_lst:
        for i in range(start_idx, end_idx + 1):

            project_name = f"{run_mode}_{key}_{num_g}_{disaster_id}_{i}"
            DISCRETE_CHOICES = np.array([0.0]) # [新增] 仅允许这三个离散值
            NUM_GENES = 15         # 基因数量 = 路段数量
            label_values = np.random.choice(DISCRETE_CHOICES, NUM_GENES)
            processor = ShapefileProcessor(
                project_name,
                INPUT_ROOT,
                OUTPUT_ROOT,
            )
            # processor.run_all(label_values, disaster_id, num_g, value)

                # """执行所有处理步骤"""
            # print(f"=== 开始处理项目: {self.project_name} ===")
            # label_values = [random.randint(2, 3) for _ in range(14)]
            processor.process_start_point()
            processor.process_buildings()
            processor.process_roads()
            processor.process_safe_areas()
            processor.process_boundaries()
            processor.process_add_widths(values=label_values)
            processor.write_crowd_config(num_g)

            # get disaster points, three modes totally
            # self.process_disaster_points(point_indices=[0, 3, 10], n=3)
            processor.process_disaster_points(point_indices = [disaster_id], strict_n = False)
            processor.write_disasterpara_overwrite_hp(mode=key, hp_mode=value)
            # self.process_disaster_points(n=5)
            # print("Input Process | ")


            project_list.append(project_name)

    write_project_names_txt(project_list, out_txt)
    
    # ✅ 追加记录到CSV日志（不断更新）
    update_project_log_csv(project_list, log_csv)
    return project_list



DISTRIBUTION_MODE_MAP = {
    1: "uniform",
    2: "center",
    3: "ring",
}

HAZARD_CLASS_MAP = {
    1: "near_exit",
    2: "near_main_road",
    3: "far_from_exit_and_main_road",
}

from typing import List, Dict, Any


import csv
from pathlib import Path
from typing import List, Dict, Any

def append_random_records_csv(records: List[Dict[str, Any]], csv_path: str) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    # 汇总所有可能字段，保持列完整
    fieldnames = []
    seen = set()
    for r in records:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in records:
            writer.writerow(r)
            
            
def Create_Random_Group(start_idx: int, end_idx: int, num_g = 50000,
                         run_mode = "main_density_disaster_30",
                         log_csv: str="../../data/simulation_inputs/project_log.csv",
                         random_log_csv: str = "../../data/simulation_inputs/project_random_inputs.csv",
                         out_txt: str = "../../data/simulation_inputs/name.txt",
                         global_seed: int = 42,
                         ) -> List[str]:
    """
        批量生成项目，并记录每个 project 对应的随机输入变量，方便后续分析。

        记录内容包括：
        - project_name / sample_index / run_mode
        - label_values
        - hazard_class_id / hazard_class_name / disaster_id / disaster_x / disaster_y
        - distribution_mode_id / distribution_mode_name
        - num_g
        - sample_seed
    """

    INPUT_ROOT = "../../data/shp"
    OUTPUT_ROOT = "../../data/simulation_inputs"
    
    
    
    project_list: List[str] = []
    random_records: List[Dict[str, Any]] = []
    
    # 统一设置一次全局随机种子
    random.seed(global_seed)
    np.random.seed(global_seed)

    for i in range(start_idx, end_idx + 1):
        project_name = f"{run_mode}_{i}"
        
        # 给每个样本派生一个独立 seed，避免所有样本完全重复
        sample_seed = global_seed + i

        random.seed(sample_seed)
        np.random.seed(sample_seed)
        
        DISCRETE_CHOICES = np.array([0.0]) # [新增] 仅允许这三个离散值
        NUM_GENES = 15         # 基因数量 = 路段数量
        label_values = np.random.choice(DISCRETE_CHOICES, NUM_GENES)

        processor = ShapefileProcessor(
            project_name,
            INPUT_ROOT,
            OUTPUT_ROOT,)
        
        # -------------------------
        # 基础几何处理
        # -------------------------
        safe_keep_prob = 0.8
        
        processor.process_start_point()
        
        processor.process_buildings()
        
        processor.process_roads()
        
        exit_switches, exit_switch_dict = processor.process_safe_areas(keep_prob=0.8)
        
        processor.process_boundaries()
        
        processor.process_add_widths(values = label_values)
        
        # num
        num_g = random.randint(40000, 50000)
        processor.write_crowd_config(num_g)

        # -------------------------
        # 灾害位置：先选类别，再选对应 shp 中的具体点
        # 注意：这里不要固定 seed=42，否则每次可能重复
        # -------------------------
        hazard_class_id, hazard_class_name, disaster_id, coords = processor.process_disaster_points_by_mode(
            shp_map={
                1: "disastersource_near_exit.shp",
                2: "disastersource_near_mainroad.shp",
                3: "disastersource_far.shp",
            },
            seed = sample_seed + 1000,
        )
        
        disaster_x = None
        disaster_y = None
        if coords and len(coords) > 0:
            disaster_x, disaster_y = coords[0]
        
        processor.process_boundary_random_point()
        processor.write_disaster_parameters()
        
        # -------------------------
        # 初始人群分布模式
        # -------------------------
        random.seed(sample_seed + 2000)
        dist_mode_id = random.randint(1, 3)
        dist_mode_name = DISTRIBUTION_MODE_MAP[dist_mode_id]
        
        processor.generate_population_distribution_from_boundary_2(
            mode_id=dist_mode_id,
            seed=sample_seed + 3000,
        )
        
        project_list.append(project_name)
        
        # -------------------------
        # 记录这一条样本的所有随机输入
        # -------------------------
        rec: Dict[str, Any] = {
            "project_name": project_name,
            "sample_index": i,
            "run_mode": run_mode,
            "global_seed": global_seed,
            "sample_seed": sample_seed,

            # population / crowd
            "num_g": num_g,
            "distribution_mode_id": dist_mode_id,
            "distribution_mode_name": dist_mode_name,

            # disaster
            "hazard_class_id": hazard_class_id,
            "hazard_class_name": hazard_class_name,
            "disaster_id": disaster_id,
            "disaster_x": disaster_x,
            "disaster_y": disaster_y,

            # safe area config
            # safe area config
            "safe_keep_prob": safe_keep_prob,
            "n_open_exits": processor.n_open_exits,
            "n_total_exits": processor.n_total_exits,
            "exit_switches_json": json.dumps(exit_switches),
            "exit_switch_dict_json": json.dumps(exit_switch_dict),

            # genes
            "num_genes": NUM_GENES,
            "label_values_json": json.dumps(label_values.tolist()),
        }

        # 也可以把每个 gene 单独展开，方便后面直接做表格分析
        for k, v in enumerate(label_values):
            rec[f"label_{k}"] = float(v)

        random_records.append(rec)
        


    write_project_names_txt(project_list, out_txt)
    # ✅ 追加记录到CSV日志（不断更新）
    update_project_log_csv(project_list, log_csv)
    
    # 新增：随机输入日志
    append_random_records_csv(random_records, random_log_csv)


    return project_list

def Create_Disastersource_Group(start_idx: int, end_idx: int, num_g = 50000,
                                ids = range(1, 62),
                                experiment_name = "",
                         hazard_mode_g = "custom", run_mode = "main_density_disaster_30",
                         log_csv: str="../../data/simulation_inputs/project_log.csv",
                         out_txt: str = "../../data/simulation_inputs/name.txt") -> List[str]:
    """
    Create a list of project names from start_idx to end_idx (inclusive),
    call Generate_TXT for each project, then write all names to out_txt.
    Example:
      start_idx=1, end_idx=3
      -> ["shipai_1", "shipai_2", "shipai_3"]
      out_txt content:
        3
        shipai_1
        shipai_2
        shipai_3
    """
    # ids = range(0, 61)

    project_list: List[str] = []
    for disaster_id in ids:
        for i in range(start_idx, end_idx + 1):
            project_name = f"{experiment_name}_{run_mode}_{num_g}_{hazard_mode_g}_{disaster_id}_{i}"
            Generate_TXT(project_name, [disaster_id], num = num_g, hazard_mode = hazard_mode_g)  # 保留你的逻辑
            project_list.append(project_name)
    write_project_names_txt(project_list, out_txt)
    
    # ✅ 追加记录到CSV日志（不断更新）
    update_project_log_csv(project_list, log_csv)
    return project_list


if __name__ == "__main__":
    # 示例：处理 shipai_2250 项目

    # project_name = "All_disaster_points"
    # # [generate TXT file]
    # disa_list = range(0, 61)
    # Generate_TXT(project_name, disaster_indices= disa_list)

    # Create_Project_Group(1, 50, mode = "custom", out_txt = "../../data/simulation_inputs/name.txt")

    # [trandfer bin file to tif file
    convert_bin_to_tif_for_project("static_10")

    # lst = ShapefileProcessor("1").list_preset_para()
    # print(lst)  # 应该输出 15

    # [plot and analyze the result: time and distance distribution]
    # plot_and_analyze(project_name, state_done=0,
    #                  dist_thresh=300, time_thresh=800,
    #                  bins_time=50, bins_dist=50)
    
    # [paradigm]the density, velocity, flow relationship, for project and time
    Draw_fundenmental_paradigm("density_10", 20)

    # [export shp file for visualization]
    # export_points_shp_by_project(project_name, 2500)
    