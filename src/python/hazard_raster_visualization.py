from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import rasterio
except ImportError as e:
    raise ImportError("Please install rasterio: pip install rasterio") from e

# ----------------------------
# Helpers
# ----------------------------
def _read_tif(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    """Read first band. Return (array, nodata)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
    return arr, nodata

def _finite_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    m = np.isfinite(arr)
    if nodata is not None:
        m &= (arr != float(nodata))
    return m

def _robust_range(arr_list: List[np.ndarray], nodata_list: List[Optional[float]],
                  q_lo: float = 2.0, q_hi: float = 98.0) -> Tuple[float, float]:
    """Robust vmin/vmax based on percentiles of finite pixels across a list."""
    vals = []
    for arr, nd in zip(arr_list, nodata_list):
        m = _finite_mask(arr, nd)
        if np.any(m):
            vals.append(arr[m])
    if not vals:
        return 0.0, 1.0
    x = np.concatenate(vals)
    vmin = float(np.percentile(x, q_lo))
    vmax = float(np.percentile(x, q_hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.min(x)), float(np.max(x))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
    return vmin, vmax

def _set_scientific_style():
    plt.rcParams.update({
        "figure.figsize": (14.5, 3.2),   # wide for 1x5
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })

def _case_id_from_project(project_name: str) -> str:
    """
    Extract A01/B03/C05/S02 from strings like:
    main_disaster_para_A01_5000_41_1
    """
    m = re.search(r"_(A|B|C|S)\d{2}_", project_name)
    if not m:
        m2 = re.search(r"(A|B|C|S)\d{2}", project_name)
        return m2.group(0) if m2 else project_name
    return m.group(0).strip("_")

def _group_from_case(case_id: str) -> str:
    if case_id and case_id[0] in ("A", "B", "C", "S"):
        return case_id[0]
    return "?"

# ----------------------------
# Core plotting: one layer × one group (5 cases) -> 1 figure
# ----------------------------
def _plot_group_layer(
    group: str,
    layer_key: str,
    cases: List[str],
    arrays: Dict[str, np.ndarray],
    nodata: Dict[str, Optional[float]],
    out_png: Path,
    out_pdf: Path,
    *,
    vmin: float,
    vmax: float,
    suptitle: str,
    cbar_position: str = "right",   # "right" or "bottom"
    cbar_size: str = "2.5%",        # thickness
    cbar_pad: float = 0.06,         # gap
    wspace: float = 0.02,           # tighter subplots
    outer_left: float = 0.02,
    outer_right: float = 0.96,
    outer_bottom: float = 0.02,
    outer_top: float = 0.88,
):
    """
    Tight 1x5 comparison plot with a non-overlapping colorbar.
    Requires: from mpl_toolkits.axes_grid1 import make_axes_locatable
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(16, 5.5))

    im = None
    for ax, case in zip(axes, cases):
        arr = arrays.get(case)
        nd = nodata.get(case)
        if arr is None:
            ax.set_title(f"{case}\n(MISSING)")
            ax.axis("off")
            continue

        m = _finite_mask(arr, nd)
        show = np.where(m, arr, np.nan)

        # "cividis"（色盲友好） "magma" / "inferno"（对比强，亮区更突出） "plasma" 
        # "turbo"（更鲜艳，但有时显得“炫”） "gray"（灰度）
        im = ax.imshow(show, vmin=vmin, vmax=vmax, cmap="turbo", interpolation="nearest")

        ax.set_title(case, fontsize=11, pad = 6)
        ax.set_xticks([])
        ax.set_yticks([])

        # thinner frame for a cleaner look
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # # compact spacing
    # fig.subplots_adjust(
    #     left=outer_left, right=outer_right,
    #     bottom=outer_bottom, top=outer_top,
    #     wspace=wspace, hspace=0.0
    # )

    # --- tighter title (less height) ---
    # 让子图区域更“满”，减少顶部留白
    fig.subplots_adjust(
        left=0.02, right=0.98,
        bottom=0.12,   # 给底部色标留空间
        top=0.86,      # 子图区域更高；标题在 0.92，不会挤
        wspace=0.05
    )

    # global title (less vertical space)
    fig.suptitle(suptitle, y=0.92, fontsize=14)

    # ---- colorbar (no overlap) ----
    if im is not None:
        if cbar_position.lower() == "right":
            # attach colorbar to the last axes but OUTSIDE it (no overlap)
            divider = make_axes_locatable(axes[-1])
            cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
            cbar = fig.colorbar(im, cax=cax)
        elif cbar_position.lower() == "bottom":
            # put a horizontal colorbar under ALL subplots
            # use a new axes spanning the figure width
            cax = fig.add_axes([0.02, 0.06, 0.96, 0.05])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        else:
            raise ValueError("cbar_position must be 'right' or 'bottom'.")

        cbar.ax.tick_params(labelsize=10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def _plot_group_3layers_stack(
    group: str,
    cases: List[str],
    arrays_by_layer: Dict[str, Dict[str, Optional[np.ndarray]]],
    nodata_by_layer: Dict[str, Dict[str, Optional[float]]],
    ranges_by_layer: Dict[str, Tuple[float, float]],
    out_png: Path,
    out_pdf: Path,
    *,
    layer_order: List[str],
    layer_titles: Optional[Dict[str, str]] = None,
    suptitle: Optional[str] = None,
    cmap_by_layer: Optional[Dict[str, str]] = None,
    # ---- NEW: compact layout knobs (match single-figure style) ----
    base_figsize: Tuple[float, float] = (16, 5.5),  # your single: (16, 5.5)
    height_scale: float = 3.0,                      # stack 3 rows => ×3
    left: float = 0.02,
    right: float = 0.985,
    bottom: float = 0.04,
    top: float = 0.90,
    wspace: float = 0.03,
    hspace: float = 0.06,
    cbar_col_width: float = 0.06,   # thinner colorbar column
):
    """
    Make a summary figure for one group:
      rows = len(layer_order) (e.g., 3)
      cols = 5 cases
      Each row has its own colorbar (separate scale).

    Layout uses GridSpec with an extra colorbar column.
    """
    import matplotlib.gridspec as gridspec

    nrows = len(layer_order)
    ncols = len(cases)

    # ---- compact figsize: single height ×3 ----
    fig_w, fig_h = base_figsize[0], base_figsize[1] * height_scale
    fig = plt.figure(figsize=(fig_w, fig_h))

    # ---- compact GridSpec: 5 cols + 1 thin cbar col ----
    width_ratios = [1.0] * ncols + [cbar_col_width]
    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=width_ratios,
        wspace=wspace,
        hspace=hspace,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )

    if layer_titles is None:
        layer_titles = {k: k for k in layer_order}

    if cmap_by_layer is None:
        # You can customize per layer if you want
        cmap_by_layer = {k: "turbo" for k in layer_order}

    ims = {}

    for r, layer_key in enumerate(layer_order):
        vmin, vmax = ranges_by_layer[layer_key]

        # Row label (layer title) placed at first subplot's left via ylabel-like text
        for c, case in enumerate(cases):
            ax = fig.add_subplot(gs[r, c])

            arr = arrays_by_layer[layer_key].get(case)
            nd = nodata_by_layer[layer_key].get(case)

            if arr is None:
                ax.set_title(case if r == 0 else "", fontsize=18, pad=6)
                if c == 0:
                    ax.text(
                        -0.04, 0.5, layer_titles.get(layer_key, layer_key),
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=15
                    )
                ax.axis("off")
                continue

            m = _finite_mask(arr, nd)
            show = np.where(m, arr, np.nan)

            im = ax.imshow(
                show,
                vmin=vmin, vmax=vmax,
                cmap=cmap_by_layer.get(layer_key, "turbo"),
                interpolation="nearest"
            )
            ims[layer_key] = im

            # Only top row shows case titles (cleaner)
            if r == 0:
                ax.set_title(case, fontsize=21, pad=8)

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

            # Put row label on first column
            if c == 0:
                ax.set_ylabel(layer_titles.get(layer_key, layer_key),
                              fontsize=18, rotation=90, labelpad=10)

        # ---- per-row colorbar ----
        cax = fig.add_subplot(gs[r, ncols])
        if layer_key in ims:
            cbar = fig.colorbar(ims[layer_key], cax=cax)
            cbar.ax.tick_params(labelsize=14)
        else:
            cax.axis("off")

    # ---- title: reduce vertical cost ----
    if suptitle is None:
        suptitle = f"Group {group}: Arrival / Intensity / HardMask (stacked)"
    fig.suptitle(suptitle, fontsize=21, y=0.94)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Public function: 15 projects -> 9 figures
# ----------------------------
def visualize_disaster_tif_groups(
    project_names: List[str],
    *,
    base_output_dir: str | Path = "../../results/simulation",
    out_dir: str | Path = "../../results/analysis/disasterparas",
    tif_subdir: str = "tif",
    save_pdf: bool = True,
) -> List[Path]:
    """
    Make 9 comparison figures:
      Group A: source/arrival/intensity (each is 1x5)
      Group B: source/arrival/intensity
      Group C: source/arrival/intensity

    Returns list of saved file paths.
    """
    _set_scientific_style()

    base_output_dir = Path(base_output_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tif filenames (exact as you showed)
    layer_files = {
        # "source": "13_disaster_source.tif",
        "arrival": "14_disaster_arrival_time.tif",
        "intensity": "15_disaster_intensity.tif",
        "hardmask": "16_disaster_hard_mask.tif",
    }

    # Build mapping: group -> case -> project_name
    group_map: Dict[str, Dict[str, str]] = {"A": {}, "B": {}, "C": {}, "S": {}}
    for pn in project_names:
        case_id = _case_id_from_project(pn)  # e.g., A01
        g = _group_from_case(case_id)
        if g in group_map:
            group_map[g][case_id] = pn

    saved: List[Path] = []

    # For each group and each layer, read 5 rasters, compute common range, then plot
    for g in ["A", "B", "C", "S"]:
        # Sort cases A01..A05 etc
        cases = sorted(group_map[g].keys())
        # Keep only first 5 if extra; ideally exactly 5
        cases = cases[:5]
        if len(cases) == 0:
            print(f"[WARN] Group {g} has no cases. Check project_names or regex.")
            continue

        # ---- NEW: cache for summary figure ----
        arrays_by_layer: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        nodata_by_layer: Dict[str, Dict[str, Optional[float]]] = {}
        ranges_by_layer: Dict[str, Tuple[float, float]] = {}

        for layer_key, tif_name in layer_files.items():
            arrays: Dict[str, np.ndarray] = {}
            nodata: Dict[str, Optional[float]] = {}

            arr_list, nd_list = [], []

            for case in cases:
                pn = group_map[g].get(case)
                tif_path = base_output_dir / pn / tif_subdir / tif_name
                if not tif_path.exists():
                    arrays[case] = None  # mark missing
                    nodata[case] = None
                    continue
                arr, nd = _read_tif(tif_path)
                arrays[case] = arr
                nodata[case] = nd
                arr_list.append(arr)
                nd_list.append(nd)

            # Decide consistent vmin/vmax for comparability
            if layer_key == "source":
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = _robust_range(arr_list, nd_list, q_lo=2.0, q_hi=98.0)

            # ---- NEW: store for summary ----
            arrays_by_layer[layer_key] = arrays
            nodata_by_layer[layer_key] = nodata
            ranges_by_layer[layer_key] = (vmin, vmax)

            # Filenames
            stem = f"disaster_{layer_key}_group{g}_1x5"
            png_path = out_dir / f"{stem}.png"
            pdf_path = out_dir / f"{stem}.pdf"

            suptitle = f"Group {g}_{layer_key} (same color scale within group)"
            _plot_group_layer(
                g, layer_key, cases, arrays, nodata,
                png_path, (pdf_path if save_pdf else png_path),
                vmin=vmin, vmax=vmax,
                suptitle=suptitle,
                cbar_position="bottom",
                wspace=0.05
            )
            saved.append(png_path)
            if save_pdf:
                saved.append(pdf_path)

            # ---- NEW: stacked summary figure for this group ----
            # pick only the 3 layers you want
            stack_layers = ["arrival", "intensity", "hardmask"]

        # guard: ensure they exist
        if all(k in arrays_by_layer for k in stack_layers):
            stem_sum = f"disaster_group{g}_arrival_intensity_hardmask_stack3x5"
            png_sum = out_dir / f"{stem_sum}.png"
            pdf_sum = out_dir / f"{stem_sum}.pdf"

            _plot_group_3layers_stack(
                group=g,
                cases=cases,
                arrays_by_layer=arrays_by_layer,
                nodata_by_layer=nodata_by_layer,
                ranges_by_layer=ranges_by_layer,
                out_png=png_sum,
                out_pdf=(pdf_sum if save_pdf else png_sum),
                layer_order=stack_layers,
                layer_titles={
                    "arrival": "Arrival time",
                    "intensity": "Intensity",
                    "hardmask": "Hard mask"
                },
                suptitle=f"Group {g}: Arrival / Intensity / HardMask",
                cmap_by_layer={
                    "arrival": "turbo",
                    "intensity": "turbo",
                    "hardmask": "gray"
                },
                # ---- compact knobs ----
                base_figsize=(22, 6),
                height_scale=3.0,
                left=0.02, right=0.985,
                bottom=0.035, top=0.895,
                wspace=0.05, hspace=0.05,
                cbar_col_width=0.06,
            )
            saved.append(png_sum)
            if save_pdf:
                saved.append(pdf_sum)
        else:
            print(f"[WARN] Group {g}: cannot make stacked summary (missing layers).")
    return saved


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    project_names = [
        "main_disaster_para_A01_5000_41_1",
        "main_disaster_para_A02_5000_41_1",
        "main_disaster_para_A03_5000_41_1",
        "main_disaster_para_A04_5000_41_1",
        "main_disaster_para_A05_5000_41_1",
        "main_disaster_para_B01_5000_41_1",
        "main_disaster_para_B02_5000_41_1",
        "main_disaster_para_B03_5000_41_1",
        "main_disaster_para_B04_5000_41_1",
        "main_disaster_para_B05_5000_41_1",
        "main_disaster_para_C01_5000_41_1",
        "main_disaster_para_C02_5000_41_1",
        "main_disaster_para_C03_5000_41_1",
        "main_disaster_para_C04_5000_41_1",
        "main_disaster_para_C05_5000_41_1",
        "main_disaster_para_S01_5000_41_1",
        "main_disaster_para_S02_5000_41_1",
        "main_disaster_para_S03_5000_41_1",
        "main_disaster_para_S04_5000_41_1",
        "main_disaster_para_S05_5000_41_1",
    ]

    files = visualize_disaster_tif_groups(
        project_names,
        base_output_dir="../../results/simulation",
        out_dir="../../results/analysis/disasterparas",
        save_pdf=True,
    )
    print("Saved:")
    for f in files:
        print(" ", f)
