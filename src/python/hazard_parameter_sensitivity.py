import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ----------------------------
# Model (matches your CUDA)
# ----------------------------
@dataclass(frozen=True)
class HazardParams:
    Rmax: float = 10.0
    tau_pre: float = 500.0
    tau_rise: float = 20.0
    alpha_pre: float = 0.1  # R0 = alpha_pre * Rmax
    speed_open: float = 0.10


def hazard_intensity(t: np.ndarray, Th: float, p: HazardParams) -> np.ndarray:
    """
    Piecewise intensity model:
      s = t - Th
      R0 = alpha_pre * Rmax
      if s < 0:  R = R0 * exp(s / tau_pre)              (pre-arrival, s<0)
      else:      R = R0 + (Rmax-R0)*(1-exp(-s/tau_rise)) (post-arrival, s>=0)
    """
    t = np.asarray(t, dtype=float)
    s = t - float(Th)

    R0 = float(p.alpha_pre) * float(p.Rmax)
    tau_pre = max(float(p.tau_pre), 1e-12)
    tau_rise = max(float(p.tau_rise), 1e-12)

    R = np.empty_like(t)
    pre = s < 0.0
    post = ~pre

    R[pre] = R0 * np.exp(s[pre] / tau_pre)
    R[post] = R0 + (float(p.Rmax) - R0) * (1.0 - np.exp(-s[post] / tau_rise))

    return np.clip(R, 0.0, float(p.Rmax))


# ----------------------------
# Scientific plotting style
# ----------------------------
def set_scientific_style() -> None:
    """A clean, journal-like Matplotlib style without forcing specific colors."""
    plt.rcParams.update({
        "figure.figsize": (7.2, 4.6),
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.borderpad": 0.3,
        "legend.handlelength": 2.2,
        "axes.grid": False,  # keep clean; add light grid manually if needed
    })


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # light grid (journal-like; not too strong)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.12)


# ----------------------------
# Core plotting helper
# ----------------------------
def plot_intensity_comparison(
    params_list: Sequence[HazardParams],
    *,
    Th: float = 0.0,
    t_window: Tuple[float, float] = (-800.0, 400.0),  # relative time window (t-Th)
    n: int = 2400,
    title: Optional[str] = None,
    xlabel: str = r"Relative time $\,(t - T_h)$",
    ylabel: str = r"Intensity $R$",
    show_arrival: bool = True,
    show_R0_markers: bool = False,   # True if you want per-curve R0 guides (may clutter)
    legend_loc: str = "best",
    outpath_png: Optional[Path] = None,
    outpath_pdf: Optional[Path] = None,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw multiple curves on a single axis with scientific styling and save at high DPI.
    """
    t_rel = np.linspace(float(t_window[0]), float(t_window[1]), int(n))
    t_abs = t_rel + float(Th)

    fig, ax = plt.subplots()

    for p in params_list:
        R = hazard_intensity(t_abs, Th=Th, p=p)
        label = rf"$R_{{\max}}$={p.Rmax:g}, $\alpha_{{pre}}$={p.alpha_pre:g}, $\tau_{{pre}}$={p.tau_pre:g}, $\tau_{{rise}}$={p.tau_rise:g}"
        ax.plot(t_rel, R, label=label)

        if show_R0_markers:
            R0 = p.alpha_pre * p.Rmax
            ax.axhline(R0, linestyle="--", linewidth=1.0, alpha=0.7)

    if show_arrival:
        ax.axvline(0.0, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(0.0, 0.98, r"$t=T_h$",
                transform=ax.get_xaxis_transform(), ha="left", va="top")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # if title:
    #     ax.set_title(title)

    _style_axes(ax)
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    if outpath_png:
        outpath_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath_png, dpi=dpi, bbox_inches="tight")
    if outpath_pdf:
        outpath_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath_pdf, bbox_inches="tight")  # vector-friendly

    return fig, ax


def plot_intensity_speed_sweep(
    params_list: Sequence[HazardParams],
    *,
    distance: float = 100.0,                 # ✅ 传播距离(米)——你可以改
    t_window_abs: Tuple[float, float] = (-2000.0, 2000.0),  # ✅ 绝对时间窗口
    n: int = 2400,
    title: Optional[str] = None,
    xlabel: str = r"Absolute time $t$",
    ylabel: str = r"Intensity $R$",
    legend_loc: str = "best",
    outpath_png: Optional[Path] = None,
    outpath_pdf: Optional[Path] = None,
    dpi: int = 600,
):
    t_abs = np.linspace(float(t_window_abs[0]), float(t_window_abs[1]), int(n))

    fig, ax = plt.subplots()

    for p in params_list:
        # ✅ 每条曲线的到达时间由速度决定
        Th_i = float(distance) / max(float(p.speed_open), 1e-12)
        R = hazard_intensity(t_abs, Th=Th_i, p=p)

        label = (rf"$v_{{open}}$={p.speed_open:g}, "
                 rf"$T_h$={Th_i:.1f}, "
                 rf"$R_{{\max}}$={p.Rmax:g}, "
                 rf"$\alpha_{{pre}}$={p.alpha_pre:g}, "
                 rf"$\tau_{{pre}}$={p.tau_pre:g}, "
                 rf"$\tau_{{rise}}$={p.tau_rise:g}")
        ax.plot(t_abs, R, label=label)

        # ✅ 标出每条曲线自己的到达时刻
        ax.axvline(Th_i, linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    _style_axes(ax)
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    if outpath_png:
        outpath_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath_png, dpi=dpi, bbox_inches="tight")
    if outpath_pdf:
        outpath_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath_pdf, bbox_inches="tight")

    return fig, ax



# ----------------------------
# Produce a standard "paper-ready" set of comparisons
# ----------------------------
def make_disaster_param_plots(
    *,
    out_dir: Union[str, Path] = "../../results/analysis/disasterparas",
    Th: float = 0.0,
    t_window: Tuple[float, float] = (-800.0, 400.0),
    dpi: int = 600,
    # Baseline params
    base: HazardParams = HazardParams(
        Rmax=10.0, tau_pre=500.0, tau_rise=120.0, alpha_pre=0.3,
        speed_open=0.5,   # ✅ NEW: baseline speed (edit if you want)
    ),
    # Sweeps (edit these freely)
    sweep_tau_rise: Sequence[float] = (10.0, 20.0, 60.0),
    sweep_tau_pre: Sequence[float] = (100.0, 300.0, 700.0),
    sweep_alpha_pre: Sequence[float] = (0.05, 0.10, 0.30),
    sweep_speed_open: Sequence[float] = (0.01, 0.05, 0.10),
    save_pdf: bool = True,
) -> List[Path]:
    """
    Generates high-quality comparison figures and saves them.
    Returns a list of saved file paths.
    """
    set_scientific_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []

    # 1) Post-arrival rise (tau_rise)
    params1 = [HazardParams(Rmax=base.Rmax, tau_pre=base.tau_pre, tau_rise=tr, alpha_pre=base.alpha_pre,
                            speed_open=base.speed_open)
               for tr in sweep_tau_rise]
    f1_png = out_dir / "intensity_sweep_tau_rise.png"
    f1_pdf = out_dir / "intensity_sweep_tau_rise.pdf"
    plot_intensity_comparison(
        params1, Th=Th, t_window=t_window, dpi=dpi,
        title=r"Post-arrival rise controlled by $\tau_{rise}$",
        outpath_png=f1_png,
        outpath_pdf=(f1_pdf if save_pdf else None),
        legend_loc="best",
    )
    plt.close()
    saved.append(f1_png)
    if save_pdf:
        saved.append(f1_pdf)

    # 2) Pre-arrival exposure extent (tau_pre)
    params2 = [HazardParams(Rmax=base.Rmax, tau_pre=tp, tau_rise=base.tau_rise, alpha_pre=base.alpha_pre,
                            speed_open=base.speed_open)
               for tp in sweep_tau_pre]
    f2_png = out_dir / "intensity_sweep_tau_pre.png"
    f2_pdf = out_dir / "intensity_sweep_tau_pre.pdf"
    plot_intensity_comparison(
        params2, Th=Th, t_window=t_window, dpi=dpi,
        title=r"Pre-arrival exposure controlled by $\tau_{pre}$",
        outpath_png=f2_png,
        outpath_pdf=(f2_pdf if save_pdf else None),
        legend_loc="best",
    )
    plt.close()
    saved.append(f2_png)
    if save_pdf:
        saved.append(f2_pdf)

    # 3) Front intensity at arrival (alpha_pre)
    params3 = [HazardParams(Rmax=base.Rmax, tau_pre=base.tau_pre, tau_rise=base.tau_rise, alpha_pre=ap,
                            speed_open=base.speed_open)
               for ap in sweep_alpha_pre]
    f3_png = out_dir / "intensity_sweep_alpha_pre.png"
    f3_pdf = out_dir / "intensity_sweep_alpha_pre.pdf"
    plot_intensity_comparison(
        params3, Th=Th, t_window=t_window, dpi=dpi,
        title=r"Arrival-front intensity controlled by $\alpha_{pre}$ ($R_0=\alpha_{pre}R_{max}$)",
        outpath_png=f3_png,
        outpath_pdf=(f3_pdf if save_pdf else None),
        legend_loc="best",
    )
    plt.close()
    saved.append(f3_png)
    if save_pdf:
        saved.append(f3_pdf)

    # ✅ 4) Propagation speed (speed_open)
    params4 = [HazardParams(Rmax=base.Rmax, tau_pre=base.tau_pre, tau_rise=base.tau_rise, alpha_pre=base.alpha_pre,
                            speed_open=sp)
               for sp in sweep_speed_open]
    f4_png = out_dir / "intensity_sweep_speed_open.png"
    f4_pdf = out_dir / "intensity_sweep_speed_open.pdf"


    plot_intensity_speed_sweep(
        params4,
        distance=100.0,                    # 你自己定
        t_window_abs=(-2000, 8000),        # 你自己定
        dpi=dpi,
        title=r"Propagation speed controlled by $v_{open}$ (arrival time shift)",
        outpath_png=f4_png,
        outpath_pdf=(f4_pdf if save_pdf else None),
        legend_loc="best",
    )

    plt.close()
    saved.append(f4_png)

    if save_pdf:
        saved.append(f4_pdf)

    return saved


# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    # You can edit sweeps here
    files = make_disaster_param_plots(
        out_dir = "../../results/analysis/disasterparas",
        Th = 0.0,
        t_window = (-1600, 800),
        dpi = 600,
        base=HazardParams(Rmax=10.0, tau_pre=500.0, tau_rise=120.0, alpha_pre=0.3, speed_open= 0.10),
        sweep_tau_rise = (20, 80, 120, 180, 240, 300),
        sweep_tau_pre = (100, 300, 500, 700, 900),
        sweep_alpha_pre = (0.1, 0.30, 0.50, 0.70, 0.90),
        sweep_speed_open = (0.02, 0.05, 0.10, 0.50, 1.00),

        save_pdf = True,
    )
    print("Saved:")
    for f in files:
        print("  ", f)
