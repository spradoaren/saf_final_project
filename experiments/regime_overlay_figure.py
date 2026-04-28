"""Regime overlay figure for paper Section 5.3 (R3).

Single PDF with three vertically stacked panels (top → bottom: HMM,
MS-AR, IOHMM). Each panel shows the realized SPY volatility series
sqrt(rv_gk) * 100 over 2019-01-02 → 2024-12-31 as a black line, with
the smoothed high-regime posterior gamma_t(K-1) (after R1's
volatility-ordered permutation) overlaid as red background shading
whose alpha is proportional to gamma_t(K-1). Cells with
gamma_t(K-1) < 0.1 receive no shading, and the alpha ceiling is 0.6
so the black line remains visible on the most stressed days.

The HMM panel additionally annotates two stress periods with faint
grey vertical spans: the COVID episode (2020-02-24 to 2020-04-30) and
the 2022 inflation repricing window (2022-08-01 to 2022-10-31).

Pure consumer of R1's posteriors and the canonical rv_gk cache.
matplotlib only — no seaborn or other plotting deps. Plotting is fully
deterministic; no random seed is required.

Output: results/regime_overlay.pdf  (vector PDF, no rasterization).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from experiments._shared import get_canonical_rv_gk


CACHE_DIR = Path(__file__).resolve().parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Top-to-bottom panel order: HMM (K=3), then MS-AR (K=2), then IOHMM (K=2).
# IOHMM at the bottom puts its broader macro-driven sensitivity as the
# visual takeaway of the figure.
PANELS: Tuple[Tuple[str, str, int], ...] = (
    ("hmm",   "HMM (K=3)",   3),
    ("msar",  "MS-AR (K=2)", 2),
    ("iohmm", "IOHMM (K=2)", 2),
)

ALPHA_FLOOR = 0.1                  # gamma_high < 0.1 -> no shading
ALPHA_CEILING = 0.6                # gamma_high == 1 -> alpha = 0.6 (line stays visible)
SHADING_RGB = (0.85, 0.20, 0.20)   # red/salmon
Y_MAX = 100.0                      # vol_pct y-limit (chosen so the March 2020 ≈96 peak fits)

# Stress-period vertical spans on the top (HMM) panel.
STRESS_SPANS = (
    ("COVID-19",       pd.Timestamp("2020-02-24"), pd.Timestamp("2020-04-30")),
    ("2022 repricing", pd.Timestamp("2022-08-01"), pd.Timestamp("2022-10-31")),
)
COVID_START = pd.Timestamp("2020-02-24")
COVID_END = pd.Timestamp("2020-04-30")


def _make_shading_cmap() -> ListedColormap:
    """Constant-hue colormap: alpha sweeps 0 → ALPHA_CEILING across the value range."""
    n = 256
    colors = np.empty((n, 4), dtype=np.float64)
    colors[:, 0] = SHADING_RGB[0]
    colors[:, 1] = SHADING_RGB[1]
    colors[:, 2] = SHADING_RGB[2]
    colors[:, 3] = np.linspace(0.0, ALPHA_CEILING, n, dtype=np.float64)
    return ListedColormap(colors)


def _load_high_regime(name: str, K: int, canonical_index: pd.DatetimeIndex) -> np.ndarray:
    df = pd.read_parquet(CACHE_DIR / f"regime_diagnostics_{name}.parquet")
    df.index = pd.to_datetime(df.index)
    df = df.astype(np.float64).reindex(canonical_index)
    return df.iloc[:, K - 1].to_numpy(dtype=np.float64)


def _date_cell_edges(dates_num: np.ndarray) -> np.ndarray:
    """Cell edges at midpoints between adjacent dates, half-step extrapolation
    at the ends. Length = len(dates_num) + 1."""
    if len(dates_num) < 2:
        return np.array([dates_num[0] - 0.5, dates_num[0] + 0.5], dtype=np.float64)
    diffs = np.diff(dates_num)
    edges = np.empty(len(dates_num) + 1, dtype=np.float64)
    edges[1:-1] = dates_num[:-1] + diffs / 2.0
    edges[0] = dates_num[0] - diffs[0] / 2.0
    edges[-1] = dates_num[-1] + diffs[-1] / 2.0
    return edges


def _render_panel(
    ax,
    dates_num: np.ndarray,
    vol_pct: np.ndarray,
    gamma_high: np.ndarray,
    title: str,
    show_xlabel: bool,
    annotate_stress: bool,
    cmap: ListedColormap,
) -> Tuple[float, float, float]:
    """Render a single panel.

    Returns (max_vol_pct, frac_full_high, frac_covid_high) where the
    fractions are share of dates with gamma_high > 0.5 over the full
    sample and over the COVID window respectively.
    """
    g_show = np.where(
        np.isnan(gamma_high) | (gamma_high < ALPHA_FLOOR),
        0.0,
        gamma_high,
    ).astype(np.float64)

    x_edges = _date_cell_edges(dates_num)
    y_edges = np.array([0.0, Y_MAX], dtype=np.float64)

    ax.pcolormesh(
        x_edges,
        y_edges,
        g_show[None, :],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=False,
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        zorder=0,
    )

    if annotate_stress:
        for label, t_start, t_end in STRESS_SPANS:
            x0 = mdates.date2num(t_start.to_pydatetime())
            x1 = mdates.date2num(t_end.to_pydatetime())
            ax.axvspan(x0, x1, color="grey", alpha=0.15, lw=0.0, zorder=1)
            ax.text(
                (x0 + x1) / 2.0, Y_MAX * 0.93, label,
                fontsize=8, ha="center", va="top", color="0.25", zorder=3,
            )

    ax.plot(dates_num, vol_pct, color="black", linewidth=0.8, zorder=2)

    ax.set_ylim(0.0, Y_MAX)
    ax.set_ylabel("Vol (%)")
    ax.set_title(title, loc="left")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    if show_xlabel:
        ax.set_xlabel("Date")

    max_vol = float(np.nanmax(vol_pct))
    valid = ~np.isnan(gamma_high)
    frac_full = float(np.mean(gamma_high[valid] > 0.5)) if valid.any() else float("nan")
    return max_vol, frac_full, float("nan")  # frac_covid filled by caller


def _covid_frac_for(name: str, K: int, canonical_index: pd.DatetimeIndex) -> Tuple[float, int]:
    """Compute fraction of COVID-window dates with gamma_high > 0.5 (matches R2).
    Also return n_covid_days for sanity."""
    g_high = _load_high_regime(name, K, canonical_index)
    covid_mask = (canonical_index >= COVID_START) & (canonical_index <= COVID_END)
    valid = covid_mask & ~np.isnan(g_high)
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(g_high[valid] > 0.5)), n


def main() -> None:
    print("R3: regime overlay figure (revised)")
    print()

    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["pdf.fonttype"] = 42  # embed Type 42 / TrueType for selectable text

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rv_gk = get_canonical_rv_gk().astype(np.float64)
    canonical_index = pd.DatetimeIndex(rv_gk.index)
    vol_pct = (np.sqrt(rv_gk.to_numpy(dtype=np.float64)) * 100.0).astype(np.float64)
    print(
        f"canonical rv_gk: T={len(canonical_index)}, "
        f"{canonical_index[0].date()} → {canonical_index[-1].date()}"
    )

    dates_num = mdates.date2num(canonical_index.to_pydatetime()).astype(np.float64)
    cmap = _make_shading_cmap()

    fig, axes = plt.subplots(
        nrows=len(PANELS),
        ncols=1,
        figsize=(7.0, 6.5),
        sharex=True,
    )

    summary = {}
    for i, (name, title, K) in enumerate(PANELS):
        gamma_high = _load_high_regime(name, K, canonical_index)
        is_top = (i == 0)
        is_bot = (i == len(PANELS) - 1)
        max_vol, frac_full, _ = _render_panel(
            axes[i],
            dates_num=dates_num,
            vol_pct=vol_pct,
            gamma_high=gamma_high,
            title=title,
            show_xlabel=is_bot,
            annotate_stress=is_top,
            cmap=cmap,
        )
        frac_covid, n_covid = _covid_frac_for(name, K, canonical_index)
        summary[name] = {
            "max_vol": max_vol,
            "frac_full": frac_full,
            "frac_covid": frac_covid,
            "n_covid": n_covid,
        }

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlim(dates_num[0], dates_num[-1])

    fig.tight_layout()

    out_path = RESULTS_DIR / "regime_overlay.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")

    # Also write a PNG preview at dpi=200 (publication PDF stays the vector
    # source of truth; PNG is for screen preview and informal sharing).
    png_path = RESULTS_DIR / "regime_overlay.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    # ── Validation ────────────────────────────────────────────────
    print()
    print("── Validation ────────────────────────────────────────────")

    # (a) file exists, size > 0
    assert out_path.exists(), f"PDF not created at {out_path}"
    size = out_path.stat().st_size
    assert size > 0, f"PDF has zero size: {out_path}"
    print(f"  (a) {out_path} exists, size = {size} bytes")

    # (b) date range
    print(
        f"  (b) figure date range: "
        f"{canonical_index[0].date()} → {canonical_index[-1].date()}"
    )
    assert str(canonical_index[0].date()) == "2019-01-02", "start date mismatch"
    assert str(canonical_index[-1].date()) == "2024-12-31", "end date mismatch"

    # (c) max vol_pct (shared across panels)
    print("  (c) max vol_pct per panel:")
    mvs = []
    for name, _, _ in PANELS:
        mv = summary[name]["max_vol"]
        mvs.append(mv)
        print(f"        {name.upper():<5}: {mv:.4f}")
    assert all(mv == mvs[0] for mv in mvs), (
        f"max vol differs across panels: {mvs} (should all be identical)"
    )
    clipped = mvs[0] > Y_MAX
    print(
        f"        Y_MAX={Y_MAX:g}; line clipped: {clipped} "
        f"(peak {mvs[0]:.4f} {'>' if clipped else '<='} Y_MAX)"
    )

    # (d) full-sample fraction with gamma_high > 0.5
    print("  (d) full-sample fraction γ_high > 0.5 per panel:")
    expected_full = {"hmm": (0.20, 0.30), "msar": (0.30, 0.50), "iohmm": (0.30, 0.45)}
    for name, _, _ in PANELS:
        f = summary[name]["frac_full"]
        lo, hi = expected_full[name]
        in_range = lo <= f <= hi
        marker = "" if in_range else "  ← outside expected range"
        print(f"        {name.upper():<5}: {f:.4f}  (expected {lo}-{hi}){marker}")
        if f > 0.7 or f < 0.05:
            print(f"        ⚠ FLAG: {name} fraction {f:.4f} far outside reasonable range")
            assert False, f"{name} fraction {f} outside [0.05, 0.7]"

    # (e) COVID-window fraction must match R2's results/regime_metrics.csv
    r2_csv = RESULTS_DIR / "regime_metrics.csv"
    if not r2_csv.exists():
        raise FileNotFoundError(
            f"R2 output not found at {r2_csv}; cannot run cross-check (e)."
        )
    r2_df = pd.read_csv(r2_csv)
    r2_covid = (
        r2_df.groupby("model")["covid_frac_high_regime"].first().to_dict()
    )
    r2_n_covid = r2_df.groupby("model")["n_covid_days"].first().to_dict()
    print("  (e) COVID-window (2020-02-24 → 2020-04-30) frac γ_high > 0.5 vs R2:")
    print(f"      R2 n_covid_days (per model): {r2_n_covid}")
    drift_flag = False
    for name, _, _ in PANELS:
        ours = summary[name]["frac_covid"]
        ours_n = summary[name]["n_covid"]
        r2_val = float(r2_covid[name])
        diff = abs(ours - r2_val)
        ok = (diff < 1e-9) and (ours_n == r2_n_covid[name])
        marker = "" if ok else f"  ⚠ MISMATCH (|Δ|={diff:.6f})"
        print(
            f"        {name.upper():<5} R3={ours:.4f} (n={ours_n})  "
            f"R2={r2_val:.4f} (n={r2_n_covid[name]}){marker}"
        )
        if not ok:
            drift_flag = True
    assert not drift_flag, "R3 vs R2 COVID-frac mismatch — internal inconsistency"

    print()
    # PNG preview existence + size sanity (PDF is the publication artifact;
    # PNG is just for screen preview).
    png_path = RESULTS_DIR / "regime_overlay.png"
    assert out_path.exists(), f"PDF missing: {out_path}"
    assert png_path.exists(), f"PNG missing: {png_path}"
    pdf_size = out_path.stat().st_size
    png_size = png_path.stat().st_size
    print(f"  PDF: {out_path}  ({pdf_size} bytes)")
    print(f"  PNG: {png_path}  ({png_size} bytes)")
    if png_size < 50_000 or png_size > 2_000_000:
        print(
            f"  ⚠ PNG size {png_size} bytes outside the expected ~200-600 KB "
            f"range for a 7×6.5\" figure at dpi=200."
        )
    print()
    print("R3 complete: figure written.")
    print()
    print("── LaTeX figure block ────────────────────────────────────")
    print(r"\begin{figure}[t]")
    print(r"\centering")
    print(r"\includegraphics[width=\textwidth]{results/regime_overlay.pdf}")
    print(
        r"\caption{Smoothed posterior probability of the high volatility regime "
        r"(red shading, intensity proportional to probability) overlaid on the "
        r"realized volatility series for SPY (black line, "
        r"$\sqrt{\widehat{\mathrm{RV}}_{GK,t}} \cdot 100$). "
        r"Panels show the HMM ($K=3$), MS-AR ($K=2$), and IOHMM ($K=2$) "
        r"respectively. The HMM fit produces a persistent low volatility regime "
        r"punctuated by short single day high regime visits. The MS-AR switches "
        r"more uniformly between its two regimes across the sample. The IOHMM's "
        r"input dependent transitions, driven by lagged macro features on TLT, "
        r"HYG, UUP, and GLD, produce broader high regime activation including "
        r"sustained elevation during the March 2020 COVID episode and the 2022 "
        r"inflation driven repricing.}"
    )
    print(r"\label{fig:regime_overlay}")
    print(r"\end{figure}")


if __name__ == "__main__":
    main()
