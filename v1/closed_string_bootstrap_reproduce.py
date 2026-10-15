"""
Minimal reproduction utilities for the closed-string bootstrap note.

This file is intentionally smaller than the exploratory notebooks.  It contains
only the data-loading, normalization, summary, and plotting code needed to
reproduce the main numerical figures from the saved CSV files.

The expensive Gurobi scans are kept in the original notebooks.  This module is
for producing polished tables and figures from the scan outputs.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "closed_string_bootstrap_matplotlib_cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import zeta


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT

GRAVITY_POLE_RHS_SCALE = 1.0e-4

C00_STRING = 2.0 * float(zeta(3.0))
C10_STRING = -2.0 * float(zeta(5.0))
C20_STRING = 2.0 * float(zeta(7.0))

POLE_SUBTRACTED_X_STRING = C10_STRING / C00_STRING
POLE_SUBTRACTED_Y_STRING = C20_STRING / C00_STRING


def _parse_value(value):
    if value is None:
        return value
    text = str(value).strip()
    if text == "":
        return np.nan
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        return float(text)
    except ValueError:
        return value


def load_records(path, data_dir=ROOT):
    """Load a CSV file and convert numeric/bool-looking entries."""

    path = Path(path)
    if not path.is_absolute():
        path = Path(data_dir) / path
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        return [{key: _parse_value(value) for key, value in row.items()} for row in csv.DictReader(f)]


def finite_float(value, default=np.nan):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def save_figure(fig, path):
    """Save both PNG and PDF versions of a figure."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path


def note_output_path(name, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


def _closed(points):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(0, 2)
    return np.vstack([points, points[0]])


def _polygon_from_boundary(records):
    points = np.asarray(
        [
            [finite_float(rec.get("x")), finite_float(rec.get("y"))]
            for rec in records
            if bool(rec.get("ok", False))
        ],
        dtype=float,
    )
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 3:
        return points
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    return points[order]


def _band_arrays(records, x_key="x", y_min_key="y_min", y_max_key="y_max"):
    rows = []
    for rec in records:
        x = finite_float(rec.get(x_key))
        y_min = finite_float(rec.get(y_min_key))
        y_max = finite_float(rec.get(y_max_key))
        if bool(rec.get("ok", rec.get("ok_min", False))) and np.isfinite([x, y_min, y_max]).all():
            rows.append((x, min(y_min, y_max), max(y_min, y_max)))
    rows.sort(key=lambda item: item[0])
    if not rows:
        return np.asarray([]), np.asarray([]), np.asarray([])
    return tuple(np.asarray(col, dtype=float) for col in zip(*rows))


def _band_polygon(records, x_key="x", y_min_key="y_min", y_max_key="y_max"):
    x, y_min, y_max = _band_arrays(records, x_key=x_key, y_min_key=y_min_key, y_max_key=y_max_key)
    if x.size == 0:
        return np.asarray([]).reshape(0, 2)
    upper = np.column_stack([x, y_max])
    lower = np.column_stack([x[::-1], y_min[::-1]])
    return np.vstack([upper, lower])


def gravity_g0_convergence_table(csv_name="amxsusy_g0_null_convergence.csv", data_dir=ROOT):
    """Return the note-normalized g0 convergence table."""

    rows = []
    for rec in load_records(csv_name, data_dir=data_dir):
        raw = finite_float(rec.get("c00_max"))
        rows.append(
            {
                "max_power": int(finite_float(rec.get("max_power"), 0.0)),
                "num_null": int(finite_float(rec.get("num_null"), 0.0)),
                "ok": bool(rec.get("ok", False)),
                "c00_raw": raw,
                "c00_over_8piG": raw / GRAVITY_POLE_RHS_SCALE if np.isfinite(raw) else np.nan,
            }
        )
    return rows


def plot_gravity_g0_convergence(output_dir=DEFAULT_OUTPUT_DIR, data_dir=ROOT):
    """Plot the convergence of max c00 M^6/(8 pi G) with null constraints."""

    rows = [rec for rec in gravity_g0_convergence_table(data_dir=data_dir) if rec["ok"]]
    x = np.asarray([rec["num_null"] for rec in rows], dtype=float)
    y = np.asarray([rec["c00_over_8piG"] for rec in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=180)
    ax.plot(x, y, marker="o", color="0.15", linewidth=1.4)
    ax.axhline(C00_STRING, color="#c62828", linestyle="--", linewidth=1.2, label="Virasoro-Shapiro")
    ax.set_xlabel("number of null constraints")
    ax.set_ylabel(r"max $c_{0,0}M^6/(8\pi G)$")
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    path = save_figure(fig, note_output_path("gravity_g0_null_convergence.png", output_dir))
    return fig, ax, path


def _string_curve(num=400):
    alpha = np.linspace(0.0, 1.0, int(num))
    c00 = 2.0 * float(zeta(3.0)) * alpha**3
    c10 = -2.0 * float(zeta(5.0)) * alpha**5
    return c00, c10


def plot_gravity_allowed_region(output_dir=DEFAULT_OUTPUT_DIR, data_dir=ROOT):
    """Plot the gravity-pole NL3/direct-condition allowed region."""

    level3 = load_records("level3_adaptive_final.csv", data_dir=data_dir)
    direct = load_records("amxsusy_gravity_pole_c00_c10_direct_c3_scan_band.csv", data_dir=data_dir)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=180)

    x, y_min, y_max = _band_arrays(level3, x_key="c00", y_min_key="c10_min", y_max_key="c10_max")
    if x.size:
        ax.fill_between(x, y_min, y_max, color="0.82", alpha=0.55, label="NL3_allowed")
        ax.plot(x, y_min, color="0.35", linewidth=1.1)
        ax.plot(x, y_max, color="0.35", linewidth=1.1)

    x, y_min, y_max = _band_arrays(direct, x_key="c00", y_min_key="c10_min", y_max_key="c10_max")
    if x.size:
        ax.fill_between(x, y_min, y_max, color="#9dd9cd", alpha=0.65, label="NL6_allowed")
        ax.plot(x, y_min, color="#00796b", linewidth=1.3)
        ax.plot(x, y_max, color="#00796b", linewidth=1.3)

    sx, sy = _string_curve()
    ax.plot(sx, sy, color="#c62828", linestyle="--", linewidth=1.4, label="Virasoro-Shapiro")
    ax.plot(C00_STRING, C10_STRING, marker="o", color="#c62828", markersize=4.5, linestyle="None")

    ax.set_xlabel(r"$c_{0,0}M^6/(8\pi G)$")
    ax.set_ylabel(r"$c_{1,0}M^{10}/(8\pi G)$")
    ax.set_xlim(-0.03, 2.48)
    ax.set_ylim(-2.18, 0.08)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()

    path = save_figure(fig, note_output_path("gravity_allowed_region.png", output_dir))
    return fig, ax, path


def plot_pole_subtracted_allowed_region(output_dir=DEFAULT_OUTPUT_DIR, data_dir=ROOT):
    """Plot the pole-subtracted NL0/NL3/NL6 allowed region."""

    no_nl = load_records("pole_subtracted_c10_c20_gurobi_no_NL_boundary.csv", data_dir=data_dir)
    nl3 = load_records("pole_subtracted_c10_c20_gurobi_NL3_boundary.csv", data_dir=data_dir)
    nl6 = load_records("pole_subtracted_c10_c20_gurobi_NL6_band.csv", data_dir=data_dir)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=180)

    pts = _polygon_from_boundary(no_nl)
    if pts.shape[0] >= 3:
        closed = _closed(pts)
        ax.fill(closed[:, 0], closed[:, 1], color="0.82", alpha=0.42, label="NL0_allowed")
        ax.plot(closed[:, 0], closed[:, 1], color="0.35", linewidth=1.2)

    pts = _polygon_from_boundary(nl3)
    if pts.shape[0] >= 3:
        closed = _closed(pts)
        ax.fill(closed[:, 0], closed[:, 1], color="#8dbbd8", alpha=0.42, label="NL3_allowed")
        ax.plot(closed[:, 0], closed[:, 1], color="#2f6f9f", linewidth=1.3)

    pts = _band_polygon(nl6)
    if pts.shape[0] >= 3:
        closed = _closed(pts)
        ax.fill(closed[:, 0], closed[:, 1], color="#f2a65a", alpha=0.44, label="NL6_allowed")
        ax.plot(closed[:, 0], closed[:, 1], color="#b85c00", linewidth=1.35)

    ax.plot(
        POLE_SUBTRACTED_X_STRING,
        POLE_SUBTRACTED_Y_STRING,
        marker="o",
        color="#c62828",
        markersize=4.5,
        linestyle="None",
        label="Virasoro-Shapiro",
    )

    ax.set_xlabel(r"$M^4 c_{1,0}/c_{0,0}$")
    ax.set_ylabel(r"$M^8 c_{2,0}/c_{0,0}$")
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()

    path = save_figure(fig, note_output_path("pole_subtracted_allowed_region.png", output_dir))
    return fig, ax, path


def _display_label(label):
    label = str(label)
    label = label.replace("J_max", r"J_{\max}")
    label = label.replace("J_{max}", r"J_{\max}")
    return label


def plot_spectrum_panel(records, output_path, j_max, mu_max, label_filter=None, figsize=(5.4, 4.0)):
    """Plot one spectrum panel from spectrum CSV records."""

    if label_filter is not None:
        records = [rec for rec in records if str(rec.get("label", "")) == str(label_filter)]

    labels = []
    for rec in records:
        label = rec.get("setting_label", "")
        if label not in labels:
            labels.append(label)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    for idx, label in enumerate(labels):
        rows = []
        for rec in records:
            j = finite_float(rec.get("J"))
            mu = finite_float(rec.get("mu"))
            if rec.get("setting_label", "") == label and j <= j_max and mu <= mu_max:
                rows.append((j, mu))
        if not rows:
            continue
        rows = np.asarray(rows, dtype=float)
        ax.scatter(
            rows[:, 0],
            rows[:, 1],
            s=24,
            color=colors[idx % len(colors)],
            edgecolor="none",
            alpha=0.9,
            label=_display_label(label),
        )

    ax.set_xlim(-1.0, float(j_max))
    ax.set_ylim(0.0, float(mu_max))
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\mu$")
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    path = save_figure(fig, output_path)
    return fig, ax, path


def plot_pole_subtracted_c20_min_spectra(output_dir=DEFAULT_OUTPUT_DIR, data_dir=ROOT):
    """Plot the full and zoomed c20_min extremal spectra."""

    records = load_records("pole_subtracted_c10_c20_gurobi_NL6_c20_min_spectrum_settings_points.csv", data_dir=data_dir)
    full = plot_spectrum_panel(
        records,
        note_output_path("pole_subtracted_c20_min_spectrum_full.png", output_dir),
        j_max=50.0,
        mu_max=40.0,
        label_filter="c20_min",
    )
    zoom = plot_spectrum_panel(
        records,
        note_output_path("pole_subtracted_c20_min_spectrum_zoom.png", output_dir),
        j_max=25.0,
        mu_max=7.0,
        label_filter="c20_min",
    )
    return full, zoom


def note_summaries(data_dir=ROOT):
    """Return compact numerical summaries used in the reproduction notebook."""

    g0 = [rec for rec in gravity_g0_convergence_table(data_dir=data_dir) if rec["ok"]]
    level3 = load_records("level3_adaptive_final.csv", data_dir=data_dir)
    direct = load_records("amxsusy_gravity_pole_c00_c10_direct_c3_scan_band.csv", data_dir=data_dir)
    nl6 = load_records("pole_subtracted_c10_c20_gurobi_NL6_band.csv", data_dir=data_dir)

    def feasible_band(records, x_key, y_min_key, y_max_key):
        x, lo, hi = _band_arrays(records, x_key=x_key, y_min_key=y_min_key, y_max_key=y_max_key)
        if x.size == 0:
            return {}
        return {
            "num_columns": int(x.size),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "y_min": float(lo.min()),
            "y_max": float(hi.max()),
        }

    return {
        "g0_last_bound": g0[-1] if g0 else None,
        "gravity_NL3_band": feasible_band(level3, "c00", "c10_min", "c10_max"),
        "gravity_NL6_band": feasible_band(direct, "c00", "c10_min", "c10_max"),
        "pole_subtracted_NL6_band": feasible_band(nl6, "x", "y_min", "y_max"),
        "virasoro_gravity_point": {"c00": C00_STRING, "c10": C10_STRING},
        "virasoro_pole_subtracted_point": {
            "x": POLE_SUBTRACTED_X_STRING,
            "y": POLE_SUBTRACTED_Y_STRING,
        },
    }


def make_all_figures(output_dir=DEFAULT_OUTPUT_DIR, data_dir=ROOT):
    """Regenerate the polished reproduction figures from saved CSV files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    figures["gravity_g0_convergence"] = plot_gravity_g0_convergence(output_dir=output_dir, data_dir=data_dir)[2]
    figures["gravity_allowed_region"] = plot_gravity_allowed_region(output_dir=output_dir, data_dir=data_dir)[2]
    figures["pole_subtracted_allowed_region"] = plot_pole_subtracted_allowed_region(output_dir=output_dir, data_dir=data_dir)[2]
    full, zoom = plot_pole_subtracted_c20_min_spectra(output_dir=output_dir, data_dir=data_dir)
    figures["pole_subtracted_c20_min_spectrum_full"] = full[2]
    figures["pole_subtracted_c20_min_spectrum_zoom"] = zoom[2]
    return figures


if __name__ == "__main__":
    print(note_summaries())
    for name, path in make_all_figures().items():
        print(f"{name}: {path}")
