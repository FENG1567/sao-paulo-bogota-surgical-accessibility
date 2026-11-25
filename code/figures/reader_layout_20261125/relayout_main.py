#!/usr/bin/env python3
"""Layout-only redraw of the five current v13 main figures.

All plotted values come from the frozen Source_Data CSVs. This script changes
panel placement and typography only; it does not estimate or route any data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Importing the shared helper registers the supplied Arial fonts and sets the
# figure-wide Arial Bold/editable-text configuration.
from figure_style import panel_tag, save_figure


X = Path(__file__).resolve().parent
SOURCE_DATA = Path(os.environ.get('FROZEN_SOURCE_DATA',str(X/'Source_Data')))
REFERENCE_PDF = Path(os.environ.get('FROZEN_REFERENCE_PDF',str(X/'reference_pdf')))
OUT = X / "figures_main" / "run01"
SOURCE_DATA = Path(os.environ.get('FROZEN_SOURCE_DATA',str(SOURCE_DATA)))
REFERENCE_PDF = Path(os.environ.get('FROZEN_REFERENCE_PDF',str(REFERENCE_PDF)))

PALETTE = {
    "car": "#4D4D4D", "wt": "#0F4D92", "foot": "#767676",
    "cap1": "#3775BA", "cap2": "#6B9BC7", "cap3": "#A6C3DC",
    "R": "#4D4D4D", "E": "#0F4D92", "A": "#42949E",
    "both": "#4B8F63", "car_only": "#C47A2C", "wt_only": "#267C9C",
    "neither": "#B8B8B8", "indeterminate": "#8C6BB1",
}
CITIES = ("Sao Paulo", "Bogota")
SCOPES = ("R", "E", "A")
DISPLAY_THRESHOLDS = (15, 30, 45)
MODE_LABELS = {
    "car": "Car",
    "foot": "Direct walking",
    "wt": "Transit + walking (WT)",
    "wt_cap1km": "WT, ≤1 km walk cap",
    "wt_cap2km": "WT, ≤2 km walk cap",
    "wt_cap3km": "WT, ≤3 km walk cap",
}
MAP_CLASSES = (
    ("both", "Both covered", PALETTE["both"]),
    ("car_only", "Car only", PALETTE["car_only"]),
    ("wt_only", "WT only", PALETTE["wt_only"]),
    ("neither", "Neither covered", PALETTE["neither"]),
    ("indeterminate", "Technically indeterminate", PALETTE["indeterminate"]),
)
SCOPE_STYLE = {"R": "-", "E": "--", "A": ":"}
CURVE_SCOPE_ORDER = ("R", "E", "A")
CURVE_MODE_ORDER = ("car", "wt")
WALK_STYLES = {
    "foot": (PALETTE["foot"], (0, (3, 1.5))),
    "wt": (PALETTE["wt"], "-"),
    "wt_cap1km": (PALETTE["cap1"], "--"),
    "wt_cap2km": (PALETTE["cap2"], "-."),
    "wt_cap3km": (PALETTE["cap3"], ":"),
}
FILES = {
    "facility": "Figure1_facility_scenario_counts.csv",
    "population": "Table1_population_facility_source.csv",
    "curves": "Figure2_main_curve_source.csv",
    "contrasts": "Figure2_paired_scope_contrast_source.csv",
    "paired": "Figure3_paired_mode_source.csv",
    "gradients": "Figure4_income_gradient_source.csv",
    "maps": "Figure4_origin_classification_source.csv",
    "validation": "Figure5_validation_sensitivity_source.csv",
    "sensitivity": "Figure5_sensitivity_source.csv",
}

PANEL_IDENTITY = {
    "Figure1_population_facilities": ["A", "B"],
    "Figure2_mode_scope_threshold": ["A", "B", "C", "D"],
    "Figure3_walk_caps": ["A", "B", "C", "D", "E", "F"],
    "Figure4_income_geography": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "Figure5_validation_sensitivity": ["A", "B", "C", "D", "E", "F", "G", "H"],
}

DATA: dict[str, pd.DataFrame] = {}
PANEL_ROWS: dict[str, list[dict[str, object]]] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources() -> None:
    for key, filename in FILES.items():
        path = SOURCE_DATA / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen source table: {path}")
        DATA[key] = pd.read_csv(path, dtype={"uid": str})


def record_rows(panel: str, source_key: str, frame: pd.DataFrame, rule: str) -> None:
    """Record exact zero-based CSV row indices for each panel input."""
    PANEL_ROWS.setdefault(panel, []).append({
        "source_csv": FILES[source_key],
        "selection_rule": rule,
        "row_count": int(len(frame)),
        "zero_based_csv_data_row_indices": [int(i) for i in frame.index.tolist()],
    })


def style_axis(ax, title: str | None = None) -> None:
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=5)
    ax.tick_params(axis="both", labelsize=8.5, length=2.5, pad=3)
    ax.grid(color="#E8E8E8", linewidth=0.55)
    ax.set_axisbelow(True)


def add_tag(fig, ax, letter: str, x: float | None = None, y: float | None = None) -> None:
    panel_tag(fig, ax, letter.upper(), x=x, y=y)


def city_label(city: str) -> str:
    return {"Sao Paulo": "São Paulo", "Bogota": "Bogotá"}[city]


def scope_label(city: str, scope: str) -> str:
    return {
        "Sao Paulo": {
            "R": "Registry",
            "E": "Equipment-SUS",
            "A": "Retained historical activity-flag inventory",
        },
        "Bogota": {
            "R": "Registry",
            "E": "Public-provider signal",
        },
    }[city].get(scope, scope)


def fmt_int(value: object) -> str:
    return f"{int(float(value)):,}"


def write_contract(selected_figures: set[int]) -> None:
    contract_path = X / "main_data_contract.json"
    previous_rendered: set[int] = set()
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        previous_rendered = set(previous.get("rendered_figures_this_run", []))
        old_panels = previous.get("panel_source_selection", {})
        for panel, records in old_panels.items():
            PANEL_ROWS.setdefault(panel, records)
    selected_figures = set(selected_figures) | previous_rendered
    source_audit = {}
    for key, filename in FILES.items():
        path = SOURCE_DATA / filename
        frame = DATA[key]
        source_audit[filename] = {
            "sha256": sha256(path),
            "data_row_count": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
        }

    reference_audit = {}
    for path in sorted(REFERENCE_PDF.glob("Figure*.pdf")):
        reference_audit[path.name] = {"sha256": sha256(path)}

    contract = {
        "title": "Layout-only main-figure source and selection contract",
        "backend": "Python / matplotlib",
        "layout_only": True,
        "analysis_reestimated": False,
        "routing_rerun": False,
        "fixed_width_inches": 7.2,
        "font_contract": {
            "family": "Arial",
            "weight": "bold",
            "panel_tags": "uppercase A–H, Arial Bold 18 pt",
            "minimum_axis_tick_legend_note_font_pt": 8.5,
            "axis_label_font_pt": 10,
            "panel_title_font_pt": 11,
            "font_files": ["fonts/arial.ttf", "fonts/arialbd.ttf"],
        },
        "panel_identity": PANEL_IDENTITY,
        "expected_total_panels": 25,
        "rendered_figures_this_run": sorted(selected_figures),
        "frozen_source_directory": str(SOURCE_DATA),
        "source_csv_audit": source_audit,
        "current_reference_pdf_audit": reference_audit,
        "panel_source_selection": PANEL_ROWS,
        "selection_note": (
            "Each panel selection uses the listed zero-based row indices from the "
            "named current frozen CSV. Plotted estimates, confidence intervals, "
            "technical bounds, mode/scope/category assignments, and origin "
            "coordinates are read directly from those rows."
        ),
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")


def add_bottom_notes(fig, lines: list[str], y: float, x: float = 0.045,
                     fontsize: float = 8.5, linespacing: float = 1.15) -> None:
    fig.text(
        x, y, "\n".join(lines), ha="left", va="bottom",
        fontsize=fontsize, fontweight="bold", linespacing=linespacing,
    )


def figure1() -> None:
    facility = DATA["facility"]
    population = DATA["population"]
    pop = population.loc[
        population.record_type.astype(str).eq("population_frame")
        & population.measure.astype(str).eq("fixed origin frame")
    ].copy()
    pop = pop.set_index("city").loc[list(CITIES)].reset_index()
    facility = facility.loc[
        facility.scope_label.astype(str).isin(["R", "E", "A"])
    ].copy()
    city_order = {"Sao Paulo": 0, "Bogota": 1}
    scope_order = {"R": 0, "E": 1, "A": 2}
    facility["_city_order"] = facility.city.map(city_order)
    facility["_scope_order"] = facility.scope_label.map(scope_order)
    facility = facility.sort_values(["_city_order", "_scope_order"]).drop(
        columns=["_city_order", "_scope_order"]
    )
    record_rows(
        "Figure1A", "population", pop,
        "record_type == population_frame; measure == fixed origin frame; city order São Paulo, Bogotá",
    )
    record_rows(
        "Figure1B", "facility", facility,
        "scope_label in R/E/A; preserve all current city/scope rows in original display order",
    )

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(
        1, 2, width_ratios=(0.9, 1.1),
        left=0.105, right=0.98, bottom=0.22, top=0.90, wspace=0.32,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    add_tag(fig, ax0, "A")
    add_tag(fig, ax1, "B")

    sizes = pd.to_numeric(pop.estimate, errors="raise").to_numpy(float)
    yy = np.arange(len(pop))
    ax0.barh(yy, sizes, color=[PALETTE["E"], "#767676"], height=0.54, edgecolor="white")
    max_size = float(np.max(sizes))
    for yv, value, city in zip(yy, sizes, pop.city.astype(str)):
        kind = "sampled origins" if city == "Sao Paulo" else "census origins"
        ax0.text(value + max_size * 0.035, yv, f"{int(value):,} {kind}",
                 va="center", fontsize=8.5, fontweight="bold")
    ax0.set_yticks(yy, [city_label(c) for c in pop.city.astype(str)])
    ax0.invert_yaxis()
    ax0.set_xlim(0, max_size * 1.78)
    ax0.set_xticks([0, 500, 1000, 1500])
    ax0.set_xlabel("Origins", fontsize=10, fontweight="bold")
    ax0.set_title("Analysis-origin counts", loc="left", fontsize=11, fontweight="bold", pad=5)
    ax0.grid(axis="x", color="#E8E8E8", linewidth=0.55)
    ax0.spines["left"].set_visible(False)
    ax0.tick_params(axis="both", labelsize=8.5, length=2.5, pad=3)
    ax0.set_axisbelow(True)

    sp_label = {
        "R": "São Paulo\nRegistry",
        "E": "São Paulo\nEquipment-SUS",
        "A": "São Paulo\nRetained historical\nactivity-flag inventory",
    }
    bo_label = {"R": "Bogotá\nRegistry", "E": "Bogotá\nPublic-provider signal"}
    labels = [
        (sp_label if city == "Sao Paulo" else bo_label).get(scope, f"{city} · {scope}")
        for city, scope in zip(facility.city.astype(str), facility.scope_label.astype(str))
    ]
    values = pd.to_numeric(facility.facility_count, errors="raise").to_numpy(float)
    yvals = np.arange(len(facility))
    ax1.barh(
        yvals, values,
        color=[PALETTE.get(scope, "#767676") for scope in facility.scope_label.astype(str)],
        height=0.62, edgecolor="white",
    )
    for yv, value in zip(yvals, values):
        ax1.text(value + max(1, float(values.max())) * 0.028, yv, f"{int(value):,}",
                 va="center", fontsize=8.5, fontweight="bold")
    ax1.set_yticks(yvals, labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, float(values.max()) * 1.34)
    ax1.set_xlabel("Facilities", fontsize=10, fontweight="bold")
    ax1.set_title("Scenario counts (not a nested funnel)", loc="left",
                  fontsize=11, fontweight="bold", pad=5)
    ax1.grid(axis="x", color="#E8E8E8", linewidth=0.55)
    ax1.tick_params(axis="both", labelsize=8.5, length=2.5, pad=3)
    ax1.set_axisbelow(True)

    fig.suptitle(
        "Analysis origins and facility definitions",
        x=0.045, y=0.978, ha="left", fontsize=12, fontweight="bold",
    )
    add_bottom_notes(fig, [
        "São Paulo: Registry=339, Equipment-SUS=89, retained historical activity-flag inventory=118;",
        "the latter two overlap at 78 sites but are not nested.",
        "SP: 1,100 sampled origins (stratified probability sample); BO: 1,057 complete-grid census origins.",
        "Equipment-SUS is not patient eligibility; Bogotá's 16-site Public-provider signal is a different construct.",
        "Full overlaps are in Table 1.",
    ], y=0.035, fontsize=8.5)
    save_figure(fig, OUT, "Figure1_population_facilities")


def curves_for_city(curves: pd.DataFrame, city: str, scopes=CURVE_SCOPE_ORDER,
                    modes=CURVE_MODE_ORDER) -> pd.DataFrame:
    d = curves.loc[
        curves.city.astype(str).eq(city)
        & curves.scope.astype(str).isin(scopes)
        & curves["mode"].astype(str).isin(modes)
        & ~(curves.city.astype(str).eq("Bogota") & curves.scope.astype(str).eq("A"))
    ].copy()
    return d


def finite_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    vals = []
    for col in columns:
        if col in frame:
            v = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
            vals.append(v[np.isfinite(v)])
    return np.concatenate(vals) if vals else np.array([], dtype=float)


def figure2() -> None:
    curves = DATA["curves"]
    contrasts = DATA["contrasts"]
    contrast_scopes = ("E", "A")
    display_curves = curves.loc[
        curves.city.astype(str).isin(CITIES)
        & curves.scope.astype(str).isin(CURVE_SCOPE_ORDER)
        & curves["mode"].astype(str).isin(CURVE_MODE_ORDER)
        & ~(curves.city.astype(str).eq("Bogota") & curves.scope.astype(str).eq("A"))
    ].copy()
    display_contrasts = contrasts.loc[
        contrasts.city.astype(str).isin(CITIES)
        & contrasts.scope1.astype(str).eq("R")
        & contrasts.scope2.astype(str).isin(contrast_scopes)
    ].copy()
    upper = pd.to_numeric(display_curves.coverage_upper, errors="coerce").to_numpy(float)
    lower = pd.to_numeric(display_curves.coverage_lower, errors="coerce").to_numpy(float)
    valid = np.isfinite(lower) & np.isfinite(upper)
    coverage_bounds_differ = bool(valid.any() and not np.allclose(lower[valid], upper[valid], rtol=0, atol=1e-8))
    dpoint = pd.to_numeric(display_contrasts.contrast, errors="coerce").to_numpy(float)
    dblo = pd.to_numeric(display_contrasts.missing_bound_lower, errors="coerce").to_numpy(float)
    dbhi = pd.to_numeric(display_contrasts.missing_bound_upper, errors="coerce").to_numpy(float)
    valid_lo = np.isfinite(dpoint) & np.isfinite(dblo)
    valid_hi = np.isfinite(dpoint) & np.isfinite(dbhi)
    contrast_bounds_differ = bool(
        (valid_lo.any() and not np.allclose(dpoint[valid_lo], dblo[valid_lo], rtol=0, atol=1e-8))
        or (valid_hi.any() and not np.allclose(dpoint[valid_hi], dbhi[valid_hi], rtol=0, atol=1e-8))
    )
    for pid, city in (("Figure2A", "Sao Paulo"), ("Figure2B", "Bogota")):
        d = display_curves.loc[display_curves.city.astype(str).eq(city)]
        record_rows(pid, "curves", d, "city == requested city; mode in car/wt; scope in R/E/A; Bogotá A excluded as in current figure")
    for pid, city in (("Figure2C", "Sao Paulo"), ("Figure2D", "Bogota")):
        d = display_contrasts.loc[display_contrasts.city.astype(str).eq(city)]
        record_rows(pid, "contrasts", d, "city == requested city; scope1 == R; scope2 in E/A")

    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(
        2, 2, left=0.12, right=0.98, bottom=0.13, top=0.66,
        hspace=0.38, wspace=0.28,
    )
    axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                     [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    for ax, letter in zip(axes.flat, ("A", "B", "C", "D")):
        add_tag(fig, ax, letter)

    city_titles = {
        "Sao Paulo": "São Paulo · probability sample",
        "Bogota": "Bogotá · census frame",
    }
    for ci, city in enumerate(CITIES):
        ax = axes[0, ci]
        dcity = display_curves.loc[display_curves.city.astype(str).eq(city)]
        n_values = pd.to_numeric(dcity.n, errors="coerce").dropna()
        n_text = f"n={int(n_values.iloc[0]):,}" if len(n_values) else "n unavailable"
        ax.set_title(f"{city_titles[city]} ({n_text})", loc="left",
                     fontsize=11, fontweight="bold", pad=5)
        for scope in CURVE_SCOPE_ORDER:
            if city == "Bogota" and scope == "A":
                continue
            dscope = dcity.loc[dcity.scope.astype(str).eq(scope)]
            for mode in CURVE_MODE_ORDER:
                d = dscope.loc[dscope["mode"].astype(str).eq(mode)].sort_values("threshold")
                if d.empty:
                    continue
                x = pd.to_numeric(d.threshold, errors="coerce").to_numpy(float)
                y = pd.to_numeric(d.coverage_lower, errors="coerce").to_numpy(float)
                color = PALETTE[mode]
                ax.plot(x, y, color=color, linestyle=SCOPE_STYLE[scope],
                        marker="o", ms=3.0, lw=1.25, zorder=3)
                up = pd.to_numeric(d.coverage_upper, errors="coerce").to_numpy(float)
                up_differs = np.isfinite(up) & np.isfinite(y) & (np.abs(up-y) > 1e-8)
                if up_differs.any():
                    ax.plot(x, up, color=color, linestyle="-.", lw=0.85,
                            alpha=0.78, zorder=2)
                ci_lo = pd.to_numeric(d.design_ci_lower, errors="coerce").to_numpy(float)
                ci_hi = pd.to_numeric(d.design_ci_upper, errors="coerce").to_numpy(float)
                ci_mask = (
                    (city == "Sao Paulo")
                    & np.isin(x, DISPLAY_THRESHOLDS)
                    & np.isfinite(ci_lo) & np.isfinite(ci_hi) & np.isfinite(y)
                )
                if ci_mask.any():
                    ax.errorbar(
                        x[ci_mask], y[ci_mask],
                        yerr=np.vstack((np.maximum(0, y[ci_mask]-ci_lo[ci_mask]),
                                        np.maximum(0, ci_hi[ci_mask]-y[ci_mask]))),
                        fmt="none", ecolor=color, elinewidth=0.75, capsize=2.0,
                        alpha=0.9, zorder=4,
                    )
        ax.set_ylim(0, 103)
        ax.set_xlim(9, 61)
        ax.set_xticks((15, 30, 45, 60))
        ax.set_yticks((0, 50, 100))
        if ci == 0:
            ax.set_ylabel("Coverage (%)", fontsize=10, fontweight="bold")
        else:
            ax.tick_params(axis="y", labelleft=True)
        ax.tick_params(axis="x", labelbottom=False, labelsize=8.5)
        style_axis(ax)

    contrast_vals = finite_values(
        display_contrasts,
        ("contrast", "design_ci_lower", "design_ci_upper",
         "missing_bound_lower", "missing_bound_upper"),
    )
    if len(contrast_vals):
        span = max(float(np.max(contrast_vals)-np.min(contrast_vals)), 1.0)
        pad = max(1.0, span*0.04)
        y_min = min(-30.0, math.floor((float(np.min(contrast_vals))-pad)/5)*5)
        y_max = max(45.0, math.ceil((float(np.max(contrast_vals))+pad)/5)*5)
    else:
        y_min, y_max = -30.0, 45.0
    for ci, city in enumerate(CITIES):
        ax = axes[1, ci]
        ax.set_title(f"{city_label(city)} · descriptive additive contrasts",
                     loc="left", fontsize=11, fontweight="bold", pad=5)
        dcity = display_contrasts.loc[display_contrasts.city.astype(str).eq(city)]
        for scope2 in contrast_scopes:
            d = dcity.loc[dcity.scope2.astype(str).eq(scope2)].sort_values("threshold")
            if d.empty:
                continue
            x = pd.to_numeric(d.threshold, errors="coerce").to_numpy(float)
            y = pd.to_numeric(d.contrast, errors="coerce").to_numpy(float)
            color = PALETTE[scope2]
            ax.plot(x, y, color=color, marker="o", ms=3.0, lw=1.25, zorder=3)
            lo_bound = pd.to_numeric(d.missing_bound_lower, errors="coerce").to_numpy(float)
            hi_bound = pd.to_numeric(d.missing_bound_upper, errors="coerce").to_numpy(float)
            if np.any(np.isfinite(lo_bound) & np.isfinite(y) & (np.abs(lo_bound-y)>1e-8)):
                ax.plot(x, lo_bound, color=color, linestyle="-.", lw=0.85, alpha=0.78, zorder=2)
            if np.any(np.isfinite(hi_bound) & np.isfinite(y) & (np.abs(hi_bound-y)>1e-8)):
                ax.plot(x, hi_bound, color=color, linestyle="-.", lw=0.85, alpha=0.78, zorder=2)
            ci_lo = pd.to_numeric(d.design_ci_lower, errors="coerce").to_numpy(float)
            ci_hi = pd.to_numeric(d.design_ci_upper, errors="coerce").to_numpy(float)
            ci_mask = (
                (city == "Sao Paulo")
                & np.isin(x, DISPLAY_THRESHOLDS)
                & np.isfinite(ci_lo) & np.isfinite(ci_hi) & np.isfinite(y)
            )
            if ci_mask.any():
                ax.errorbar(
                    x[ci_mask], y[ci_mask],
                    yerr=np.vstack((np.maximum(0, y[ci_mask]-ci_lo[ci_mask]),
                                    np.maximum(0, ci_hi[ci_mask]-y[ci_mask]))),
                    fmt="none", ecolor=color, elinewidth=0.75, capsize=2.0,
                    alpha=0.9, zorder=4,
                )
        ax.axhline(0, color="#767676", lw=0.8, linestyle="--", zorder=1)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(9, 61)
        ax.set_xticks((15, 30, 45, 60))
        if ci == 0:
            ax.set_ylabel("D (percentage points)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Threshold (min)", fontsize=10, fontweight="bold", labelpad=3)
        ax.tick_params(axis="both", labelsize=8.5)
        style_axis(ax)

    scope_labels = [
        Line2D([0], [0], color=PALETTE["car"], lw=1.5, marker="o", ms=3,
               label="Car"),
        Line2D([0], [0], color=PALETTE["wt"], lw=1.5, marker="o", ms=3,
               label="WT (transit + walking)"),
        Line2D([0], [0], color="#3F3F3F", lw=1.4, linestyle="-",
               label="Registry (city-specific)"),
        Line2D([0], [0], color="#3F3F3F", lw=1.4, linestyle="--",
               label="Equipment-SUS (São Paulo)"),
        Line2D([0], [0], color="#3F3F3F", lw=1.4, linestyle="--",
               label="Public-provider signal (Bogotá)"),
        Line2D([0], [0], color="#3F3F3F", lw=1.4, linestyle=":",
               label="Retained historical activity-flag inventory (São Paulo)"),
    ]
    fig.legend(
        handles=scope_labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=2, fontsize=8.5, handlelength=1.5, handletextpad=0.4,
        columnspacing=0.7, labelspacing=0.35, borderaxespad=0.0,
    )
    contrast_handles = [
        Line2D([0], [0], color=PALETTE["E"], lw=1.4, marker="o", ms=3,
               label="D: Registry − Equipment-SUS (SP) / Public-provider signal (Bogotá)"),
        Line2D([0], [0], color=PALETTE["A"], lw=1.4, marker="o", ms=3,
               label="D: Registry − Legacy activity (SP only)"),
    ]
    if not display_contrasts.empty:
        if display_contrasts.city.astype(str).eq("Sao Paulo").any():
            contrast_handles.append(
                Line2D([0], [0], color="#555555", lw=0.8, marker="|", ms=5,
                       label="95% design CI at 15/30/45 (SP only)")
            )
        if coverage_bounds_differ or contrast_bounds_differ:
            contrast_handles.append(
                Line2D([0], [0], color="#555555", lw=0.85, linestyle="-.",
                       label="Technical bound (not CI)")
            )
    fig.legend(
        handles=contrast_handles, loc="upper center", bbox_to_anchor=(0.5, 0.825),
        ncol=1, fontsize=8.5, handlelength=1.5, handletextpad=0.4,
        labelspacing=0.3, borderaxespad=0.0,
    )
    fig.suptitle(
        "Mode, facility scope and threshold",
        x=0.045, y=0.982, ha="left", fontsize=12, fontweight="bold",
    )
    note = "D is descriptive, not causal."
    if coverage_bounds_differ:
        note += " Dash-dot lines show technical upper bounds, not CIs."
    else:
        note += " Technical bounds coincide with coverage."
    if contrast_bounds_differ and not coverage_bounds_differ:
        note += " Dash-dot D lines show technical bounds, not CIs."
    add_bottom_notes(fig, [note], y=0.035, fontsize=8.5)
    save_figure(fig, OUT, "Figure2_mode_scope_threshold")


def figure3() -> None:
    curves = DATA["curves"]
    paired = DATA["paired"]
    display_curves = curves.loc[
        curves.city.astype(str).isin(CITIES)
        & curves.scope.astype(str).isin(SCOPES)
        & curves["mode"].astype(str).isin(WALK_STYLES)
        & ~(curves.city.astype(str).eq("Bogota") & curves.scope.astype(str).eq("A"))
    ].copy()
    lower = pd.to_numeric(display_curves.coverage_lower, errors="coerce").to_numpy(float)
    upper = pd.to_numeric(display_curves.coverage_upper, errors="coerce").to_numpy(float)
    valid = np.isfinite(lower) & np.isfinite(upper)
    technical_bounds_differ = bool(valid.any() and not np.allclose(lower[valid], upper[valid], rtol=0, atol=1e-8))

    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(
        3, 2, left=0.12, right=0.98, bottom=0.08, top=0.90,
        hspace=0.40, wspace=0.28,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]
    letters = iter("ABCDEF")
    for row in axes:
        for ax in row:
            add_tag(fig, ax, next(letters))

    note_inputs = display_curves.copy()
    record_rows(
        "Figure3F", "curves", note_inputs,
        "all displayed city/scope/mode curve rows used for sample-size and technical-bound notes",
    )
    for ri, scope in enumerate(SCOPES):
        for ci, city in enumerate(CITIES):
            ax = axes[ri][ci]
            pid = f"Figure3{chr(ord('A') + ri * 2 + ci)}"
            if city == "Bogota" and scope == "A":
                ax.set_axis_off()
                ax.text(0.02, 0.93, "Mode key and definitions", transform=ax.transAxes,
                        ha="left", va="top", fontsize=11, fontweight="bold")
                handles = [
                    Line2D([0], [0], color=color, linestyle=ls, lw=1.6,
                           label=MODE_LABELS[mode])
                    for mode, (color, ls) in WALK_STYLES.items()
                ]
                ax.legend(
                    handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.80),
                    ncol=1, fontsize=8.5, handlelength=1.6, handletextpad=0.35,
                    columnspacing=0.65, labelspacing=0.18, borderaxespad=0.0,
                )
                nvals_sp = pd.to_numeric(curves.loc[curves.city.astype(str).eq("Sao Paulo"), "n"], errors="coerce").dropna()
                nvals_bo = pd.to_numeric(curves.loc[curves.city.astype(str).eq("Bogota"), "n"], errors="coerce").dropna()
                sp_n = fmt_int(nvals_sp.iloc[0]) if len(nvals_sp) else "unavailable"
                bo_n = fmt_int(nvals_bo.iloc[0]) if len(nvals_bo) else "unavailable"
                notes = [
                    f"SP probability sample n={sp_n}; BO census n={bo_n}.",
                    "Whiskers: 95% design CI at 15/30/45 min (SP only).",
                    "Panel annotations give paired 30-min WT−walking Δ;",
                    "brackets are SP CIs.",
                    "All walking caps filter the same returned-route pool.",
                    "Technical bounds coincide with coverage." if not technical_bounds_differ
                    else "Dotted lines show technical upper bounds, not CIs.",
                ]
                ax.text(0.02, 0.32, "\n".join(notes), transform=ax.transAxes,
                        ha="left", va="top", fontsize=8.5, linespacing=1.18)
                # Source rows used by note-panel sample-size/technical-bound content.
                record_rows("Figure3F", "curves", curves.loc[curves.city.astype(str).isin(CITIES)],
                            "both city rows for n annotation; no plotted values are re-estimated")
                continue

            d = curves.loc[
                curves.city.astype(str).eq(city)
                & curves.scope.astype(str).eq(scope)
                & curves["mode"].astype(str).isin(WALK_STYLES)
            ].copy()
            dpaired = paired.loc[
                paired.city.astype(str).eq(city)
                & paired.scope.astype(str).eq(scope)
                & paired.comparison.astype(str).eq("WT minus direct walking")
                & pd.to_numeric(paired.threshold, errors="coerce").eq(30)
            ].copy()
            record_rows(
                pid, "curves", d,
                f"city == {city}; scope == {scope}; mode in direct walking/WT/1,2,3-km caps",
            )
            record_rows(
                pid, "paired", dpaired,
                f"city == {city}; scope == {scope}; comparison == WT minus direct walking; threshold == 30",
            )
            if d.empty:
                continue
            for mode, (color, linestyle) in WALK_STYLES.items():
                dm = d.loc[d["mode"].astype(str).eq(mode)].sort_values("threshold")
                if dm.empty:
                    continue
                x = pd.to_numeric(dm.threshold, errors="coerce").to_numpy(float)
                y = pd.to_numeric(dm.coverage_lower, errors="coerce").to_numpy(float)
                ax.plot(x, y, color=color, linestyle=linestyle, marker="o",
                        ms=3.0, lw=1.2, zorder=3)
                up = pd.to_numeric(dm.coverage_upper, errors="coerce").to_numpy(float)
                up_differs = np.isfinite(up) & np.isfinite(y) & (np.abs(up-y) > 1e-8)
                if up_differs.any():
                    ax.plot(x, up, color=color, linestyle=(0, (1, 2)),
                            lw=0.85, alpha=0.75, zorder=2)
                ci_lo = pd.to_numeric(dm.design_ci_lower, errors="coerce").to_numpy(float)
                ci_hi = pd.to_numeric(dm.design_ci_upper, errors="coerce").to_numpy(float)
                ci_mask = np.isin(x, DISPLAY_THRESHOLDS) & np.isfinite(ci_lo) & np.isfinite(ci_hi) & np.isfinite(y)
                if ci_mask.any():
                    ax.errorbar(
                        x[ci_mask], y[ci_mask],
                        yerr=np.vstack((np.maximum(0, y[ci_mask]-ci_lo[ci_mask]),
                                        np.maximum(0, ci_hi[ci_mask]-y[ci_mask]))),
                        fmt="none", ecolor=color, elinewidth=0.75, capsize=1.8,
                        alpha=0.9, zorder=4,
                    )
            annotation = ""
            if not dpaired.empty:
                r = dpaired.iloc[0]
                value = pd.to_numeric(pd.Series([r.get("difference_pp")]), errors="coerce").iloc[0]
                lo = pd.to_numeric(pd.Series([r.get("design_ci_lower_pp")]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([r.get("design_ci_upper_pp")]), errors="coerce").iloc[0]
                annotation = f"Δ30 WT−foot {value:+.1f} pp"
                if np.isfinite(lo) and np.isfinite(hi):
                    annotation += f" [{lo:+.1f}, {hi:+.1f}]"
            ax.text(
                0.98, 0.05, annotation, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, fontweight="bold",
                color=PALETTE["wt"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
            )
            title_scope = scope_label(city, scope)
            if city == "Bogota" and scope == "E":
                title_scope = "Public-provider signal"
            if city == "Sao Paulo" and scope == "A":
                title_scope = "Retained historical\nactivity-flag inventory"
            ax.set_title(f"{city_label(city)} · {title_scope}", loc="left",
                         fontsize=11, fontweight="bold", pad=5)
            ax.set_ylim(0, 103)
            ax.set_xlim(9, 61)
            ax.set_xticks((15, 30, 45, 60))
            ax.set_xlabel("Threshold (min)", fontsize=10, fontweight="bold", labelpad=3)
            if ci == 0:
                ax.set_ylabel("Coverage (%)", fontsize=10, fontweight="bold")
            ax.tick_params(axis="both", labelsize=8.5)
            style_axis(ax)

    fig.suptitle(
        "Direct walking, WT net coverage and common-pool walking caps",
        x=0.045, y=0.982, ha="left", fontsize=12, fontweight="bold",
    )
    save_figure(fig, OUT, "Figure3_walk_caps")


def figure4() -> None:
    gradients = DATA["gradients"]
    maps = DATA["maps"]
    gd = gradients.loc[
        gradients.city.astype(str).isin(CITIES)
        & pd.to_numeric(gradients.threshold, errors="coerce").eq(30)
        & gradients.scope.astype(str).isin(SCOPES)
        & gradients["mode"].astype(str).isin(CURVE_MODE_ORDER)
        & ~(gradients.city.astype(str).eq("Bogota") & gradients.scope.astype(str).eq("A"))
    ].copy()
    map_display = maps.loc[
        maps.city.astype(str).isin(CITIES)
        & maps.scope.astype(str).isin(SCOPES)
    ].copy()
    gd_vals = finite_values(
        gd, ("gradient", "design_ci_lower", "design_ci_upper",
             "missing_bound_lower", "missing_bound_upper"),
    )
    if len(gd_vals):
        spread = max(float(np.max(gd_vals)-np.min(gd_vals)), 1.0)
        pad = max(2.0, spread*0.05)
        ymin = math.floor((float(np.min(gd_vals))-pad)/5)*5
        ymax = math.ceil((float(np.max(gd_vals))+pad)/5)*5
    else:
        ymin, ymax = -40.0, 40.0
    estimate = pd.to_numeric(gd.gradient, errors="coerce").to_numpy(float)
    blo = pd.to_numeric(gd.missing_bound_lower, errors="coerce").to_numpy(float)
    bhi = pd.to_numeric(gd.missing_bound_upper, errors="coerce").to_numpy(float)
    valid_lo = np.isfinite(estimate) & np.isfinite(blo)
    valid_hi = np.isfinite(estimate) & np.isfinite(bhi)
    technical_bounds_differ = bool(
        (valid_lo.any() and not np.allclose(estimate[valid_lo], blo[valid_lo], rtol=0, atol=1e-8))
        or (valid_hi.any() and not np.allclose(estimate[valid_hi], bhi[valid_hi], rtol=0, atol=1e-8))
    )

    record_rows("Figure4A", "gradients", gd.loc[gd.city.astype(str).eq("Sao Paulo")],
                "city == Sao Paulo; threshold == 30; mode in car/wt; scope in R/E/A")
    record_rows("Figure4B", "gradients", gd.loc[gd.city.astype(str).eq("Bogota")],
                "city == Bogota; threshold == 30; mode in car/wt; scope in R/E; A excluded")
    for pid, city, scope in (
        ("Figure4C", "Sao Paulo", "R"), ("Figure4D", "Bogota", "R"),
        ("Figure4E", "Sao Paulo", "E"), ("Figure4F", "Bogota", "E"),
        ("Figure4G", "Sao Paulo", "A"),
    ):
        d = map_display.loc[map_display.city.astype(str).eq(city) & map_display.scope.astype(str).eq(scope)]
        record_rows(pid, "maps", d, f"city == {city}; scope == {scope}; preserve every mapped origin row")
    record_rows("Figure4H", "maps", map_display, "all displayed map rows used for paired-origin n notes")
    record_rows("Figure4H", "gradients", gd, "all displayed 30-min gradient rows used for technical-bound note")

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(
        4, 2, height_ratios=(1.0, 1.0, 1.0, 1.35),
        left=0.12, right=0.98, bottom=0.07, top=0.78,
        hspace=0.38, wspace=0.30,
    )
    axes0 = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for letter, ax in zip(("A", "B"), axes0):
        pos = ax.get_position()
        add_tag(fig, ax, letter, x=pos.x0+0.025, y=pos.y1+0.008)
    for ci, city in enumerate(CITIES):
        ax = axes0[ci]
        sub = gd.loc[gd.city.astype(str).eq(city)]
        if sub.empty:
            continue
        scope_positions = {scope: i for i, scope in enumerate(SCOPES)}
        for mode, offset, color in (("car", -0.10, PALETTE["car"]), ("wt", 0.10, PALETTE["wt"])):
            dm = sub.loc[sub["mode"].astype(str).eq(mode)]
            for _, row in dm.iterrows():
                scope = str(row.scope)
                if scope not in scope_positions:
                    continue
                x = scope_positions[scope] + offset
                value = float(row.gradient)
                lo = pd.to_numeric(pd.Series([row.get("design_ci_lower")]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([row.get("design_ci_upper")]), errors="coerce").iloc[0]
                bound_lo = pd.to_numeric(pd.Series([row.get("missing_bound_lower")]), errors="coerce").iloc[0]
                bound_hi = pd.to_numeric(pd.Series([row.get("missing_bound_upper")]), errors="coerce").iloc[0]
                if technical_bounds_differ and np.isfinite(bound_lo) and np.isfinite(bound_hi) and (
                    abs(value-bound_lo) > 1e-8 or abs(value-bound_hi) > 1e-8
                ):
                    ax.vlines(x, bound_lo, bound_hi, color=color, alpha=0.3, lw=3.2, zorder=1)
                if city == "Sao Paulo" and np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar(x, value, yerr=[[max(0, value-lo)], [max(0, hi-value)]],
                                fmt="o", ms=4.0, color=color, capsize=2.0, lw=0.9, zorder=2)
                else:
                    ax.plot(x, value, "o", ms=4.0, color=color, zorder=2)
        ax.axhline(0, color="#777777", lw=0.8, linestyle="--")
        if city == "Sao Paulo":
            tick_labels = ["Registry", "Equipment-\nSUS", "Historical\nactivity flag"]
            ax.set_xticks(range(3), tick_labels)
            ax.set_xlim(-0.45, 2.45)
        else:
            ax.set_xticks((0, 1), ["Registry", "Public-provider\nsignal"])
            ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(ymin, ymax)
        ax.set_title(city_label(city), loc="center", fontsize=11, fontweight="bold", pad=5)
        if ci == 0:
            ax.set_ylabel("Lower − higher group\n(percentage points)", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8.5, pad=3)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.55)
        ax.set_axisbelow(True)

    grad_handles = [
        Line2D([0], [0], color=PALETTE["car"], marker="o", lw=0, ms=4, label="Car"),
        Line2D([0], [0], color=PALETTE["wt"], marker="o", lw=0, ms=4, label="WT"),
    ]
    if gd.city.astype(str).eq("Sao Paulo").any():
        grad_handles.append(
            Line2D([0], [0], color="#555555", lw=0.8, marker="|", ms=5,
                   label="95% design CI at 30 min (SP only)")
        )
    if technical_bounds_differ:
        grad_handles.append(Patch(facecolor="#888888", alpha=0.3, label="Technical bound (not CI)"))
    fig.legend(
        handles=grad_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=4, fontsize=8.5, handlelength=1.4, handletextpad=0.4,
        columnspacing=0.75, labelspacing=0.3, borderaxespad=0.0,
    )
    map_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5,
               label=label)
        for _, label, color in MAP_CLASSES
    ]
    fig.legend(
        handles=map_handles, loc="upper center", bbox_to_anchor=(0.5, 0.895),
        ncol=3, fontsize=8.5, handlelength=0.8, handletextpad=0.35,
        columnspacing=0.7, labelspacing=0.3, borderaxespad=0.0,
    )

    map_panel_specs = (
        (1, 0, "C", "Sao Paulo", "R"),
        (1, 1, "D", "Bogota", "R"),
        (2, 0, "E", "Sao Paulo", "E"),
        (2, 1, "F", "Bogota", "E"),
        (3, 0, "G", "Sao Paulo", "A"),
    )
    for ri, ci, letter, city, scope in map_panel_specs:
        ax = fig.add_subplot(gs[ri, ci])
        pos = ax.get_position()
        add_tag(fig, ax, letter, x=pos.x0+0.025, y=pos.y1+0.008)
        d = map_display.loc[
            map_display.city.astype(str).eq(city)
            & map_display.scope.astype(str).eq(scope)
        ]
        if not d.empty:
            for key, _, color in MAP_CLASSES:
                pts = d.loc[d.classification.astype(str).eq(key)]
                if not pts.empty:
                    ax.scatter(
                        pd.to_numeric(pts.lon, errors="coerce"),
                        pd.to_numeric(pts.lat, errors="coerce"),
                        s=7 if len(pts) < 300 else 4.8,
                        color=color, alpha=0.76, linewidths=0,
                        rasterized=True, zorder=2,
                    )
        ax.set_aspect("equal", adjustable="datalim")
        if letter in ("F", "G"):
            ax.set_xlabel("Longitude", fontsize=10, fontweight="bold", labelpad=2)
        if ci == 0:
            ax.set_ylabel("Latitude", fontsize=10, fontweight="bold", labelpad=2)
        ax.tick_params(axis="both", labelsize=8.5, length=2.2, pad=2)
        ax.grid(color="#EEEEEE", linewidth=0.45)
        ax.set_axisbelow(True)
        full_scope = scope_label(city, scope)
        if city == "Sao Paulo" and scope == "A":
            title = "São Paulo · Retained historical\nactivity-flag inventory · 30 min"
        elif city == "Bogota" and scope == "E":
            title = "Bogotá · Public-provider signal\n30 min"
        else:
            if full_scope == "Equipment-SUS":
                title = f"{city_label(city)} · Equipment-\nSUS · 30 min"
            else:
                title = f"{city_label(city)} · {full_scope}\n30 min"
        ax.set_title(title, loc="center", fontsize=11, fontweight="bold", pad=4)

    hax = fig.add_subplot(gs[3, 1])
    hax.set_axis_off()
    hpos = hax.get_position()
    add_tag(fig, hax, "H", x=hpos.x0+0.025, y=hpos.y1+0.008)
    hax.text(0.02, 0.98, "Map classes and definitions",
             transform=hax.transAxes, ha="left", va="top",
             fontsize=11, fontweight="bold")
    map_n_sp = maps.loc[maps.city.astype(str).eq("Sao Paulo"), "uid"].nunique()
    map_n_bo = maps.loc[maps.city.astype(str).eq("Bogota"), "uid"].nunique()
    notes = [
        "30-min maps include all paired origins:",
        f"SP n={map_n_sp:,}; BO n={map_n_bo:,}.",
        "Unknown SES remains in each denominator;",
        "no unsampled area is filled.",
        "Q1−Q5: SP tract-income groups;",
        "low−high: BO residential estrato.",
        "Technical bounds coincide with estimates." if not technical_bounds_differ
        else "Thick bars show technical bounds, not CIs.",
    ]
    hax.text(0.02, 0.80, "\n".join(notes), transform=hax.transAxes,
             ha="left", va="top", fontsize=8.5, linespacing=1.05)
    fig.suptitle(
        "Socioeconomic differences and 30-min car/WT origin classification",
        x=0.045, y=0.982, ha="left", fontsize=12, fontweight="bold",
    )
    save_figure(fig, OUT, "Figure4_income_geography")


def wrap_label(value: str, width: int) -> str:
    return "\n".join(textwrap.wrap(
        value, width=width, break_long_words=False, break_on_hyphens=False,
    ))


def figure5() -> None:
    same_data = DATA["validation"]
    sensitivity = DATA["sensitivity"]
    validation_display = same_data.loc[
        same_data.city.astype(str).isin(CITIES)
        & same_data.scope.astype(str).isin(SCOPES)
        & same_data["mode"].astype(str).isin(["car", "wt"])
    ].copy()
    sensitivity_display = sensitivity.loc[
        sensitivity.city.astype(str).isin(CITIES)
        & sensitivity["mode"].astype(str).isin(["car", "wt"])
    ].copy()
    for pid, city, scope in (
        ("Figure5A", "Sao Paulo", "R"), ("Figure5B", "Bogota", "R"),
        ("Figure5C", "Sao Paulo", "E"), ("Figure5D", "Bogota", "E"),
        ("Figure5E", "Sao Paulo", "A"),
    ):
        d = validation_display.loc[
            validation_display.city.astype(str).eq(city)
            & validation_display.scope.astype(str).eq(scope)
        ]
        record_rows(pid, "validation", d, f"city == {city}; scope == {scope}; mode in car/wt")
    record_rows(
        "Figure5F", "validation", validation_display,
        "all validation period rows supplying the current same-frame period panels and legend note",
    )
    for pid, city in (("Figure5G", "Sao Paulo"), ("Figure5H", "Bogota")):
        d = sensitivity_display.loc[sensitivity_display.city.astype(str).eq(city)]
        record_rows(pid, "sensitivity", d, f"city == {city}; mode in car/wt; all current sensitivity arms")

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(
        4, 2, height_ratios=(1.1, 1.1, 1.3, 2.5),
        left=0.15, right=0.98, bottom=0.09, top=0.90,
        hspace=0.44, wspace=0.36,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]
    letter_map = {
        ("Sao Paulo", "R"): "A", ("Bogota", "R"): "B",
        ("Sao Paulo", "E"): "C", ("Bogota", "E"): "D",
        ("Sao Paulo", "A"): "E", ("Bogota", "A"): "F",
    }
    profile_colors = {"baseline": "#4D4D4D", "weekday": "#0F4D92", "weekend": "#42949E"}
    x_positions = {"baseline": 0, "weekday": 1, "weekend": 2}
    for ri, scope in enumerate(SCOPES):
        for ci, city in enumerate(CITIES):
            ax = axes[ri][ci]
            letter = letter_map[(city, scope)]
            add_tag(fig, ax, letter)
            sub = validation_display.loc[
                validation_display.city.astype(str).eq(city)
                & validation_display.scope.astype(str).eq(scope)
            ].copy()
            if city == "Bogota" and scope == "A":
                ax.set_axis_off()
                ax.text(0.02, 0.94, "Reading the panels", transform=ax.transAxes,
                        fontsize=11, fontweight="bold", va="top")
                handles = [
                    Line2D([0], [0], marker="o", color="#555555", lw=0,
                           label="Car"),
                    Line2D([0], [0], marker="s", color="#555555", lw=0,
                           label="Walking + transit (WT)"),
                    Line2D([0], [0], marker="o", color=profile_colors["baseline"], lw=0,
                           label="Baseline"),
                    Line2D([0], [0], marker="o", color=profile_colors["weekday"], lw=0,
                           label="Weekday"),
                    Line2D([0], [0], marker="o", color=profile_colors["weekend"], lw=0,
                           label="Weekend"),
                ]
                ax.legend(
                    handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.76),
                    fontsize=8.5, ncol=2, labelspacing=0.3, columnspacing=0.55,
                    borderaxespad=0, handletextpad=0.3,
                )
                nvals = validation_display[["city", "n"]].drop_duplicates()
                sp_n = int(pd.to_numeric(nvals.loc[nvals.city.astype(str).eq("Sao Paulo"), "n"], errors="coerce").dropna().iloc[0])
                bo_n = int(pd.to_numeric(nvals.loc[nvals.city.astype(str).eq("Bogota"), "n"], errors="coerce").dropna().iloc[0])
                ax.text(
                    0.02, 0.33,
                    f"Validation frames: SP n={sp_n}; BO n={bo_n}.\n"
                    "Intervals are pointwise 95% design CIs.\n"
                    "BO has no primary legacy-activity arm.",
                    transform=ax.transAxes, va="top", fontsize=8.5, linespacing=1.15,
                )
                continue
            if sub.empty:
                continue
            for mode, offset, marker in (("car", -0.08, "o"), ("wt", 0.08, "s")):
                for period in ("baseline", "weekday", "weekend"):
                    dm = sub.loc[
                        sub["mode"].astype(str).eq(mode)
                        & sub.period.astype(str).eq(period)
                    ]
                    if dm.empty:
                        continue
                    row = dm.iloc[0]
                    x = x_positions[period] + offset
                    y = float(row.coverage_lower)
                    color = profile_colors[period]
                    lo = pd.to_numeric(pd.Series([row.get("design_ci_lower")]), errors="coerce").iloc[0]
                    hi = pd.to_numeric(pd.Series([row.get("design_ci_upper")]), errors="coerce").iloc[0]
                    if np.isfinite(lo) and np.isfinite(hi):
                        ax.errorbar(
                            x, y, yerr=[[max(0, y-lo)], [max(0, hi-y)]],
                            fmt=marker, ms=4.0, color=color, capsize=2.0, lw=0.9,
                        )
                    else:
                        ax.plot(x, y, marker=marker, ms=4.0, color=color)
            city_display = city_label(city)
            short_scope = {
                "Sao Paulo": {"R": "Registry", "E": "Equipment-SUS", "A": "Legacy activity"},
                "Bogota": {"R": "Registry", "E": "Public-provider signal"},
            }[city][scope]
            ax.set_title(f"{city_display} · {short_scope}", loc="left",
                         fontsize=11, fontweight="bold", pad=5)
            ax.set_xticks([0, 1, 2], ["Baseline", "Weekday", "Weekend"])
            ax.set_ylim(-2, 102)
            ax.set_yticks([0, 50, 100])
            if ci == 0:
                ax.set_ylabel("Covered at 30 min (%)", fontsize=10, fontweight="bold")
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=8.5, pad=2)

    # The sensitivity panels use a dedicated label strip and a separate plot
    # strip within each panel, preventing long São Paulo arm labels from entering
    # the adjacent Bogotá panel. Line breaks preserve complete category names.
    base = fig.add_subplot(gs[3, :])
    pos = base.get_position()
    base.remove()
    gap = 0.035
    total_w = pos.width
    panel_g_w = (total_w-gap)*0.68
    panel_h_w = (total_w-gap)-panel_g_w
    panel_g_left = pos.x0
    panel_h_left = pos.x0 + panel_g_w + gap
    label_g_w = panel_g_w*0.59
    label_h_w = panel_h_w*0.52
    tag_y = pos.y1 + 0.012
    vertical_pad = 0.015
    axes_height = pos.height - 0.035

    for letter, city, left, panel_w, label_w, title in (
        ("G", "Sao Paulo", panel_g_left, panel_g_w, label_g_w,
         "São Paulo · destination-set scenarios"),
        ("H", "Bogota", panel_h_left, panel_h_w, label_h_w,
         "Bogotá · 142/158-site frames"),
    ):
        sub = sensitivity_display.loc[sensitivity_display.city.astype(str).eq(city)].copy()
        labels = sorted(sub.arm_label.astype(str).unique())
        label_positions = {name: i for i, name in enumerate(labels)}
        plot_left = left + label_w
        plot_w = panel_w - label_w - 0.015
        label_ax = fig.add_axes([left, pos.y0+vertical_pad, label_w-0.025, axes_height], frameon=False)
        plot_ax = fig.add_axes([plot_left, pos.y0+vertical_pad, plot_w, axes_height])
        add_tag(fig, plot_ax, letter, x=left, y=tag_y)
        plot_ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=5)
        label_ax.set_ylim(-0.5, max(len(labels)-0.5, 0.5))
        plot_ax.set_ylim(-0.5, max(len(labels)-0.5, 0.5))
        label_ax.set_xlim(0, 1)
        label_ax.set_axis_off()
        if city == "Bogota":
            display_labels = {
                "Registry-linked primary (142)": "Registry-linked\nprimary (142)",
                "Public-provider signal · primary (16)": "Public-provider\nsignal · primary\n(16)",
                "Public-provider signal · historical (16)": "Public-provider\nsignal · historical\n(16)",
                "Historical capacity (158)": "Historical\ncapacity (158)",
            }
        else:
            display_labels = {name: name for name in labels}
        for name, yi in label_positions.items():
            label_ax.text(
                1.0, yi, display_labels.get(name, name),
                ha="right", va="center", fontsize=8.5, fontweight="bold",
                linespacing=0.95, clip_on=False,
            )
        for mode, offset, color in (("car", -0.12, PALETTE["car"]), ("wt", 0.12, PALETTE["wt"])):
            dm = sub.loc[sub["mode"].astype(str).eq(mode)]
            for _, row in dm.iterrows():
                yi = label_positions[str(row.arm_label)] + offset
                xval = float(row.coverage_lower)
                lo = pd.to_numeric(pd.Series([row.get("design_ci_lower")]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([row.get("design_ci_upper")]), errors="coerce").iloc[0]
                if np.isfinite(lo) and np.isfinite(hi):
                    plot_ax.errorbar(
                        xval, yi, xerr=[[max(0, xval-lo)], [max(0, hi-xval)]],
                        fmt="o", ms=3.4, color=color, capsize=1.7, lw=0.8,
                    )
                else:
                    plot_ax.plot(xval, yi, "o", ms=3.4, color=color)
        plot_ax.set_xlim(-2, 102)
        plot_ax.set_xticks([0, 50, 100])
        plot_ax.set_yticks([])
        plot_ax.tick_params(axis="x", labelsize=8.5, pad=2)
        plot_ax.grid(axis="x", color="#E8E8E8", linewidth=0.55)
        plot_ax.set_axisbelow(True)
        for spine in ("left", "right", "top"):
            plot_ax.spines[spine].set_visible(False)

    fig.text(
        0.5, 0.035, "30-min modeled lower coverage (%)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
    fig.suptitle(
        "Departure-time and destination-set sensitivity",
        x=0.045, y=0.982, ha="left", fontsize=12, fontweight="bold",
    )
    save_figure(fig, OUT, "Figure5_validation_sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures", nargs="*", type=int, choices=(1, 2, 3, 4, 5),
        default=[1, 2, 3, 4, 5],
        help="Figure numbers to render. The final run renders all five.",
    )
    parser.add_argument("--run", choices=("run01", "run02"), default="run01")
    return parser.parse_args()


def main() -> None:
    global OUT
    args = parse_args()
    selected = set(args.figures or [1, 2, 3, 4, 5])
    OUT = X / "figures_main" / args.run
    OUT.mkdir(parents=True, exist_ok=True)
    load_sources()
    if 1 in selected:
        figure1()
    if 2 in selected:
        figure2()
    if 3 in selected:
        figure3()
    if 4 in selected:
        figure4()
    if 5 in selected:
        figure5()
    write_contract(selected)
    audit = {
        "rendered_figures": sorted(selected),
        "output_directory": str(OUT),
        "panel_source_selection_count": sum(len(v) for v in PANEL_ROWS.values()),
        "panel_ids_with_source_rows": sorted(PANEL_ROWS),
        "requested_output_formats": ["PDF editable text", "SVG editable text", "PNG 300 dpi", "TIFF 600 dpi LZW"],
        "layout_notes": {
            "Figure4": "Map class key is in a dedicated figure legend row; panel H retains the full map-origin and SES notes.",
            "Figure5": "Panels G/H have separate label and plotting strips; long labels cannot intrude into the adjacent panel.",
        },
    }
    (OUT / "layout_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
