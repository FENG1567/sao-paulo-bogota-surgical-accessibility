#!/usr/bin/env python3
"""Layout-only redraw of frozen supplemental figures S1–S3.

The plotted estimates and intervals are read directly from the five frozen
supplement source CSVs. This script does not recompute scientific estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import os

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import figure_style as style

# figure_style.save_figure writes .pdf, editable .svg, .png, and LZW .tiff (dpi=300, dpi=600).
EXPORT_FORMATS = (".pdf", ".svg", ".png", ".tiff")
RASTER_DPI = (300, 600)


X = Path(__file__).resolve().parent
SOURCE_DIR = Path(os.environ.get('FROZEN_SOURCE_DATA',str(X/'Source_Data'))) / 'supplement_figures'
OUT = X / "figures_supp" / "run01"
SOURCE_DIR = Path(os.environ.get('FROZEN_SOURCE_DATA',str(SOURCE_DIR.parent))) / 'supplement_figures'
CONTRACT_PATH = X / "supp_data_contract.json"
AUDIT_PATH = X / "layout_audit.json"

S1_FILE = "FigureS1_car_time_sensitivity_source.csv"
S2A_FILE = "FigureS2a_activity_expansion_source.csv"
S2B_FILE = "FigureS2b_regional_sensitivity_source.csv"
S2C_FILE = "FigureS2c_headway_sensitivity_source.csv"
S3_FILE = "FigureS3_bogota_classification_source.csv"

SCENARIO_ORDER = (
    "main_factor_1_over_0_6",
    "free_flow_factor_1_0",
    "intermediate_factor_1_over_0_8",
    "reconstructed_base_plus_5_min",
    "main_adjusted_plus_5_min",
)
SCENARIO_LABELS = {
    "main_factor_1_over_0_6": "Main car time\n(native ÷ 0.60)",
    "free_flow_factor_1_0": "Native road time",
    "intermediate_factor_1_over_0_8": "Native road time\n(÷ 0.80)",
    "reconstructed_base_plus_5_min": "Native road time + 5 min",
    "main_adjusted_plus_5_min": "Main car time + 5 min",
}
SCENARIO_COLORS = {
    "main_factor_1_over_0_6": "#0072B2",
    "free_flow_factor_1_0": "#D55E00",
    "intermediate_factor_1_over_0_8": "#009E73",
    "reconstructed_base_plus_5_min": "#CC79A7",
    "main_adjusted_plus_5_min": "#E69F00",
}
THRESHOLDS = (15, 30, 45)
THRESHOLD_COLORS = {15: "#0072B2", 30: "#009E73", 45: "#D55E00"}
CITY_ORDER = ("Sao Paulo", "Bogota")
SCOPE_ORDER = ("R", "E", "A")
CLASSIFICATION_ORDER = (
    "legacy_original30_frozen", "mean_30_epsg4326", "mean_50_epsg4326",
    "mean_80_epsg4326", "dominant_area_epsg4326", "mean_30_epsg3116",
    "mean_50_epsg3116", "mean_80_epsg3116",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(name: str) -> pd.DataFrame:
    path = SOURCE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen source CSV: {path}")
    df = pd.read_csv(path)
    df.insert(0, "_source_row_0", np.arange(len(df), dtype=int))
    return df


def selected(df: pd.DataFrame, predicate: pd.Series, label: str) -> pd.DataFrame:
    out = df.loc[predicate].copy()
    if out.empty:
        raise ValueError(f"Selection for {label} returned no rows.")
    return out


def row_ids(df: pd.DataFrame) -> list[dict[str, int]]:
    return [{"data_row_index_0based": int(i), "csv_line_1based": int(i) + 2}
            for i in df["_source_row_0"].tolist()]


def finite_values(df: pd.DataFrame, columns: list[str], label: str) -> None:
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: {col} contains missing or non-finite values.")


def check_intervals(df: pd.DataFrame, point: str, lo: str, hi: str,
                    label: str, allow_missing: bool = False) -> None:
    p = pd.to_numeric(df[point], errors="coerce")
    low = pd.to_numeric(df[lo], errors="coerce")
    high = pd.to_numeric(df[hi], errors="coerce")
    both_missing = low.isna() & high.isna()
    if (low.isna() ^ high.isna()).any() or (both_missing.any() and not allow_missing):
        raise ValueError(f"{label}: incomplete interval endpoints.")
    keep = ~both_missing
    if ((low[keep] > high[keep]).any() or
            (p[keep] < low[keep] - 1e-7).any() or
            (p[keep] > high[keep] + 1e-7).any()):
        raise ValueError(f"{label}: interval bounds do not contain the point estimate.")


def text_wrap(value: str, width: int) -> str:
    return textwrap.fill(str(value), width=width, break_long_words=False,
                         break_on_hyphens=False)


def numeric_artist_fingerprint(fig) -> str:
    values = []
    for ax in fig.axes:
        for line in ax.lines:
            values.append(("line", np.asarray(line.get_xdata()).tolist(),
                           np.asarray(line.get_ydata()).tolist()))
        for collection in ax.collections:
            if hasattr(collection, "get_offsets"):
                values.append(("offsets", np.asarray(collection.get_offsets()).tolist()))
            if hasattr(collection, "get_segments"):
                values.append(("segments", [np.asarray(v).tolist()
                                             for v in collection.get_segments()]))
            if hasattr(collection, "get_paths"):
                values.append(("collection_paths", [np.asarray(p.vertices).tolist()
                                                      for p in collection.get_paths()]))
        for patch in ax.patches:
            if hasattr(patch, "get_width"):
                values.append(("bar", patch.get_x(),
                               patch.get_y() + patch.get_height() / 2,
                               patch.get_width()))
            elif hasattr(patch, "get_path"):
                path = patch.get_patch_transform().transform_path(patch.get_path())
                values.append(("patch_path", np.asarray(path.vertices).tolist()))
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()



def compact_reader_footer(fig, stem):
    """Remove caption-owned note text; compact layouts are sized directly."""
    before = numeric_artist_fingerprint(fig)
    starts = (
        "WT allows transit with walking",
        "Colored circles mark paired WT-coverage",
        "Residential estrato:",
    )
    for item in list(fig.texts):
        if item.get_text().startswith(starts):
            item.remove()
    if before != numeric_artist_fingerprint(fig):
        raise AssertionError("Supplement data changed during footer cleanup")


def save_with_artist_hash(fig, stem: str, audit: dict) -> None:
    compact_reader_footer(fig,stem)
    fig.canvas.draw()
    before = numeric_artist_fingerprint(fig)
    style.save_figure(fig, OUT, stem)
    fig.canvas.draw()
    after = numeric_artist_fingerprint(fig)
    if before != after:
        raise AssertionError(f"{stem}: numeric plot artists changed during export.")
    audit.setdefault("numeric_artist_hashes", {})[stem] = {
        "before_sha256": before,
        "after_sha256": after,
        "unchanged": True,
    }


def register_selection(contract: dict, key: str, df: pd.DataFrame,
                       predicate: str, panels: list[str]) -> None:
    contract["selections"][key] = {
        "predicate": predicate,
        "selected_rows": int(len(df)),
        "source_rows": row_ids(df),
        "panel_identity": panels,
    }


def plot_s1(df: pd.DataFrame, audit: dict) -> None:
    scenarios = set(df.scenario.astype(str))
    if scenarios != set(SCENARIO_ORDER):
        raise ValueError("S1 does not contain exactly the five frozen car-time inputs.")
    thresholds = tuple(sorted(pd.to_numeric(df.threshold_min).astype(int).unique()))
    if thresholds != tuple(range(10, 61, 5)):
        raise ValueError("S1 does not contain every 10–60 minute threshold.")
    counts = {city: int(df.loc[df.city.eq(city), "n_origins"].iloc[0])
              for city in CITY_ORDER}
    if counts != {"Sao Paulo": 1100, "Bogota": 1057}:
        raise ValueError(f"S1 origin-frame counts changed: {counts}")

    fig = plt.figure(figsize=(7.2, 3.65))
    fig.text(0.055, 0.962, "Registry inventories: car-time sensitivity",
             ha="left", va="top", fontsize=10.5, fontweight="bold")
    ax_positions = ((0.12, 0.31, 0.36, 0.39),
                    (0.60, 0.31, 0.36, 0.39))
    axes = [fig.add_axes(pos) for pos in ax_positions]
    y_values: list[float] = []
    plot_data = []
    for city, ax, letter in zip(CITY_ORDER, axes, ("A", "B")):
        city_df = df.loc[df.city.astype(str).eq(city)]
        n = int(city_df.n_origins.iloc[0])
        if city == "Sao Paulo":
            city_label = "São Paulo"
            frame_label = "stratified probability sample"
        else:
            city_label = "Bogotá"
            frame_label = "complete-frame census"
        pos = ax.get_position()
        header_x = 0.12 if city == "Sao Paulo" else 0.62
        tag_x = 0.035 if city == "Sao Paulo" else 0.515
        style.panel_tag(fig, ax, letter, x=tag_x, y=0.775)
        short_frame = ("stratified probability sample" if city == "Sao Paulo"
                       else "complete-frame census")
        fig.text(header_x, 0.785,
                 f"{city_label} | n = {n:,}\n{short_frame}",
                 ha="left", va="bottom", fontsize=9.5, linespacing=1.0,
                 fontweight="bold")
        ax.axhline(0, color="#555555", linewidth=0.7,
                   linestyle=(0, (2, 2)), zorder=0)
        ax.grid(axis="y", color="#D9DDE1", linewidth=0.5)
        ax.set_xlim(9, 61)
        ax.set_xticks(thresholds)
        ax.set_xlabel("Time threshold (min)", fontsize=10.5, fontweight="bold")
        ax.tick_params(length=2.5, pad=3, labelsize=9.5)
        series = []
        for scenario in SCENARIO_ORDER:
            part = city_df.loc[city_df.scenario.astype(str).eq(scenario)].sort_values(
                "threshold_min", kind="stable")
            if len(part) != len(thresholds):
                raise ValueError(f"S1 {city}/{scenario} curve is incomplete.")
            x = pd.to_numeric(part.threshold_min).to_numpy(float)
            y = pd.to_numeric(part.car_minus_wt_pp).to_numpy(float)
            lo_t = pd.to_numeric(part.technical_delta_lower_pp).to_numpy(float)
            hi_t = pd.to_numeric(part.technical_delta_upper_pp).to_numpy(float)
            color = SCENARIO_COLORS[scenario]
            ax.plot(x, y, color=color, marker="o", markersize=3.1,
                    linewidth=1.25, zorder=3)
            y_values.extend(y.tolist())
            y_values.extend(lo_t.tolist())
            y_values.extend(hi_t.tolist())
            if city == "Sao Paulo":
                lo = pd.to_numeric(part.design_ci_lower_pp).to_numpy(float)
                hi = pd.to_numeric(part.design_ci_upper_pp).to_numpy(float)
                if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
                    raise ValueError("S1 São Paulo design confidence intervals are incomplete.")
                ax.fill_between(x, lo, hi, color=color, alpha=0.12,
                                linewidth=0, zorder=1)
                y_values.extend(lo.tolist())
                y_values.extend(hi.tolist())
            else:
                lo = pd.to_numeric(part.design_ci_lower_pp).to_numpy(float)
                hi = pd.to_numeric(part.design_ci_upper_pp).to_numpy(float)
                if not (np.isnan(lo).all() and np.isnan(hi).all()):
                    raise ValueError("S1 Bogotá census must not show a sampling CI.")
            # These technical bounds remain distinct from the design CI.
            for xx, low, high in zip(x, lo_t, hi_t):
                if high - low > 1e-7:
                    ax.vlines(xx, low, high, color="#666666", linewidth=0.8,
                              alpha=0.8, zorder=2)
            series.append({
                "scenario": scenario,
                "source_rows": row_ids(part),
                "threshold_min": x.tolist(),
                "estimate_car_minus_wt_pp": y.tolist(),
                "design_ci_lower_pp": lo.tolist(),
                "design_ci_upper_pp": hi.tolist(),
                "technical_bound_lower_pp": lo_t.tolist(),
                "technical_bound_upper_pp": hi_t.tolist(),
            })
        plot_data.append({"panel": letter, "city": city, "n_origins": n,
                          "sampling_frame": frame_label, "series": series})
    low, high = min(y_values), max(y_values)
    margin = max((high - low) * 0.045, 0.6)
    for ax in axes:
        ax.set_ylim(low - margin, high + margin)
    axes[0].set_ylabel("Car − WT lower coverage\n(percentage points)",
                       fontsize=9.5, fontweight="bold", rotation=90,
                       labelpad=18)

    handles = [Line2D([0], [0], color=SCENARIO_COLORS[s], marker="o",
                      linewidth=1.25, markersize=3.5,
                      label=SCENARIO_LABELS[s]) for s in SCENARIO_ORDER]
    handles.append(Patch(facecolor="#888888", alpha=0.18, edgecolor="none",
                         label="95% design CI (São Paulo only)"))
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.54, 0.205), frameon=False,
               columnspacing=0.7, handlelength=1.4, labelspacing=0.35,
               handletextpad=0.55, borderaxespad=0, fontsize=9.5,
               alignment="left")
    s1_note = textwrap.fill(
        "WT allows transit with walking and separately queried direct walking. Shading shows the "
        "95% design CI for São Paulo; Bogotá is a complete-frame census.",
        width=100, break_long_words=False)
    fig.text(0.055, 0.08, s1_note,
             ha="left", va="bottom", fontsize=9.5, linespacing=1.05)
    audit["plots"]["S1"] = plot_data
    save_with_artist_hash(fig, "FigureS1_car_time_sensitivity", audit)


def draw_forest_panel(ax, df: pd.DataFrame, group_col: str, group_keys: list[str],
                      labels: dict[str, str], estimate: str, ci_low: str,
                      ci_high: str, tech_low: str, tech_high: str,
                      panel_name: str, audit_rows: list[dict]) -> None:
    offsets = {15: -0.19, 30: 0.0, 45: 0.19}
    centers = np.arange(len(group_keys), dtype=float)[::-1]
    for center, key in zip(centers, group_keys):
        group = df.loc[df[group_col].astype(str).eq(key)]
        for _, row in group.sort_values("threshold_min", kind="stable").iterrows():
            threshold = int(row.threshold_min)
            y = center + offsets[threshold]
            tlo, thi = float(row[tech_low]), float(row[tech_high])
            # Technical and design intervals are separately offset and styled.
            ax.hlines(y + 0.055, tlo, thi, color="#777777", linewidth=1.9,
                      alpha=0.85, zorder=2)
            cap = 0.045
            ax.vlines([tlo, thi], y + 0.055 - cap, y + 0.055 + cap,
                      color="#777777", linewidth=1.0, alpha=0.85, zorder=2)
            clo, chi = row[ci_low], row[ci_high]
            if pd.notna(clo) and pd.notna(chi):
                ax.hlines(y - 0.055, float(clo), float(chi), color="#222222",
                          linewidth=0.85, alpha=0.98, zorder=2)
                ax.vlines([float(clo), float(chi)], y - 0.055 - cap,
                          y - 0.055 + cap, color="#222222", linewidth=0.65,
                          alpha=0.98, zorder=2)
            point = float(row[estimate])
            ax.scatter(point, y, s=25, marker="o", color=THRESHOLD_COLORS[threshold],
                       edgecolor="white", linewidth=0.45, zorder=4)
            audit_rows.append({
                "source_row_index_0based": int(row["_source_row_0"]),
                "csv_line_1based": int(row["_source_row_0"]) + 2,
                "category": key,
                "threshold_min": threshold,
                "estimate_pp": point,
                "design_ci_lower_pp": None if pd.isna(clo) else float(clo),
                "design_ci_upper_pp": None if pd.isna(chi) else float(chi),
                "technical_bound_lower_pp": tlo,
                "technical_bound_upper_pp": thi,
            })
    ax.axvline(0, color="#555555", linewidth=0.7,
               linestyle=(0, (2, 2)), zorder=0)
    ax.grid(axis="x", color="#D9DDE1", linewidth=0.5)
    ax.set_yticks(centers)
    ax.set_yticklabels([text_wrap(labels[k], 28) for k in group_keys],
                       fontsize=9.5, linespacing=1.0)
    ax.set_ylim(-0.55, len(group_keys) - 0.45)
    ax.tick_params(length=2.5, pad=4, labelsize=9.5)
    ax.set_xlabel("Paired coverage change (percentage points)",
                  fontsize=10.5, fontweight="bold")
    finite = []
    for col in (estimate, ci_low, ci_high, tech_low, tech_high):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        finite.extend(vals[np.isfinite(vals)].tolist())
    if not finite:
        raise ValueError(f"{panel_name} has no plotted numerical values.")
    audit_rows.sort(key=lambda r: (group_keys.index(str(r["category"])),
                                  int(r["threshold_min"])))


def prepare_s2(activity: pd.DataFrame, regional: pd.DataFrame,
               headway: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, ...]:
    a = selected(activity,
        activity["mode"].astype(str).eq("wt")
        & pd.to_numeric(activity.threshold_min).isin(THRESHOLDS)
        & activity.activity_set_pair.astype(str).isin(("156->161", "156->162", "161->162")),
        "S2a activity expansion")
    b = selected(regional,
        regional["mode"].astype(str).eq("wt")
        & regional.group.astype(str).eq("overall")
        & pd.to_numeric(regional.threshold_min).isin(THRESHOLDS)
        & regional.scope.astype(str).isin(SCOPE_ORDER),
        "S2b regional sensitivity")
    c = selected(headway,
        headway["mode"].astype(str).eq("wt")
        & headway.group.astype(str).eq("overall")
        & pd.to_numeric(headway.threshold_min).isin(THRESHOLDS)
        & headway.scope.astype(str).isin(SCOPE_ORDER),
        "S2c headway sensitivity")
    if len(a) != 9 or len(b) != 9 or len(c) != 9:
        raise ValueError(f"S2 source selections must each contain 9 rows; got {len(a)}, {len(b)}, {len(c)}.")
    finite_values(a, ["gain_estimate_pp", "technical_bound_lower_pp",
                      "technical_bound_upper_pp"], "S2a")
    finite_values(b, ["paired_change_pp", "technical_missing_bound_lower_pp",
                      "technical_missing_bound_upper_pp"], "S2b")
    finite_values(c, ["headway_change_pp", "headway_technical_lower_pp",
                      "headway_technical_upper_pp"], "S2c")
    check_intervals(a, "gain_estimate_pp", "technical_bound_lower_pp",
                    "technical_bound_upper_pp", "S2a technical bounds")
    check_intervals(a, "gain_estimate_pp", "design_ci95_lower_pp",
                    "design_ci95_upper_pp", "S2a design CIs", allow_missing=True)
    check_intervals(b, "paired_change_pp", "technical_missing_bound_lower_pp",
                    "technical_missing_bound_upper_pp", "S2b technical bounds")
    check_intervals(b, "paired_change_pp", "paired_design_ci_lower_pp",
                    "paired_design_ci_upper_pp", "S2b design CIs", allow_missing=True)
    check_intervals(c, "headway_change_pp", "headway_technical_lower_pp",
                    "headway_technical_upper_pp", "S2c technical bounds")
    check_intervals(c, "headway_change_pp", "headway_ci_lower_pp",
                    "headway_ci_upper_pp", "S2c design CIs", allow_missing=True)
    register_selection(contract, "S2a_activity", a,
        "mode=wt; threshold_min in {15,30,45}; activity_set_pair in "
        "{156->161,156->162,161->162}", ["S2-A"])
    register_selection(contract, "S2b_regional", b,
        "mode=wt; group=overall; threshold_min in {15,30,45}; scope in {R,E,A}",
        ["S2-B"])
    register_selection(contract, "S2c_headway", c,
        "mode=wt; group=overall; threshold_min in {15,30,45}; scope in {R,E,A}",
        ["S2-C"])
    return a, b, c


def plot_s2(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame,
            audit: dict) -> None:
    fig = plt.figure(figsize=(7.2, 5.5))
    fig.text(0.055, 0.972,
             "São Paulo: paired walking-plus-transit coverage sensitivity",
             ha="left", va="top", fontsize=10.5, fontweight="bold")
    panel_defs = [
        ("A", a, "activity_set_pair", ["156->161", "156->162", "161->162"],
         {"156->161": "156 → 161 facilities",
          "156->162": "156 → 162 facilities",
          "161->162": "161 → 162 facilities"},
         "gain_estimate_pp", "design_ci95_lower_pp", "design_ci95_upper_pp",
         "technical_bound_lower_pp", "technical_bound_upper_pp",
         "Annual surgical-activity inventory · São Paulo main frame, n = 1,100",
         0.665, "S2-A"),
        ("B", b, "scope", list(SCOPE_ORDER),
         {"R": "Registry", "E": "Equipment-SUS",
          "A": "Historical activity inventory"},
         "paired_change_pp", "paired_design_ci_lower_pp", "paired_design_ci_upper_pp",
         "technical_missing_bound_lower_pp", "technical_missing_bound_upper_pp",
         "Regional transit network vs baseline · São Paulo validation frame, n = 220",
         0.425, "S2-B"),
        ("C", c, "scope", list(SCOPE_ORDER),
         {"R": "Registry", "E": "Public administration", "A": "Annual claims"},
         "headway_change_pp", "headway_ci_lower_pp", "headway_ci_upper_pp",
         "headway_technical_lower_pp", "headway_technical_upper_pp",
         "Nominal half-headway scenario vs baseline · São Paulo validation frame, n = 220",
         0.185, "S2-C"),
    ]
    ax_positions = ((0.405, 0.70, 0.575, 0.15),
                    (0.405, 0.45, 0.575, 0.15),
                    (0.405, 0.20, 0.575, 0.15))
    axes = [fig.add_axes(pos) for pos in ax_positions]
    plotted = []
    scale_values: dict[str, list[float]] = {"A": [], "B/C": []}
    for panel, ax, definition, pos in zip(("A", "B", "C"), axes,
                                           panel_defs, ax_positions):
        (letter, data, group_col, keys, labels, estimate, ci_low, ci_high,
         tech_low, tech_high, title, _, panel_name) = definition
        tag_y = pos[1] + pos[3] + 0.014
        style.panel_tag(fig, ax, letter, x=0.02, y=tag_y)
        fig.text(0.095, tag_y + 0.004, title, ha="left", va="bottom",
                 fontsize=10.5, fontweight="bold")
        ax.set_ylim(-0.55, len(keys) - 0.45)
        rows: list[dict] = []
        draw_forest_panel(ax, data, group_col, keys, labels, estimate,
                          ci_low, ci_high, tech_low, tech_high, panel_name, rows)
        if panel != "C":
            ax.set_xlabel("")
        ax.grid(axis="x", color="#D9DDE1", linewidth=0.5)
        for col in (estimate, ci_low, ci_high, tech_low, tech_high):
            values = pd.to_numeric(data[col], errors="coerce").to_numpy(float)
            scale_key = "A" if panel == "A" else "B/C"
            scale_values[scale_key].extend(values[np.isfinite(values)].tolist())
        plotted.append({"panel": f"S2-{panel}", "title": title,
                        "rows": rows})
    def padded_xlim(values: list[float], fraction: float) -> tuple[float, float]:
        xlo, xhi = min(0.0, min(values)), max(0.0, max(values))
        margin = fraction * max(xhi - xlo, 1.0)
        return xlo - margin, xhi + margin

    axes[0].set_xlim(*padded_xlim(scale_values["A"], 0.10))
    shared_xlim = padded_xlim(scale_values["B/C"], 0.07)
    axes[1].set_xlim(*shared_xlim)
    axes[2].set_xlim(*shared_xlim)

    handles = [
        Line2D([0], [0], color="#222222", linewidth=0.9,
               label="95% design CI"),
        Line2D([0], [0], color="#777777", linewidth=1.9,
               label="Technical bounds for unresolved route outcomes"),
        *[Line2D([0], [0], color=THRESHOLD_COLORS[t], marker="o",
                 linewidth=0, markersize=4, label=f"{t} min") for t in THRESHOLDS],
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.56, 0.115), frameon=False,
               columnspacing=1.5, handlelength=1.6, labelspacing=0.45,
               handletextpad=0.55, borderaxespad=0, fontsize=9.5,
               alignment="left")
    s2_note = textwrap.fill(
        "Colored circles mark paired WT-coverage changes (percentage points) at 15, 30 and 45 min. "
        "Black lines are 95% design CIs; grey lines are technical bounds. WT = transit + walking. "
        "Headway is the time between scheduled services.",
        width=100, break_long_words=False)
    s2_note = s2_note.replace("S2a", "S2A").replace("S2b", "S2B").replace("S2c", "S2C")
    fig.text(0.055, 0.018, s2_note,
             ha="left", va="bottom", fontsize=9.5, linespacing=1.05)
    audit["plots"]["S2"] = plotted
    save_with_artist_hash(fig, "FigureS2_paired_sensitivity", audit)


def prepare_s3(df: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = selected(df,
        df.panel.astype(str).eq("all_classification_schemes")
        & df.scenario.astype(str).isin(CLASSIFICATION_ORDER)
        & df.scope.astype(str).isin(("R", "E")),
        "S3 classification schemes")
    b = selected(df,
        df.panel.astype(str).eq("facility_frame_sensitivity")
        & df.grouping.astype(str).eq("mean_30_epsg4326")
        & df.facility_frame.astype(str).isin(("primary142", "historical158"))
        & df.scope.astype(str).isin(("R", "E")),
        "S3 facility frames")
    if len(a) != 16 or len(b) != 4:
        raise ValueError(f"S3 must contain 16 scheme rows and 4 frame rows; got {len(a)} and {len(b)}.")
    for scenario in CLASSIFICATION_ORDER:
        part = a.loc[a.scenario.astype(str).eq(scenario)]
        if set(part.scope.astype(str)) != {"R", "E"}:
            raise ValueError(f"S3 classification {scenario} lacks its R/E pair.")
    for frame in ("primary142", "historical158"):
        part = b.loc[b.facility_frame.astype(str).eq(frame)]
        if set(part.scope.astype(str)) != {"R", "E"}:
            raise ValueError(f"S3 facility frame {frame} lacks its R/E pair.")
    finite_values(a, ["low_minus_high_pp"], "S3 classification schemes")
    finite_values(b, ["low_minus_high_pp"], "S3 facility frames")
    register_selection(contract, "S3_classification", a,
        "panel=all_classification_schemes; scenario in frozen 8-scheme order; scope in {R,E}",
        ["S3-A"])
    register_selection(contract, "S3_facility_frames", b,
        "panel=facility_frame_sensitivity; grouping=mean_30_epsg4326; "
        "facility_frame in {primary142,historical158}; scope in {R,E}",
        ["S3-B"])
    return a, b


def plot_s3(a: pd.DataFrame, b: pd.DataFrame, audit: dict) -> None:
    frame_labels = {
        "primary142": "Registry-linked sites (142)",
        "historical158": "Historical full facility inventory (158)",
    }
    scope_colors = {"R": "#4D4D4D", "E": "#0F4D92"}
    selected_values = pd.concat([a.low_minus_high_pp, b.low_minus_high_pp]).astype(float)
    lo, hi = float(selected_values.min()), float(selected_values.max())
    span = max(hi - lo, 1.0)
    xlim = (min(0.0, lo) - 0.08 * span, max(0.0, hi) + 0.08 * span)

    fig = plt.figure(figsize=(7.2, 5.2))
    fig.text(0.055, 0.972, "Bogotá estrato-group and facility-frame comparisons",
             ha="left", va="top", fontsize=10.5, fontweight="bold")
    ax_a = fig.add_axes((0.44, 0.43, 0.52, 0.42))
    ax_b = fig.add_axes((0.44, 0.20, 0.52, 0.10), sharex=ax_a)
    style.panel_tag(fig, ax_a, "A", x=0.02, y=0.875)
    fig.text(0.095, 0.88, "Classification schemes · WT 30 min",
             ha="left", va="bottom", fontsize=10.5, fontweight="bold")
    style.panel_tag(fig, ax_b, "B", x=0.02, y=0.31)
    fig.text(0.095, 0.32, "Facility-frame comparison · primary grouping, WT 30 min",
             ha="left", va="bottom", fontsize=10.5, fontweight="bold")

    centers_a = np.arange(len(CLASSIFICATION_ORDER), dtype=float)[::-1]
    offsets = {"R": 0.12, "E": -0.12}
    label_lookup = {}
    plot_a_rows = []
    for center, scenario in zip(centers_a, CLASSIFICATION_ORDER):
        group = a.loc[a.scenario.astype(str).eq(scenario)]
        for scope in ("R", "E"):
            row = group.loc[group.scope.astype(str).eq(scope)].iloc[0]
            estimate = float(row.low_minus_high_pp)
            y = center + offsets[scope]
            ax_a.scatter(estimate, y, color=scope_colors[scope], marker="o",
                         s=25, edgecolor="white", linewidth=0.45, zorder=3)
            source_label = str(row.classification_label)
            if scenario == "legacy_original30_frozen":
                display_label = "Original classification (historical comparison)"
            elif scenario == "mean_30_epsg4326":
                display_label = "Primary area-weighted mean, >30% mapped area (EPSG:4326)"
            else:
                display_label = source_label
            label_lookup[scenario] = display_label
            plot_a_rows.append({
                "source_row_index_0based": int(row["_source_row_0"]),
                "csv_line_1based": int(row["_source_row_0"]) + 2,
                "scenario": scenario, "classification_label": display_label,
                "scope": scope, "estimate_low_minus_high_pp": estimate,
            })
    ax_a.set_yticks(centers_a)
    ax_a.set_yticklabels([text_wrap(label_lookup[k], 40) for k in CLASSIFICATION_ORDER],
                         fontsize=9.5, linespacing=1.0)
    ax_a.set_ylim(-0.58, len(CLASSIFICATION_ORDER) - 0.42)
    ax_a.axvline(0, color="#666666", linewidth=0.7,
                 linestyle=(0, (2, 2)), zorder=0)
    ax_a.grid(axis="x", color="#D9DDE1", linewidth=0.5)
    ax_a.tick_params(length=2.5, pad=4, labelbottom=True, labelsize=9.5)
    ax_a.set_xlim(*xlim)

    frame_order = ("primary142", "historical158")
    centers_b = np.arange(len(frame_order), dtype=float)[::-1]
    plot_b_rows = []
    for center, frame in zip(centers_b, frame_order):
        group = b.loc[b.facility_frame.astype(str).eq(frame)]
        for scope in ("R", "E"):
            row = group.loc[group.scope.astype(str).eq(scope)].iloc[0]
            estimate = float(row.low_minus_high_pp)
            y = center + offsets[scope]
            ax_b.scatter(estimate, y, color=scope_colors[scope], marker="o",
                         s=28, edgecolor="white", linewidth=0.5, zorder=3)
            ax_b.annotate(f"{estimate:.1f}", (estimate, y), xytext=(5, 2),
                          textcoords="offset points", fontsize=9.5,
                          ha="left", va="bottom", clip_on=False)
            plot_b_rows.append({
                "source_row_index_0based": int(row["_source_row_0"]),
                "csv_line_1based": int(row["_source_row_0"]) + 2,
                "facility_frame": frame, "facility_label": frame_labels[frame],
                "scope": scope, "estimate_low_minus_high_pp": estimate,
                "numeric_annotation": f"{estimate:.1f}",
            })
    ax_b.set_yticks(centers_b)
    ax_b.set_yticklabels([text_wrap(frame_labels[k], 31) for k in frame_order],
                         fontsize=9.5, linespacing=1.0)
    ax_b.set_ylim(-0.55, len(frame_order) - 0.45)
    ax_b.axvline(0, color="#666666", linewidth=0.7,
                 linestyle=(0, (2, 2)), zorder=0)
    ax_b.grid(axis="x", color="#D9DDE1", linewidth=0.5)
    ax_b.tick_params(length=2.5, pad=4, labelbottom=True, labelsize=9.5)
    ax_b.set_xlim(*xlim)
    ax_b.set_xlabel("WT 30-min low − high coverage (percentage points)",
                    fontsize=10.5, fontweight="bold")

    handles = [
        Line2D([0], [0], color=scope_colors["R"], marker="o", linewidth=0,
               markersize=5, label="Full facility inventory"),
        Line2D([0], [0], color=scope_colors["E"], marker="o", linewidth=0,
               markersize=5, label="Linked public-provider sites"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.55, 0.075), frameon=False,
               columnspacing=2.0, handlelength=1.2, handletextpad=0.6,
               borderaxespad=0, fontsize=9.5, alignment="left")
    s3_note = textwrap.fill(
             "Residential estrato: low (0,2], middle (2,4], high (4,6]. Unknown-group origins "
             "contribute to citywide coverage totals. A uses 142 registry-linked sites. B compares "
             "the 142-site and 158-site inventories under the primary >30% area-weighted grouping. "
             "The >80% rule classifies 13 cells (2.05% of residents). "
             "EPSG codes identify coordinate reference systems; WT = transit + walking.",
             width=120, break_long_words=False)
    fig.text(0.04, 0.08, s3_note,
             ha="left", va="bottom", fontsize=9.5, linespacing=1.05)
    audit["plots"]["S3"] = [
        {"panel": "S3-A", "title": "Classification schemes · WT 30 min",
         "rows": plot_a_rows},
        {"panel": "S3-B", "title": "Facility-frame comparison · primary grouping, WT 30 min",
         "rows": plot_b_rows},
    ]
    save_with_artist_hash(fig, "FigureS3_bogota_classification", audit)


def _json_values(values) -> list[float | None]:
    return [None if pd.isna(value) else float(value) for value in values]


def build_plot_audits(s1: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame,
                      c: pd.DataFrame, s3a: pd.DataFrame,
                      s3b: pd.DataFrame) -> list[dict]:
    plots: list[dict] = []
    for letter, city in zip(("A", "B"), CITY_ORDER):
        city_df = s1.loc[s1.city.astype(str).eq(city)]
        series = []
        for scenario in SCENARIO_ORDER:
            part = city_df.loc[city_df.scenario.astype(str).eq(scenario)].sort_values(
                "threshold_min", kind="stable")
            series.append({
                "scenario": scenario,
                "source_rows": row_ids(part),
                "threshold_min": _json_values(part.threshold_min),
                "estimate_car_minus_wt_pp": _json_values(part.car_minus_wt_pp),
                "design_ci_lower_pp": _json_values(part.design_ci_lower_pp),
                "design_ci_upper_pp": _json_values(part.design_ci_upper_pp),
                "technical_bound_lower_pp": _json_values(part.technical_delta_lower_pp),
                "technical_bound_upper_pp": _json_values(part.technical_delta_upper_pp),
            })
        plots.append({
            "panel": f"S1-{letter}", "city": city,
            "n_origins": int(city_df.n_origins.iloc[0]), "series": series,
        })

    scope_labels_b = {"R": "Registry", "E": "Equipment-SUS",
                      "A": "Retained historical activity-flag inventory"}
    scope_labels_c = {"R": "Registry", "E": "Public administration", "A": "Annual claims"}
    s2_defs = [
        ("S2-A", a, "activity_set_pair", ["156->161", "156->162", "161->162"],
         {"156->161": "156 → 161 facilities",
          "156->162": "156 → 162 facilities",
          "161->162": "161 → 162 facilities"},
         "gain_estimate_pp", "design_ci95_lower_pp", "design_ci95_upper_pp",
         "technical_bound_lower_pp", "technical_bound_upper_pp"),
        ("S2-B", b, "scope", list(SCOPE_ORDER), scope_labels_b,
         "paired_change_pp", "paired_design_ci_lower_pp", "paired_design_ci_upper_pp",
         "technical_missing_bound_lower_pp", "technical_missing_bound_upper_pp"),
        ("S2-C", c, "scope", list(SCOPE_ORDER), scope_labels_c,
         "headway_change_pp", "headway_ci_lower_pp", "headway_ci_upper_pp",
         "headway_technical_lower_pp", "headway_technical_upper_pp"),
    ]
    for (panel, df, group_col, keys, labels, estimate, ci_low, ci_high,
         tech_low, tech_high) in s2_defs:
        rows = []
        for key in keys:
            part = df.loc[df[group_col].astype(str).eq(key)].sort_values(
                "threshold_min", kind="stable")
            for _, row in part.iterrows():
                rows.append({
                    "source_row_index_0based": int(row["_source_row_0"]),
                    "csv_line_1based": int(row["_source_row_0"]) + 2,
                    "category": key, "category_label": labels[key],
                    "threshold_min": int(row.threshold_min),
                    "estimate_pp": float(row[estimate]),
                    "design_ci_lower_pp": None if pd.isna(row[ci_low]) else float(row[ci_low]),
                    "design_ci_upper_pp": None if pd.isna(row[ci_high]) else float(row[ci_high]),
                    "technical_bound_lower_pp": float(row[tech_low]),
                    "technical_bound_upper_pp": float(row[tech_high]),
                })
        plots.append({"panel": panel, "rows": rows})

    rows_a = []
    for scenario in CLASSIFICATION_ORDER:
        part = s3a.loc[s3a.scenario.astype(str).eq(scenario)]
        for scope in ("R", "E"):
            row = part.loc[part.scope.astype(str).eq(scope)].iloc[0]
            rows_a.append({
                "source_row_index_0based": int(row["_source_row_0"]),
                "csv_line_1based": int(row["_source_row_0"]) + 2,
                "scenario": scenario,
                "classification_label": str(row.classification_label),
                "scope": scope,
                "estimate_low_minus_high_pp": float(row.low_minus_high_pp),
            })
    plots.append({"panel": "S3-A", "rows": rows_a})

    frame_labels = {"primary142": "Registry-linked sites (142)",
                    "historical158": "Historical full facility inventory (158)"}
    rows_b = []
    for frame in ("primary142", "historical158"):
        part = s3b.loc[s3b.facility_frame.astype(str).eq(frame)]
        for scope in ("R", "E"):
            row = part.loc[part.scope.astype(str).eq(scope)].iloc[0]
            value = float(row.low_minus_high_pp)
            rows_b.append({
                "source_row_index_0based": int(row["_source_row_0"]),
                "csv_line_1based": int(row["_source_row_0"]) + 2,
                "facility_frame": frame, "facility_label": frame_labels[frame],
                "scope": scope, "estimate_low_minus_high_pp": value,
                "numeric_annotation": f"{value:.1f}",
            })
    plots.append({"panel": "S3-B", "rows": rows_b})
    return plots


def pdf_text_bounds(path: Path) -> dict:
    import fitz

    doc = fitz.open(path)
    if doc.page_count != 1:
        raise ValueError(f"{path.name}: expected one PDF page, found {doc.page_count}.")
    page = doc[0]
    rect = page.rect
    spans = []
    outside = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                value = span.get("text", "")
                if not value.strip():
                    continue
                bbox = tuple(float(x) for x in span["bbox"])
                spans.append(value)
                if (bbox[0] < -1.0 or bbox[1] < -1.0
                        or bbox[2] > rect.width + 1.0
                        or bbox[3] > rect.height + 1.0):
                    outside.append({"text": value, "bbox_pt": bbox})
    record = {
        "page_size_pt": [float(rect.width), float(rect.height)],
        "page_size_inches": [float(rect.width) / 72.0, float(rect.height) / 72.0],
        "text_spans_checked": len(spans),
        "out_of_canvas_text_spans": outside,
        "tolerance_pt": 1.0,
    }
    doc.close()
    if outside:
        raise ValueError(f"{path.name}: text extends outside PDF canvas: {outside[:4]}")
    return record


def verify_output_styles(audit: dict) -> None:
    expected_tags = {
        "FigureS1_car_time_sensitivity": {"A", "B"},
        "FigureS2_paired_sensitivity": {"A", "B", "C"},
        "FigureS3_bogota_classification": {"A", "B"},
    }
    for stem, tags in expected_tags.items():
        if stem not in audit.get("figures", []):
            continue
        record_path = OUT / f"{stem}.style.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        text_records = record["text"]
        bad = [t for t in text_records
               if t["weight"] != "bold"
               or t["family"] not in ("Arial", ["Arial"])
               or t["size"] < 9.5]
        if bad:
            raise ValueError(f"{stem}: noncompliant text style record(s): {bad[:3]}")
        tag_records = [t for t in text_records
                       if str(t.get("gid") or "").startswith("panel_tag_")]
        observed = {t["text"] for t in tag_records}
        if observed != tags:
            raise ValueError(f"{stem}: panel tags differ: {observed} != {tags}")
        for tag in tag_records:
            if (tag["size"] != 18 or tag["weight"] != "bold"
                    or tag["family"] not in ("Arial", ["Arial"])):
                raise ValueError(f"{stem}: noncompliant panel tag: {tag}")
        outputs = {}
        for ext in ("pdf", "svg", "png", "tiff"):
            path = OUT / f"{stem}.{ext}"
            outputs[path.name] = {"bytes": path.stat().st_size,
                                  "sha256": sha256(path)}
        svg_text = (OUT / f"{stem}.svg").read_text(encoding="utf-8")
        if ("<text" not in svg_text
                or not ("font-family: Arial" in svg_text
                        or "font-family: 'Arial'" in svg_text)):
            raise ValueError(f"{stem}: SVG text is not editable Arial text.")
        pdf_bounds = pdf_text_bounds(OUT / f"{stem}.pdf")
        audit["outputs"][stem] = {
            "size_inches": record["size_inches"],
            "panel_tags": tag_records,
            "text_objects": len(text_records),
            "all_text_bold_arial_min_9_5pt": True,
            "svg_editable_text_check": "passed",
            "pdf_text_bounds": pdf_bounds,
            "files": outputs,
        }


def build_contract_and_audit() -> tuple[dict, dict]:
    names = [S1_FILE, S2A_FILE, S2B_FILE, S2C_FILE, S3_FILE]
    contract = {
        "purpose": "Layout-only relayout of the three frozen supplemental figures.",
        "source_directory": str(SOURCE_DIR),
        "backend": "Python / matplotlib",
        "plotting_rule": "Plot values are read directly from frozen precomputed CSV columns; no scientific estimates are recomputed.",
        "inputs": {},
        "selections": {},
        "panel_identity": {
            "S1-A": "São Paulo, R scope; all five car-time inputs and 10–60 min thresholds; n=1,100; stratified probability sample.",
            "S1-B": "Bogotá, R scope; all five car-time inputs and 10–60 min thresholds; n=1,057; complete-frame census, no sampling CI.",
            "S2-A": "Activity 156→161, 156→162, 161→162 paired WT estimates at 15/30/45 min; São Paulo main frame, n=1,100.",
            "S2-B": "Regional network versus baseline for Registry, Equipment-SUS, retained historical activity-flag inventory; 15/30/45 min; São Paulo validation frame, n=220.",
            "S2-C": "Nominal half-headway versus matched baseline for Registry, public administration, annual claims; 15/30/45 min; São Paulo validation frame, n=220.",
            "S3-A": "Eight Bogotá WT 30-min low-minus-high classification schemes, R and E arms; full scheme definitions retained in labels/source data.",
            "S3-B": "Corrected mean_30_epsg4326 grouping, primary142 and historical158 frames, R and E arms; point annotations retained.",
        },
    }
    for name in names:
        path = SOURCE_DIR / name
        df = pd.read_csv(path)
        contract["inputs"][name] = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "rows_excluding_header": int(len(df)),
            "columns": list(df.columns),
        }
    audit = {
        "figures": ["FigureS1_car_time_sensitivity", "FigureS2_paired_sensitivity",
                    "FigureS3_bogota_classification"],
        "render_directory": str(OUT),
        "plots": {},
        "outputs": {},
    }
    return contract, audit


def main(audit_only: bool = False, run_id: str = "run01",
         selected_figures: tuple[str, ...] = ("S1", "S2", "S3")) -> None:
    global OUT
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a directory name, such as run01 or run02.")
    OUT = X / "figures_supp" / run_id
    OUT.mkdir(parents=True, exist_ok=True)
    contract, audit = build_contract_and_audit()
    selected_figures = set(selected_figures)
    if not selected_figures.issubset({"S1", "S2", "S3"}):
        raise ValueError(f"Unknown supplemental figure selection: {selected_figures}")
    stem_by_id = {"S1": "FigureS1_car_time_sensitivity",
                  "S2": "FigureS2_paired_sensitivity",
                  "S3": "FigureS3_bogota_classification"}
    audit["figures"] = [stem_by_id[key] for key in ("S1", "S2", "S3")
                        if key in selected_figures]
    s1_raw = read_source(S1_FILE)
    s1 = selected(s1_raw,
        s1_raw.city.astype(str).isin(CITY_ORDER)
        & s1_raw.profile.astype(str).eq("baseline_fixed_departure")
        & s1_raw.scope.astype(str).eq("R")
        & s1_raw.scenario.astype(str).isin(SCENARIO_ORDER),
        "S1 city × scenario curves")
    if len(s1) != 110:
        raise ValueError(f"S1 expected 110 plotted rows; found {len(s1)}.")
    finite_values(s1, ["threshold_min", "car_minus_wt_pp", "technical_delta_lower_pp",
                       "technical_delta_upper_pp", "n_origins"], "S1")
    check_intervals(s1, "car_minus_wt_pp", "technical_delta_lower_pp",
                    "technical_delta_upper_pp", "S1 technical bounds")
    register_selection(contract, "S1_curves", s1,
        "city in {Sao Paulo,Bogota}; profile=baseline_fixed_departure; scope=R; "
        "scenario in the five frozen car-time inputs; all 10–60 min thresholds", ["S1-A", "S1-B"])
    if not audit_only and "S1" in selected_figures:
        plot_s1(s1, audit)

    activity = read_source(S2A_FILE)
    regional = read_source(S2B_FILE)
    headway = read_source(S2C_FILE)
    a, b, c = prepare_s2(activity, regional, headway, contract)
    if not audit_only and "S2" in selected_figures:
        plot_s2(a, b, c, audit)

    s3_raw = read_source(S3_FILE)
    s3_a, s3_b = prepare_s3(s3_raw, contract)
    if not audit_only and "S3" in selected_figures:
        plot_s3(s3_a, s3_b, audit)
    audit["plots"] = [row for row in build_plot_audits(s1, a, b, c, s3_a, s3_b)
                       if row["panel"][:2] in selected_figures]
    verify_output_styles(audit)

    contract["output_directory"] = str(OUT)
    contract["figure_stems"] = audit["figures"]
    contract["shared_style_helper"] = str(X / "figure_style.py")
    contract["font_files"] = [str(X / "fonts" / "arial.ttf"),
                              str(X / "fonts" / "arialbd.ttf")]
    CONTRACT_PATH.write_text(json.dumps(contract, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "audit_only": audit_only,
        "figures": audit["figures"],
        "contract": str(CONTRACT_PATH),
        "audit": str(AUDIT_PATH),
        "output_directory": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true",
                        help="Validate existing exports without rerendering.")
    parser.add_argument("--run-id", default="run01",
                        help="Figure output subdirectory, for example run01 or run02.")
    parser.add_argument("--figures", nargs="*", choices=("S1", "S2", "S3"),
                        default=("S1", "S2", "S3"),
                        help="Supplemental figure numbers to render; defaults to all three.")
    args = parser.parse_args()
    main(audit_only=args.audit_only, run_id=args.run_id,
         selected_figures=tuple(args.figures))
