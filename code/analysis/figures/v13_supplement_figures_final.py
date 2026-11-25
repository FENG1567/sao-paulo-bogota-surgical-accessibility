#!/usr/bin/env python3
"""Build compact v13 supplemental figures from completed, verified result tables.

The script never routes or substitutes pilot/self-test outputs. It writes S1
from the completed car-time sensitivity export and adds S2 only when both
headway source tables are present and pass the frozen-frame checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


THRESHOLDS = (15, 30, 45)
ALL_THRESHOLDS = tuple(range(10, 61, 5))
SCENARIO_ORDER = (
    "main_factor_1_over_0_6",
    "free_flow_factor_1_0",
    "intermediate_factor_1_over_0_8",
    "reconstructed_base_plus_5_min",
    "main_adjusted_plus_5_min",
)
SCENARIO_LABELS = {
    "main_factor_1_over_0_6": "Input car time (1/0.6 factor already applied)",
    "free_flow_factor_1_0": "Free-flow time (×0.6)",
    "intermediate_factor_1_over_0_8": "Intermediate time (×0.6/0.8)",
    "reconstructed_base_plus_5_min": "Reconstructed time + 5 min",
    "main_adjusted_plus_5_min": "Input car time + 5 min",
}
SCENARIO_COLORS = {
    "main_factor_1_over_0_6": "#0072B2",
    "free_flow_factor_1_0": "#D55E00",
    "intermediate_factor_1_over_0_8": "#009E73",
    "reconstructed_base_plus_5_min": "#CC79A7",
    "main_adjusted_plus_5_min": "#E69F00",
}
THRESHOLD_COLORS = {15: "#0072B2", 30: "#009E73", 45: "#D55E00"}
CITY_ORDER = ("Sao Paulo", "Bogota")
SCOPE_ORDER = ("R", "E", "A")
BOGOTA_CLASSIFICATION_ORDER = (
    "legacy_original30_frozen", "mean_30_epsg4326", "mean_50_epsg4326", "mean_80_epsg4326",
    "dominant_area_epsg4326", "mean_30_epsg3116", "mean_50_epsg3116", "mean_80_epsg3116",
)
BOGOTA_CLASSIFICATION_LABELS = {
    "legacy_original30_frozen": "Frozen original classification (comparison)",
    "mean_30_epsg4326": "Area-weighted mean, >30% mapped area (EPSG:4326)",
    "mean_50_epsg4326": "Area-weighted mean, >50% mapped area (EPSG:4326)",
    "mean_80_epsg4326": "Area-weighted mean, >80% mapped area (EPSG:4326)",
    "dominant_area_epsg4326": "Dominant-area classification (EPSG:4326)",
    "mean_30_epsg3116": "Area-weighted mean, >30% mapped area (EPSG:3116)",
    "mean_50_epsg3116": "Area-weighted mean, >50% mapped area (EPSG:3116)",
    "mean_80_epsg3116": "Area-weighted mean, >80% mapped area (EPSG:3116)",
}


def text_value(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def semantic_scope_name(city: Any, scope: Any, profile: Any = "") -> str:
    raw, context, city_key = text_value(scope), text_value(profile).casefold(), text_value(city).casefold()
    if "->" in raw:
        left, right = raw.split("->", 1)
        return f"Activity {left.strip()} → Activity {right.strip()}"
    if raw == "R":
        if city_key in {"bogota", "bogotá"}:
            return "Historical-capacity frame (158 sites)" if "historical_capacity" in context else "Registry-linked frame (142 sites)"
        return "Registry"
    if raw == "E":
        if city_key in {"bogota", "bogotá"}:
            return "Public-provider signal"
        return "Public administration" if any(key in context for key in ("public_rd04", "planning_half_headway", "baseline_public_rd04_validation_frame")) else "Equipment-SUS"
    if raw == "A":
        if any(key in context for key in ("public_rd04", "planning_half_headway", "baseline_public_rd04_validation_frame")):
            return "Annual claims"
        return "Retained historical activity-flag inventory"
    return raw or "Not applicable"


def semantic_scope_definition(city: Any, scope: Any, profile: Any = "") -> str:
    raw, context, city_key = text_value(scope), text_value(profile).casefold(), text_value(city).casefold()
    if "->" in raw:
        left, right = raw.split("->", 1)
        return f"Paired annual-activity destination-set comparison: activity set {left.strip()} versus {right.strip()}."
    if raw == "R":
        if city_key in {"bogota", "bogotá"}:
            if "historical_capacity" in context:
                return "Historical-capacity Bogotá frame: 158 coordinate-linked 2022 capacity sites, including sites without later provider-register matches."
            return "Primary Bogotá registry-linked frame: positive 2022 surgical-room capacity, matched to the 2026 REPS site register, with accepted unique physical-site coordinates (142 sites)."
        return "São Paulo registry scenario: active CNES records meeting the frozen project surgical-centre, hospital-care and facility-type criteria."
    if raw == "E":
        if city_key in {"bogota", "bogotá"}:
            return "Bogotá public-provider signal among accepted physical surgical sites; distinct from patient entitlement and from São Paulo Equipment-SUS."
        return "Public-administration legal-nature scenario paired with annual group-04 claims." if any(key in context for key in ("public_rd04", "planning_half_headway", "baseline_public_rd04_validation_frame")) else "R-qualified sites with CNES code 0335 recorded in one or more 2024 monthly Equipment-SUS files."
    if raw == "A":
        return "R-qualified sites with a 2024 annual group-04 surgical-claims record." if any(key in context for key in ("public_rd04", "planning_half_headway", "baseline_public_rd04_validation_frame")) else "Retained historical activity-flag inventory; not current capacity or a complete annual-claims inventory."
    return f"Project-defined facility scenario {raw!r}."


def semantic_profile_definition(profile: Any) -> str:
    raw = text_value(profile)
    labels = {
        "baseline_fixed_departure": "Primary fixed-departure routing profile on the frozen origin frame.",
        "sensitivity_annual_activity_expansion": "São Paulo paired annual-activity destination-set comparison on the 1,100-origin main frame.",
        "baseline": "Primary São Paulo facility definitions: registry, Equipment-SUS and retained historical activity-flag inventory.",
        "planning_half_headway": "Nominal half-headway planning scenario versus the public-administration/annual-claims baseline on the same validation frame.",
        "baseline_public_rd04_validation_frame": "São Paulo public-administration/annual-claims baseline on the 220-origin validation frame.",
    }
    return labels.get(raw, f"Raw scenario/profile key {raw!r} retained in the adjacent column.") if raw else "No separate routing profile applies."


def bogota_classification_definition(scenario: Any) -> str:
    raw = text_value(scenario)
    key = raw.casefold()
    if key == "legacy_original30_frozen":
        return "Frozen original Bogotá classification retained as a historical comparison; it is not the corrected primary grouping."
    if key == "dominant_area_epsg4326":
        return "Dominant-area Bogotá estrato classification in EPSG:4326, as specified in the frozen sensitivity plan."
    if key.startswith("mean_") and ("epsg4326" in key or "epsg3116" in key):
        threshold = key.split("_", 2)[1]
        crs = "EPSG:4326" if "epsg4326" in key else "EPSG:3116"
        suffix = " Boundary values within 1e-12 of 2, 4 or 6 are snapped before classification." if key == "mean_30_epsg4326" else ""
        return f"Area-weighted mapped-block mean estrato in {crs}, requiring >{threshold}% mapped cell area.{suffix}"
    return f"Scenario key {raw!r} is retained; consult the frozen classification sensitivity plan for its operational definition."


def add_semantic_columns(df: pd.DataFrame, city_hint: str = "") -> pd.DataFrame:
    out = df.copy()
    names, definitions, profiles = [], [], []
    for _, row in out.iterrows():
        city = text_value(row.get("city", "")) or city_hint
        profile = text_value(row.get("profile", "")) or text_value(row.get("analysis_profile", ""))
        raw_scope = text_value(row.get("scope", "")) or text_value(row.get("activity_set_pair", ""))
        names.append(semantic_scope_name(city, raw_scope, profile))
        definitions.append(semantic_scope_definition(city, raw_scope, profile))
        profiles.append(semantic_profile_definition(profile))
    out["scope_name"] = names
    out["scope_definition"] = definitions
    out["profile_definition"] = profiles
    return out

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(root: Path, rel: str, required: set[str]) -> pd.DataFrame:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"Required source is absent: {rel}")
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{rel} lacks required columns: {', '.join(missing)}")
    return df


def numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for name in columns:
        df[name] = pd.to_numeric(df[name], errors="coerce")


def finite(df: pd.DataFrame, columns: list[str], label: str) -> None:
    for name in columns:
        if not np.isfinite(pd.to_numeric(df[name], errors="coerce")).all():
            raise ValueError(f"{label}: {name} contains missing or non-finite values.")


def check_interval(df: pd.DataFrame, point: str, lo: str, hi: str,
                   label: str, allow_missing: bool = False) -> None:
    lower = pd.to_numeric(df[lo], errors="coerce")
    upper = pd.to_numeric(df[hi], errors="coerce")
    estimate = pd.to_numeric(df[point], errors="coerce")
    both_missing = lower.isna() & upper.isna()
    one_missing = lower.isna() ^ upper.isna()
    if one_missing.any() or (both_missing.any() and not allow_missing):
        raise ValueError(f"{label}: interval endpoints are incomplete.")
    present = ~both_missing
    if (lower[present] > upper[present]).any():
        raise ValueError(f"{label}: interval lower endpoint exceeds upper endpoint.")
    if ((estimate[present] < lower[present] - 1e-7)
            | (estimate[present] > upper[present] + 1e-7)).any():
        raise ValueError(f"{label}: interval does not contain its point estimate.")


def require_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    if df.duplicated(keys).any():
        raise ValueError(f"{label}: duplicate rows for key {keys}.")


def save_figure(fig: plt.Figure, out: Path, stem: str) -> list[str]:
    paths = [
        out / f"{stem}.svg",
        out / f"{stem}.pdf",
        out / f"{stem}.png",
        out / f"{stem}.tiff",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=300, bbox_inches="tight")
    fig.savefig(paths[3], dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return [path.name for path in paths]


def source_record(out: Path, rel: str, role: str, nrows: int) -> dict[str, Any]:
    path = out / rel
    return {
        "role": role,
        "file": rel,
        "rows": int(nrows),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def prepare_s1(root: Path, out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rel = "extensions/car_sensitivity_curves.csv"
    df = read_csv(root, rel, {
        "city", "profile", "scope", "scenario", "threshold_min",
        "car_minus_wt_pp", "technical_delta_lower_pp",
        "technical_delta_upper_pp", "design_se_pp", "design_ci_lower_pp",
        "design_ci_upper_pp", "n_origins",
    })
    df = df.loc[
        df.city.astype(str).isin(CITY_ORDER)
        & df.profile.astype(str).eq("baseline_fixed_departure")
        & df.scope.astype(str).eq("R")
        & df.scenario.astype(str).isin(SCENARIO_ORDER)
    ].copy()
    numeric(df, [
        "threshold_min", "car_minus_wt_pp", "technical_delta_lower_pp",
        "technical_delta_upper_pp", "design_se_pp", "design_ci_lower_pp",
        "design_ci_upper_pp", "n_origins",
    ])
    require_unique(df, ["city", "scope", "profile", "scenario", "threshold_min"], "S1 car sensitivity")
    expected = {(city, scenario, threshold) for city in CITY_ORDER
                for scenario in SCENARIO_ORDER for threshold in ALL_THRESHOLDS}
    observed = set(zip(
        df.city.astype(str), df.scenario.astype(str), df.threshold_min.astype(int)
    ))
    if observed != expected:
        raise ValueError("S1 source is not a complete city × scenario × 10–60 min threshold grid.")
    if df.n_origins.isna().any():
        raise ValueError("S1 source has missing origin counts.")
    for city, expected_n in (("Sao Paulo", 1100), ("Bogota", 1057)):
        part = df.loc[df.city.eq(city)]
        if set(part.n_origins.astype(int)) != {expected_n}:
            raise ValueError(f"S1 {city} origin count differs from its frozen frame ({expected_n}).")
    sp = df.loc[df.city.eq("Sao Paulo")]
    bo = df.loc[df.city.eq("Bogota")]
    finite(sp, ["design_ci_lower_pp", "design_ci_upper_pp"], "S1 Sao Paulo design CI")
    if not np.isfinite(bo.design_se_pp).all() or not bo.design_se_pp.eq(0).all():
        raise ValueError("S1 Bogota must retain its complete-frame design SE of zero.")
    if bo.design_ci_lower_pp.notna().any() or bo.design_ci_upper_pp.notna().any():
        raise ValueError("S1 Bogota contains an interval; no census sampling CI may be plotted.")
    finite(df, ["car_minus_wt_pp", "technical_delta_lower_pp", "technical_delta_upper_pp"], "S1 point/bounds")
    check_interval(df, "car_minus_wt_pp", "technical_delta_lower_pp",
                   "technical_delta_upper_pp", "S1 technical bounds")
    check_interval(sp, "car_minus_wt_pp", "design_ci_lower_pp",
                   "design_ci_upper_pp", "S1 São Paulo design CI")
    notes = json.loads((root / "extensions" / "ANALYSIS_NOTES.json").read_text(encoding="utf-8"))
    if "car_scenarios" not in notes or "technical_bounds" not in notes:
        raise ValueError("S1 source notes do not document the scenario and interval semantics.")
    df["sampling_frame"] = np.where(df.city.eq("Sao Paulo"), "Main frame; stratified SRS", "Complete frame; census")
    df = add_semantic_columns(df)
    output = "FigureS1_car_time_sensitivity_source.csv"
    df.sort_values(["city", "scenario", "threshold_min"]).to_csv(out / output, index=False)
    source = {
        "role": "S1 city-specific R-scope car-minus-WT time-assumption sensitivity",
        "source_file": rel,
        "source_sha256": sha256(root / rel),
        "selected_rows": len(df),
        "output_file": output,
        "technical_bounds_collapsed": bool(np.allclose(
            df.technical_delta_lower_pp, df.technical_delta_upper_pp, atol=1e-9, rtol=0
        ) and np.allclose(df.technical_delta_lower_pp, df.car_minus_wt_pp, atol=1e-9, rtol=0)),
    }
    return df, source


def plot_s1(df: pd.DataFrame, out: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.55), sharex=True, sharey=True)
    for ax, city, letter in zip(axes, CITY_ORDER, ("a", "b")):
        part_city = df.loc[df.city.eq(city)]
        n = int(part_city.n_origins.iloc[0])
        frame = "stratified probability sample" if city == "Sao Paulo" else "census"
        panel_city = "São Paulo" if city == "Sao Paulo" else "Bogotá"
        frame_label = "probability sample" if city == "Sao Paulo" else "complete frame"
        ax.set_title(f"{letter}  {panel_city}\nn = {n:,}; {frame_label}",
                     loc="left", pad=5, fontweight="bold", linespacing=1.0)
        for scenario in SCENARIO_ORDER:
            part = part_city.loc[part_city.scenario.eq(scenario)].sort_values("threshold_min")
            x = part.threshold_min.to_numpy(float)
            y = part.car_minus_wt_pp.to_numpy(float)
            color = SCENARIO_COLORS[scenario]
            ax.plot(x, y, color=color, marker="o", markersize=2.4, linewidth=1.15,
                    label=SCENARIO_LABELS[scenario], zorder=3)
            if city == "Sao Paulo":
                lo = part.design_ci_lower_pp.to_numpy(float)
                hi = part.design_ci_upper_pp.to_numpy(float)
                ax.fill_between(x, lo, hi, color=color, alpha=0.11, linewidth=0, zorder=1)
            # Technical endpoints are kept separate from the design CI. Thin
            # whiskers are visible only when route-status uncertainty widens them.
            lo_t = part.technical_delta_lower_pp.to_numpy(float)
            hi_t = part.technical_delta_upper_pp.to_numpy(float)
            for xx, low, high in zip(x, lo_t, hi_t):
                if high - low > 1e-7:
                    ax.vlines(xx, low, high, color="#666666", linewidth=0.7, alpha=0.7, zorder=2)
        ax.axhline(0, color="#555555", linewidth=0.65, linestyle=(0, (2, 2)), zorder=0)
        ax.set_xlim(10, 60)
        ax.set_xticks(ALL_THRESHOLDS)
        ax.grid(axis="y", color="#D9DDE1", linewidth=0.45)
        ax.tick_params(length=2, pad=2)
    axes[0].set_ylabel("Car − WT coverage (pp)")
    for ax in axes:
        ax.set_xlabel("Time threshold (min)")
    fig.suptitle("Car-time sensitivity across thresholds (R scope)",
                 x=0.07, y=0.995, ha="left", fontsize=9, fontweight="bold")
    handles = [Line2D([0], [0], color=SCENARIO_COLORS[s], marker="o", linewidth=1.1,
                      markersize=2.5, label=SCENARIO_LABELS[s]) for s in SCENARIO_ORDER]
    handles.append(Patch(facecolor="#888888", alpha=0.15, edgecolor="none",
                         label="95% design CI (São Paulo only)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.045), columnspacing=1.3, handlelength=1.6)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.22, top=0.83, wspace=0.15)
    return save_figure(fig, out, "FigureS1_car_time_sensitivity")


def prepare_activity(root: Path, out: Path) -> pd.DataFrame:
    rel = "activity_paired_gains_verified/paired_activity_gains.csv"
    df = read_csv(root, rel, {
        "city", "profile", "activity_set_pair", "base_scope", "extra_scope",
        "base_site_count", "extra_site_count", "mode", "threshold_min",
        "n_origins", "sampling_design", "gain_estimate_pp", "design_ci95_lower_pp",
        "design_ci95_upper_pp", "technical_bound_lower_pp", "technical_bound_upper_pp",
    })
    df = df.loc[
        df.city.astype(str).eq("Sao Paulo")
        & df.profile.astype(str).eq("sensitivity_annual_activity_expansion")
        & df.threshold_min.isin(THRESHOLDS)
    ].copy()
    numeric(df, [
        "threshold_min", "n_origins", "gain_estimate_pp", "design_ci95_lower_pp",
        "design_ci95_upper_pp", "technical_bound_lower_pp", "technical_bound_upper_pp",
        "base_site_count", "extra_site_count",
    ])
    expected_pairs = {"156->161", "156->162", "161->162"}
    if set(df.activity_set_pair.astype(str)) != expected_pairs:
        raise ValueError("Activity expansion source is missing or adds an unverified activity-set pair.")
    if set(df.threshold_min.astype(int)) != set(THRESHOLDS):
        raise ValueError("Activity expansion source is missing a 15/30/45-minute node.")
    if set(df.n_origins.astype(int)) != {1100} or set(df.sampling_design.astype(str)) != {"stratified_SRS"}:
        raise ValueError("Activity expansion rows do not match the 1,100-origin São Paulo main frame.")
    if df.technical_bound_lower_pp.isna().any() or df.technical_bound_upper_pp.isna().any():
        raise ValueError("Activity expansion has missing technical bounds.")
    check_interval(df, "gain_estimate_pp", "technical_bound_lower_pp",
                   "technical_bound_upper_pp", "Activity technical bounds")
    check_interval(df, "gain_estimate_pp", "design_ci95_lower_pp",
                   "design_ci95_upper_pp", "Activity design CI", allow_missing=True)
    require_unique(df, ["activity_set_pair", "mode", "threshold_min"], "Activity expansion")
    return df


def prepare_regional(root: Path, out: Path) -> pd.DataFrame:
    rel = "regional_transit_sensitivity/paired_regional_comparison.csv"
    df = read_csv(root, rel, {
        "city", "scope", "mode", "threshold_min", "group", "n",
        "paired_change_pp", "paired_design_se_pp", "paired_design_ci_lower_pp", "paired_design_ci_upper_pp",
        "technical_missing_bound_lower_pp", "technical_missing_bound_upper_pp",
    })
    df = df.loc[
        df.city.astype(str).eq("Sao Paulo")
        & df["mode"].astype(str).eq("wt")
        & df.group.astype(str).eq("overall")
        & df.threshold_min.isin(THRESHOLDS)
        & df.scope.astype(str).isin(SCOPE_ORDER)
    ].copy()
    numeric(df, [
        "threshold_min", "n", "paired_change_pp", "paired_design_se_pp", "paired_design_ci_lower_pp",
        "paired_design_ci_upper_pp", "technical_missing_bound_lower_pp",
        "technical_missing_bound_upper_pp",
    ])
    require_unique(df, ["city", "scope", "mode", "threshold_min", "group"], "Regional paired comparison")
    expected = {(scope, threshold) for scope in SCOPE_ORDER for threshold in THRESHOLDS}
    observed = set(zip(df.scope.astype(str), df.threshold_min.astype(int)))
    if observed != expected or set(df.n.astype(int)) != {220}:
        raise ValueError("Regional WT rows do not form the complete 220-origin validation-frame R/E/A × 15/30/45 grid.")
    finite(df, ["paired_change_pp", "technical_missing_bound_lower_pp", "technical_missing_bound_upper_pp"],
           "Regional estimates/bounds")
    check_interval(df, "paired_change_pp", "technical_missing_bound_lower_pp",
                   "technical_missing_bound_upper_pp", "Regional technical bounds")
    check_interval(df, "paired_change_pp", "paired_design_ci_lower_pp",
                   "paired_design_ci_upper_pp", "Regional design CI", allow_missing=True)
    missing = df.paired_design_ci_lower_pp.isna() & df.paired_design_ci_upper_pp.isna()
    partial = df.paired_design_ci_lower_pp.isna() ^ df.paired_design_ci_upper_pp.isna()
    if partial.any() or (missing & df.paired_design_se_pp.ne(0)).any():
        raise ValueError("Regional nonzero-variance estimates require both design interval limits.")
    df["profile"] = "baseline"
    return df


def _draw_interval(ax: plt.Axes, y: float, lo: float, hi: float, color: str,
                   linewidth: float, alpha: float = 1.0) -> None:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return
    ax.hlines(y, lo, hi, color=color, linewidth=linewidth, alpha=alpha, zorder=2)
    cap = 0.045
    ax.vlines([lo, hi], y - cap, y + cap, color=color, linewidth=max(0.45, linewidth * 0.7),
              alpha=alpha, zorder=2)


def draw_forest_panel(ax: plt.Axes, df: pd.DataFrame, group_col: str,
                      group_keys: list[str], group_labels: dict[str, str],
                      estimate_col: str, ci_lo: str, ci_hi: str,
                      tech_lo: str, tech_hi: str, title: str) -> None:
    offsets = {15: -0.19, 30: 0.0, 45: 0.19}
    centers = np.arange(len(group_keys), dtype=float)[::-1]
    for center, key in zip(centers, group_keys):
        group = df.loc[df[group_col].astype(str).eq(key)]
        for _, row in group.iterrows():
            threshold = int(row.threshold_min)
            y = center + offsets[threshold]
            _draw_interval(ax, y + 0.055, float(row[tech_lo]), float(row[tech_hi]), "#777777", 1.8, 0.8)
            ci_low, ci_high = row[ci_lo], row[ci_hi]
            if pd.notna(ci_low) and pd.notna(ci_high):
                _draw_interval(ax, y - 0.055, float(ci_low), float(ci_high), "#222222", 0.75, 0.95)
            ax.scatter(float(row[estimate_col]), y, s=15, marker="o",
                       color=THRESHOLD_COLORS[threshold], edgecolor="white", linewidth=0.35, zorder=4)
    ax.axvline(0, color="#555555", linewidth=0.65, linestyle=(0, (2, 2)), zorder=0)
    ax.set_yticks(centers)
    ax.set_yticklabels([group_labels[k] for k in group_keys])
    ax.set_title(title, loc="left", pad=4, fontweight="bold")
    ax.grid(axis="x", color="#D9DDE1", linewidth=0.45)
    ax.tick_params(length=2, pad=2)
    ax.set_xlabel("Paired coverage change (percentage points)")
    ax.set_ylim(-0.5, len(group_keys) - 0.5)
    # Missing CIs remain visible as intentionally absent intervals; this
    # finite-only list controls axis limits and does not filter plotted rows.
    values = []
    for column in (estimate_col, ci_lo, ci_hi, tech_lo, tech_hi):
        for value in pd.to_numeric(df[column], errors="coerce").to_numpy():
            if np.isfinite(value):
                values.append(float(value))
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.08 * span, hi + 0.08 * span)


def load_headway(root: Path) -> tuple[pd.DataFrame, str]:
    curves_rel = "planning_complete_headway_half/planning_coverage_curves.csv"
    gains_rel = "planning_complete_headway_half/planning_paired_gains.csv"
    if not (root / curves_rel).is_file() or not (root / gains_rel).is_file():
        return pd.DataFrame(), "Headway output is still pending; no S2 figure was exported."
    try:
        curves = read_csv(root, curves_rel, {
            "city", "scope", "mode", "analysis_profile", "n", "threshold",
            "coverage_lower", "design_ci_lower", "design_ci_upper",
        })
        gains = read_csv(root, gains_rel, {
            "city", "scope", "mode", "threshold", "group", "n",
            "paired_gain_lower_estimate_pp", "paired_gain_design_se_pp", "paired_gain_design_ci_lower_pp",
            "paired_gain_design_ci_upper_pp", "technical_error_paired_bound_lower_pp",
            "technical_error_paired_bound_upper_pp",
        })
    except (FileNotFoundError, ValueError) as exc:
        return pd.DataFrame(), f"Headway outputs are present but not yet complete/valid: {exc}"
    required_profiles = {"baseline_public_rd04_validation_frame", "planning_half_headway"}
    curve_part = curves.loc[
        curves.city.astype(str).eq("Sao Paulo")
        & curves.scope.astype(str).isin(SCOPE_ORDER)
        & curves["mode"].astype(str).eq("wt")
        & curves.analysis_profile.astype(str).isin(required_profiles)
    ].copy()
    numeric(curve_part, ["threshold", "n", "coverage_lower", "design_ci_lower", "design_ci_upper"])
    require_unique(curve_part, ["scope", "analysis_profile", "threshold"], "Headway coverage source")
    for scope in SCOPE_ORDER:
        for profile in required_profiles:
            part = curve_part.loc[curve_part.scope.eq(scope) & curve_part.analysis_profile.eq(profile)]
            if set(part.threshold.astype(int)) != set(ALL_THRESHOLDS) or set(part.n.astype(int)) != {220}:
                raise ValueError(f"Headway source fails full 220-origin curve check for {scope}/{profile}.")
            finite(part, ["coverage_lower", "design_ci_lower", "design_ci_upper"], "Headway curve/CIs")
    curve_delta = (
        curve_part.pivot(index=["scope", "threshold"], columns="analysis_profile", values="coverage_lower")
        .reset_index()
    )
    gain_part = gains.loc[
        gains.city.astype(str).eq("Sao Paulo")
        & gains.scope.astype(str).isin(SCOPE_ORDER)
        & gains["mode"].astype(str).eq("wt")
        & gains.group.astype(str).eq("overall")
        & gains.threshold.isin(THRESHOLDS)
    ].copy()
    numeric(gain_part, [
        "threshold", "n", "paired_gain_lower_estimate_pp", "paired_gain_design_se_pp", "paired_gain_design_ci_lower_pp",
        "paired_gain_design_ci_upper_pp", "technical_error_paired_bound_lower_pp",
        "technical_error_paired_bound_upper_pp",
    ])
    require_unique(gain_part, ["scope", "threshold"], "Headway paired gains")
    expected = {(scope, threshold) for scope in SCOPE_ORDER for threshold in THRESHOLDS}
    observed = set(zip(gain_part.scope.astype(str), gain_part.threshold.astype(int)))
    if observed != expected or set(gain_part.n.astype(int)) != {220}:
        raise ValueError("Headway paired gains do not form the complete 220-origin WT R/E/A × 15/30/45 grid.")
    finite(gain_part, [
        "paired_gain_lower_estimate_pp", "paired_gain_design_se_pp", "technical_error_paired_bound_lower_pp",
        "technical_error_paired_bound_upper_pp",
    ], "Headway paired estimates/CIs/bounds")
    check_interval(gain_part, "paired_gain_lower_estimate_pp",
                   "technical_error_paired_bound_lower_pp",
                   "technical_error_paired_bound_upper_pp", "Headway technical bounds")
    check_interval(gain_part, "paired_gain_lower_estimate_pp",
                   "paired_gain_design_ci_lower_pp",
                   "paired_gain_design_ci_upper_pp", "Headway design CI", allow_missing=True)
    missing_ci = gain_part.paired_gain_design_ci_lower_pp.isna() & gain_part.paired_gain_design_ci_upper_pp.isna()
    if (missing_ci & gain_part.paired_gain_design_se_pp.ne(0)).any():
        raise ValueError("Headway nonzero-variance paired estimate lacks a design interval")
    paired_check = gain_part.merge(curve_delta, on=["scope", "threshold"], validate="one_to_one")
    curve_change = (
        paired_check["planning_half_headway"]
        - paired_check["baseline_public_rd04_validation_frame"]
    )
    if not np.allclose(curve_change, paired_check.paired_gain_lower_estimate_pp, atol=0.03, rtol=0):
        raise ValueError("Headway paired gains do not reconcile to the full-frame scenario-minus-baseline curves.")
    gain_part = gain_part.rename(columns={
        "threshold": "threshold_min",
        "paired_gain_lower_estimate_pp": "headway_change_pp",
        "paired_gain_design_ci_lower_pp": "headway_ci_lower_pp",
        "paired_gain_design_ci_upper_pp": "headway_ci_upper_pp",
        "technical_error_paired_bound_lower_pp": "headway_technical_lower_pp",
        "technical_error_paired_bound_upper_pp": "headway_technical_upper_pp",
    })
    gain_part["contrast"] = "planning_half_headway minus baseline_public_rd04_validation_frame"
    gain_part["profile"] = "planning_half_headway"
    return gain_part, ""


def prepare_s3(analysis_dir: Path, out: Path) -> tuple[list[str], dict[str, Any], str]:
    gradient_path = analysis_dir / "classification_low_high_gradients.csv"
    historical_path = analysis_dir / "historical142_158_low_high_gradients.csv"
    if not gradient_path.is_file() or not historical_path.is_file():
        return [], {}, "Corrected classification-gradient or historical 142/158 source is absent."
    schemes = pd.read_csv(gradient_path)
    required = {"scenario", "scope", "mode", "threshold", "low_population", "high_population", "low_coverage", "high_coverage", "low_minus_high"}
    if not required.issubset(schemes.columns):
        raise ValueError(f"S3 classification-gradient source lacks columns: {sorted(required - set(schemes.columns))}")
    schemes = schemes.loc[
        schemes.scenario.astype(str).isin(BOGOTA_CLASSIFICATION_ORDER)
        & schemes.scope.astype(str).isin(("R", "E"))
        & schemes["mode"].astype(str).eq("wt")
        & pd.to_numeric(schemes.threshold, errors="coerce").eq(30)
    ].copy()
    numeric(schemes, ["threshold", "low_population", "high_population", "low_coverage", "high_coverage", "low_minus_high"])
    expected_schemes = {(scenario, scope) for scenario in BOGOTA_CLASSIFICATION_ORDER for scope in ("R", "E")}
    observed_schemes = set(zip(schemes.scenario.astype(str), schemes.scope.astype(str)))
    if observed_schemes != expected_schemes or schemes.duplicated(["scenario", "scope"]).any():
        raise ValueError("S3 must retain all eight classification schemes for both R and E at 30-minute WT.")

    historical = pd.read_csv(historical_path)
    required_historical = {"grouping", "facility_frame", "scope", "mode", "threshold", "low_coverage", "high_coverage", "low_minus_high"}
    if not required_historical.issubset(historical.columns):
        raise ValueError(f"S3 historical 142/158 source lacks columns: {sorted(required_historical - set(historical.columns))}")
    historical = historical.loc[
        historical.grouping.astype(str).eq("mean_30_epsg4326")
        & historical.facility_frame.astype(str).isin(("primary142", "historical158"))
        & historical.scope.astype(str).isin(("R", "E"))
        & historical["mode"].astype(str).eq("wt")
        & pd.to_numeric(historical.threshold, errors="coerce").eq(30)
    ].copy()
    numeric(historical, ["threshold", "low_coverage", "high_coverage", "low_minus_high"])
    expected_frames = {(frame, scope) for frame in ("primary142", "historical158") for scope in ("R", "E")}
    observed_frames = set(zip(historical.facility_frame.astype(str), historical.scope.astype(str)))
    if observed_frames != expected_frames or historical.duplicated(["facility_frame", "scope"]).any():
        raise ValueError("S3 must retain both R/E arms for primary142 and historical158 under mean_30_epsg4326.")

    schemes["city"] = "Bogota"
    schemes["profile"] = "classification_" + schemes.scenario.astype(str)
    schemes["panel"] = "all_classification_schemes"
    schemes["classification_key"] = schemes.scenario.astype(str)
    schemes["classification_label"] = schemes.scenario.map(BOGOTA_CLASSIFICATION_LABELS)
    schemes["classification_scenario_definition"] = schemes.scenario.map(bogota_classification_definition)
    schemes["facility_frame"] = "primary142"
    schemes["low_minus_high_pp"] = schemes.low_minus_high
    schemes["source_file"] = str(gradient_path)
    schemes["source_sha256"] = sha256(gradient_path)

    historical["city"] = "Bogota"
    historical["profile"] = "sensitivity_bo_historical_capacity"
    historical["panel"] = "facility_frame_sensitivity"
    historical["classification_key"] = historical.grouping.astype(str)
    historical["classification_label"] = BOGOTA_CLASSIFICATION_LABELS["mean_30_epsg4326"]
    historical["classification_scenario_definition"] = bogota_classification_definition("mean_30_epsg4326")
    historical["low_minus_high_pp"] = historical.low_minus_high
    historical["source_file"] = str(historical_path)
    historical["source_sha256"] = sha256(historical_path)

    source = pd.concat([schemes, historical], ignore_index=True, sort=False)
    source["threshold_min"] = pd.to_numeric(source.threshold, errors="coerce")
    source["group_definition"] = "Low=(0,2], middle=(2,4], high=(4,6] residential estrato; unknown origins remain in the complete frame."
    source["sampling_interval"] = "None; Bogotá is a complete-frame census."
    source = add_semantic_columns(source)
    output_csv = "FigureS3_bogota_classification_source.csv"
    source.to_csv(out / output_csv, index=False)

    frame_labels = {
        "primary142": "Primary registry-linked frame (142 sites)",
        "historical158": "Historical-capacity frame (158 sites)",
    }
    colors = {"R": "#4D4D4D", "E": "#0F4D92"}
    values = pd.to_numeric(source.low_minus_high_pp, errors="coerce").to_numpy(float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    span = max(hi - lo, 1.0)
    xlim = (min(0.0, lo) - 0.07 * span, max(0.0, hi) + 0.07 * span)
    fig, axes = plt.subplots(
        2, 1, figsize=(7.09, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3.9, 1.35]},
    )
    ax1, ax2 = axes
    centers = np.arange(len(BOGOTA_CLASSIFICATION_ORDER), dtype=float)[::-1]
    offsets = {"R": 0.13, "E": -0.13}
    for center, scenario in zip(centers, BOGOTA_CLASSIFICATION_ORDER):
        part = schemes.loc[schemes.scenario.astype(str).eq(scenario)]
        for scope in ("R", "E"):
            row = part.loc[part.scope.astype(str).eq(scope)].iloc[0]
            y = center + offsets[scope]
            estimate = float(row.low_minus_high)
            ax1.scatter(estimate, y, color=colors[scope], marker="o", s=18, zorder=3)

    def wrapped_label(label: str, width: int = 32) -> str:
        return textwrap.fill(label, width=width, break_long_words=False, break_on_hyphens=False)

    ax1.set_yticks(
        centers,
        [wrapped_label(BOGOTA_CLASSIFICATION_LABELS[k]) for k in BOGOTA_CLASSIFICATION_ORDER],
        fontsize=7.0,
    )
    ax1.set_title("a  Classification schemes", loc="left", fontsize=6.8, fontweight="bold")
    ax1.axvline(0, color="#666666", linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    ax1.grid(axis="x", color="#D9DDE1", linewidth=0.45)
    ax1.tick_params(axis="y", length=0, pad=3)
    ax1.tick_params(axis="x", length=2, pad=2, labelsize=6.5)
    ax1.set_ylim(-0.55, len(BOGOTA_CLASSIFICATION_ORDER) - 0.45)
    frame_order = ("primary142", "historical158")
    frame_centers = np.arange(2, dtype=float)[::-1]
    for center, frame in zip(frame_centers, frame_order):
        part = historical.loc[historical.facility_frame.astype(str).eq(frame)]
        for scope in ("R", "E"):
            row = part.loc[part.scope.astype(str).eq(scope)].iloc[0]
            y = center + offsets[scope]
            estimate = float(row.low_minus_high)
            ax2.scatter(estimate, y, color=colors[scope], marker="o", s=22, zorder=3)
            ax2.annotate(f"{estimate:.1f}", (estimate, y), xytext=(3, 2), textcoords="offset points", fontsize=5.3)
    ax2.set_yticks(frame_centers, [wrapped_label(frame_labels[k]) for k in frame_order], fontsize=7.0)
    ax2.set_title("b  Facility frames", loc="left", fontsize=6.8, fontweight="bold")
    ax2.set_xlabel("WT 30-min low − high coverage (percentage points)")
    ax2.axvline(0, color="#666666", linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    ax2.grid(axis="x", color="#D9DDE1", linewidth=0.45)
    ax2.tick_params(axis="y", length=0, pad=3)
    ax2.tick_params(axis="x", length=2, pad=2, labelsize=6.5)
    ax2.set_ylim(-0.52, 1.52)
    handles = [
        Line2D([0], [0], color=colors["R"], marker="o", linewidth=0, markersize=4, label="R · Registry-linked sites"),
        Line2D([0], [0], color=colors["E"], marker="o", linewidth=0, markersize=4, label="E · Public-provider signal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=5.8,
               bbox_to_anchor=(0.5, 0.14), columnspacing=1.2, handlelength=1.2)
    fig.suptitle("Bogotá subgroup and facility-frame sensitivity", x=0.07, y=0.995, ha="left", fontsize=8.5, fontweight="bold")
    fig.text(
        0.07, 0.018,
        "Under the strict >80% mapped-area rule, only 2.05% of population is classified into low/middle/high groups. "
        "The corrected primary grouping is shown for the 142-site frame; Bogotá is a complete-frame census with no sampling CI. "
        "The frozen original grouping is a historical comparison only.",
        fontsize=5.2, va="bottom", wrap=True,
    )
    fig.subplots_adjust(left=0.43, right=0.99, bottom=0.26, top=0.92, hspace=0.5)
    figure_paths = save_figure(fig, out, "FigureS3_bogota_classification")
    meta = {
        "role": "S3 Bogotá all-scheme subgroup and primary142/historical158 sensitivity",
        "source_files": [str(gradient_path), str(historical_path)],
        "source_sha256": [sha256(gradient_path), sha256(historical_path)],
        "selected_rows": int(len(source)),
        "output_file": output_csv,
        "figure_files": figure_paths,
        "sampling_ci": "none; complete-frame census",
    }
    return figure_paths, meta, ""


def write_captions(out: Path, s1_bounds_collapsed: bool, s2_ready: bool,
                   s3_ready: bool, s3_reason: str = "", provisional: bool = False) -> None:
    tech_s1 = (
        "Technical bounds coincide with the lower-estimate contrasts in the displayed rows."
        if s1_bounds_collapsed else
        "Thin grey vertical intervals show technical bounds; translucent coloured bands show design CIs."
    )
    lines = [
        "# Supplemental figure legends",
        "",
        *(["**Export status: PROVISIONAL. These figures use the current baseline route tables pending the metro-inclusion correction; rerun before final publication use.**", ""] if provisional else []),
        "**Fig. S1 | Sensitivity of the R-scope car–WT coverage contrast to car-time assumptions**",
        "",
            "a, b, Car-minus-WT lower-coverage differences across the prespecified 10–60 min thresholds for São Paulo and Bogotá, respectively. Colored lines identify the five documented car-time scenarios; translucent bands show pointwise 95% design confidence intervals for São Paulo’s 1,100-origin stratified probability sample. Bogotá’s 1,057-origin complete-frame census has no sampling confidence interval. R denotes each city’s own registry-based scenario and is not a harmonized cross-city service definition. " + tech_s1 + " Source data are provided in the accompanying source-data CSV.",
        "",
    ]
    if s2_ready:
        lines.extend([
            "**Fig. S2 | Paired WT coverage changes under activity, regional-network and headway sensitivities**",
            "",
            "a, Paired WT coverage gains for Activity 156, Activity 161 and Activity 162 destination-set comparisons at 15, 30 and 45 min on the São Paulo 1,100-origin main frame. b, Paired WT coverage changes for regional-network versus baseline routing across Registry, Equipment-SUS and the retained historical activity-flag inventory on the São Paulo 220-origin validation frame. c, Paired WT coverage change under the half-headway planning scenario versus the public-administration/annual-claims baseline on the same 220-origin validation frame. Colored circles show paired lower-coverage estimates; black whiskers show 95% design CIs, and grey intervals show technical bounds from unresolved route outcomes. The two interval types are distinct and are not combined. Source data are provided in the accompanying source-data CSVs.",
            "",
        ])
    else:
        lines.extend([
            "**Fig. S2 | Paired WT coverage changes under activity, regional-network and headway sensitivities**",
            "",
            "Figure export is pending completion and validation of the half-headway paired results. No headway values are imputed or shown in the current package.",
            "",
        ])
    if s3_ready:
        lines.extend([
            "**Fig. S3 | Bogotá socioeconomic-classification and facility-frame sensitivity**",
            "",
            "a, WT coverage difference between the low and high residential estrato groups at 30 min for all eight classification schemes: frozen original classification (historical comparison only), three area-weighted mean schemes in EPSG:4326 (>30%, >50%, >80% mapped-area thresholds), dominant-area classification in EPSG:4326, and three area-weighted mean schemes in EPSG:3116 (>30%, >50%, >80%). Under the strict >80% mapped-area rule, only 2.05% of the Bogotá population is classified into low/middle/high groups. These classification sensitivities were exploratory and specified after the original baseline inspection; the v4 plan was frozen before new outcome summaries. b, Under the corrected primary `mean_30_epsg4326` grouping, the same WT 30-min low−high contrast is shown for the 142-site registry-linked frame and the 158-site historical-capacity frame. Low, middle and high are residential estrato groups (0,2], (2,4] and (4,6]; unknown cells remain in the full frame. Bogotá is a complete-frame census with no sampling confidence interval. Grouping sensitivity changes classification only, not routing. Exact scenario keys, subgroup populations and contrasts are provided in the source data.",
            "",
        ])
    else:
        lines.extend([
            "**Fig. S3 | Bogotá socioeconomic-classification and facility-frame sensitivity**",
            "",
            f"Figure export is pending validated classification inputs: {s3_reason or 'corrected Bogotá sources are unavailable'}",
            "",
        ])
    (out / "FIGURE_LEGENDS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Verified v13 results root")
    parser.add_argument("--out", type=Path, required=True, help="Supplement output directory")
    parser.add_argument("--bogota-reanalysis", type=Path, default=None, help="Corrected Bogotá classification output directory")
    parser.add_argument("--provisional", action="store_true", help="Mark the current baseline figures provisional pending routing correction")
    parser.add_argument("--only-s3", action="store_true", help="Render only the classification figure and its source table")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    analysis_dir = args.bogota_reanalysis.resolve() if args.bogota_reanalysis else (root / "hp_revision_20261116" / "bogota")
    out.mkdir(parents=True, exist_ok=True)
    if args.only_s3:
        s3_paths, s3_meta, s3_reason = prepare_s3(analysis_dir, out)
        if not s3_paths:
            print(json.dumps({"S3": "PENDING_CLASSIFICATION", "reason": s3_reason}, indent=2, ensure_ascii=False))
            return 1
        print(json.dumps({"S3": "READY", "provisional_bundle_status": bool(args.provisional), **s3_meta}, indent=2, ensure_ascii=False))
        return 0
    records: list[dict[str, Any]] = []
    s1, s1_meta = prepare_s1(root, out)
    records.append(s1_meta)
    figures = plot_s1(s1, out)

    activity = prepare_activity(root, out)
    activity_out = activity.loc[
        activity["mode"].astype(str).eq("wt")
        & activity.threshold_min.isin(THRESHOLDS)
    ].sort_values(["activity_set_pair", "threshold_min"])
    activity_out = add_semantic_columns(activity_out)
    activity_csv = "FigureS2a_activity_expansion_source.csv"
    activity_out.to_csv(out / activity_csv, index=False)
    records.append({
        "role": "S2a paired WT activity-set changes; São Paulo main frame",
        "source_file": "activity_paired_gains_verified/paired_activity_gains.csv",
        "source_sha256": sha256(root / "activity_paired_gains_verified/paired_activity_gains.csv"),
        "selected_rows": int(len(activity_out)),
        "output_file": activity_csv,
    })

    regional = prepare_regional(root, out)
    regional = add_semantic_columns(regional)
    regional = regional.sort_values(["scope", "threshold_min"])
    regional_csv = "FigureS2b_regional_sensitivity_source.csv"
    regional.to_csv(out / regional_csv, index=False)
    records.append({
        "role": "S2b paired WT regional-network changes; São Paulo validation frame",
        "source_file": "regional_transit_sensitivity/paired_regional_comparison.csv",
        "source_sha256": sha256(root / "regional_transit_sensitivity/paired_regional_comparison.csv"),
        "selected_rows": int(len(regional)),
        "output_file": regional_csv,
    })

    try:
        headway, headway_reason = load_headway(root)
    except ValueError as exc:
        headway, headway_reason = pd.DataFrame(), f"Headway output failed validation: {exc}"
    s2_ready = headway_reason == ""
    if s2_ready:
        headway = add_semantic_columns(headway)
        headway = headway.sort_values(["scope", "threshold_min"])
        headway_csv = "FigureS2c_headway_sensitivity_source.csv"
        headway.to_csv(out / headway_csv, index=False)
        records.append({
            "role": "S2c paired WT headway changes; São Paulo validation frame",
            "source_file": "planning_complete_headway_half/planning_paired_gains.csv",
            "source_sha256": sha256(root / "planning_complete_headway_half/planning_paired_gains.csv"),
            "selected_rows": int(len(headway)),
            "output_file": headway_csv,
        })
        records[-1]["curve_source_sha256"] = sha256(root / "planning_complete_headway_half/planning_coverage_curves.csv")
        fig, axes = plt.subplots(3, 1, figsize=(7.09, 5.35), sharex=False)
        activity_labels = {}
        for key in sorted(activity_out.activity_set_pair.astype(str).unique()):
            example = activity_out.loc[activity_out.activity_set_pair.astype(str).eq(key)].iloc[0]
            activity_labels[key] = semantic_scope_name("Sao Paulo", key, "sensitivity_annual_activity_expansion")
        pairs = [
            (axes[0], activity_out, "activity_set_pair", sorted(activity_out.activity_set_pair.astype(str).unique()),
             activity_labels,
             "gain_estimate_pp", "design_ci95_lower_pp", "design_ci95_upper_pp",
             "technical_bound_lower_pp", "technical_bound_upper_pp",
             "a  Annual-activity set expansion\nSão Paulo main frame, n = 1,100"),
            (axes[1], regional, "scope", list(SCOPE_ORDER),
             {key: semantic_scope_name("Sao Paulo", key, "baseline") for key in SCOPE_ORDER},
             "paired_change_pp", "paired_design_ci_lower_pp", "paired_design_ci_upper_pp",
             "technical_missing_bound_lower_pp", "technical_missing_bound_upper_pp",
             "b  Regional network versus baseline\nSão Paulo validation frame, n = 220"),
            (axes[2], headway, "scope", list(SCOPE_ORDER),
             {key: semantic_scope_name("Sao Paulo", key, "planning_half_headway") for key in SCOPE_ORDER},
             "headway_change_pp", "headway_ci_lower_pp", "headway_ci_upper_pp",
             "headway_technical_lower_pp", "headway_technical_upper_pp",
             "c  Nominal half-headway versus matched baseline\nSão Paulo validation frame, n = 220"),
        ]
        for ax, data, group_col, keys, labels, est, ci_lo, ci_hi, tech_lo, tech_hi, title in pairs:
            draw_forest_panel(ax, data, group_col, keys, labels, est, ci_lo, ci_hi, tech_lo, tech_hi, title)
        x_values = []
        for data, estimate, ci_lo, ci_hi, tech_lo, tech_hi in (
            (activity_out, "gain_estimate_pp", "design_ci95_lower_pp", "design_ci95_upper_pp",
             "technical_bound_lower_pp", "technical_bound_upper_pp"),
            (regional, "paired_change_pp", "paired_design_ci_lower_pp", "paired_design_ci_upper_pp",
             "technical_missing_bound_lower_pp", "technical_missing_bound_upper_pp"),
            (headway, "headway_change_pp", "headway_ci_lower_pp", "headway_ci_upper_pp",
             "headway_technical_lower_pp", "headway_technical_upper_pp"),
        ):
            for col in (estimate, ci_lo, ci_hi, tech_lo, tech_hi):
                x_values.extend(
                    float(value) for value in pd.to_numeric(data[col], errors="coerce").to_numpy()
                    if np.isfinite(value)
                )
        xlo, xhi = min(0.0, min(x_values)), max(0.0, max(x_values))
        xspan = max(xhi - xlo, 1.0)
        for ax in axes:
            ax.set_xlim(xlo - 0.06 * xspan, xhi + 0.06 * xspan)
        for ax in axes[:2]:
            ax.set_xlabel("")
        axes[0].set_ylabel("")
        axes[1].set_ylabel("Facility\nscenario")
        axes[2].set_ylabel("Facility\nscenario")
        handles = [
            Line2D([0], [0], color="#222222", linewidth=0.75, marker="o", markersize=3,
                   label="95% design CI and paired estimate"),
            Line2D([0], [0], color="#777777", linewidth=1.8, label="Technical bound"),
            *[Line2D([0], [0], color=THRESHOLD_COLORS[t], marker="o", linewidth=0, markersize=3,
                     label=f"{t} min") for t in THRESHOLDS],
        ]
        fig.suptitle("Paired WT coverage sensitivity at prespecified thresholds",
                     x=0.07, y=0.998, ha="left", fontsize=9, fontweight="bold")
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handlelength=1.5)
        fig.subplots_adjust(left=0.22, right=0.99, top=0.87, bottom=0.13, hspace=0.68)
        figures.extend(save_figure(fig, out, "FigureS2_paired_sensitivity"))

    s3_paths, s3_meta, s3_reason = prepare_s3(analysis_dir, out)
    s3_ready = bool(s3_paths)
    if s3_ready:
        figures.extend(s3_paths)
        records.append(s3_meta)

    write_captions(out, bool(s1_meta["technical_bounds_collapsed"]), s2_ready, s3_ready, s3_reason, args.provisional)
    status = {
        "S1": "READY_PROVISIONAL_M04_PENDING" if args.provisional else "READY",
        "S2": ("READY_PROVISIONAL_M04_PENDING" if args.provisional else "READY") if s2_ready else "PENDING_HEADWAY",
        "S2_status_reason": headway_reason,
        "S3": ("READY_PROVISIONAL_M04_PENDING" if args.provisional else "READY") if s3_ready else "PENDING_CLASSIFICATION",
        "S3_status_reason": s3_reason,
        "provisional": bool(args.provisional),
        "routing_status": "Current baseline route tables pending metro-inclusion correction." if args.provisional else "Final routing source.",
        "figures": figures,
        "sources": records,
        "headway_process": "Not managed or restarted by this script.",
        "producer_sha256": sha256(Path(__file__)),
    }
    (out / "supplement_figure_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(out / "supplement_source_manifest.csv", index=False)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
