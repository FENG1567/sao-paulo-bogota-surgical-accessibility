#!/usr/bin/env python3
"""Create the v13 five-figure, three-table publication bundle.

The normal run reads the frozen route-time file and precomputed analysis tables.
The pilot run uses only observed elapsed-route pilot rows, aggregates them in a
disposable mock root, and labels every output as pilot-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# Python backend contract: editable SVG text and consistent sans-serif fallback.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.facecolor"] = "white"


PALETTE = {
    "car": "#4D4D4D", "wt": "#0F4D92", "foot": "#767676",
    "cap1": "#3775BA", "cap2": "#6B9BC7", "cap3": "#A6C3DC",
    "R": "#4D4D4D", "E": "#0F4D92", "A": "#42949E",
    "both": "#4B8F63", "car_only": "#C47A2C", "wt_only": "#267C9C",
    "neither": "#B8B8B8", "indeterminate": "#8C6BB1",
}
CITIES = ("Sao Paulo", "Bogota")
SCOPES = ("R", "E", "A")
CAP_MODES = ("wt_cap1km", "wt_cap2km", "wt_cap3km")
THRESHOLDS = tuple(range(10, 61, 5))
DISPLAY_THRESHOLDS = (15, 30, 45)
MODE_LABELS = {
    "car": "Car", "foot": "Direct walking", "wt": "Transit + walking (WT)",
    "wt_cap1km": "WT, ≤1 km walk cap", "wt_cap2km": "WT, ≤2 km walk cap",
    "wt_cap3km": "WT, ≤3 km walk cap",
}
MAP_CLASSES = (
    ("both", "Both covered", PALETTE["both"]),
    ("car_only", "Car only", PALETTE["car_only"]),
    ("wt_only", "WT only", PALETTE["wt_only"]),
    ("neither", "Neither covered", PALETTE["neither"]),
    ("indeterminate", "Technically indeterminate", PALETTE["indeterminate"]),
)
BASELINE_NAMES = ("baseline", "baseline_fixed_departure", "baseline_official")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"uid": str}) if path.exists() else pd.DataFrame()


def truth(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def unknown_route_rows(df: pd.DataFrame) -> np.ndarray:
    unknown = df["finite_unknown"].map(truth).to_numpy(bool, copy=True) if "finite_unknown" in df else np.zeros(len(df), dtype=bool)
    if "status" in df:
        unknown |= df["status"].astype(str).str.lower().isin({"technical_error", "partial_error", "not_queried"}).to_numpy(bool)
    return unknown


def coverage_flags(df: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    times = pd.to_numeric(df.get("time_min", pd.Series(np.nan, index=df.index)), errors="coerce")
    lower = (times.notna() & times.le(threshold)).to_numpy(bool)
    return lower, lower | unknown_route_rows(df)


def design_estimate(df: pd.DataFrame, values: np.ndarray) -> dict[str, Any]:
    """Weighted ratio and stratified-SRS linearization used by the core analysis."""
    weights = pd.to_numeric(df.get("design_weight", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(float)
    y = np.asarray(values, dtype=float)
    good = np.isfinite(weights) & np.isfinite(y) & (weights > 0)
    d = df.loc[good].reset_index(drop=True)
    weights, y = weights[good], y[good]
    if len(weights) == 0 or weights.sum() <= 0:
        return {"estimate": np.nan, "se": np.nan, "lower": np.nan, "upper": np.nan,
                "n": int(len(weights)), "n_input": int(len(df)),
                "n_excluded_input": int(len(df)), "design": "unknown"}
    # Preserve exact constants: the weighted dot product can round a constant
    # outcome (especially all ones) slightly away from its exact value.
    mu = float(y[0]) if np.all(y == y[0]) else float(np.dot(weights, y) / weights.sum())
    var = 0.0
    census = True
    strata = d.get("stratum", pd.Series("unknown", index=d.index)).fillna("unknown").astype(str)
    for _, indices in d.groupby(strata, observed=True).indices.items():
        ix = np.asarray(indices, dtype=int)
        nh = len(ix)
        nh_values = pd.to_numeric(d.iloc[ix].get("N_h", pd.Series(nh, index=d.index[ix])), errors="coerce")
        nh_values = nh_values.loc[nh_values.notna()]
        Nh = float(nh_values.iloc[0]) if len(nh_values) else float(nh)
        if nh < Nh:
            census = False
            if nh < 2:
                return {"estimate": 100 * mu, "se": np.nan, "lower": np.nan, "upper": np.nan,
                        "n": int(len(weights)), "n_input": int(len(df)),
                        "n_excluded_input": int(len(df) - len(weights)), "design": "stratified_SRS"}
            pop = pd.to_numeric(d.iloc[ix].get("pop", pd.Series(1.0, index=d.index[ix])), errors="coerce").fillna(0).to_numpy(float)
            residual = pop * (y[ix] - mu)
            var += Nh**2 * (1 - nh / Nh) * float(np.var(residual, ddof=1)) / nh
    se = 100 * math.sqrt(max(0.0, var)) / float(weights.sum())
    suppress = census or se == 0 or not np.isfinite(se)
    return {
        "estimate": 100 * mu, "se": se,
        "lower": np.nan if suppress else 100 * mu - 1.96 * se,
        "upper": np.nan if suppress else 100 * mu + 1.96 * se,
        "n": int(len(weights)), "n_input": int(len(df)),
        "n_excluded_input": int(len(df) - len(weights)),
        "design": "census" if census else "stratified_SRS",
    }


def choose_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df.empty or "profile" not in df:
        return df.copy(), ""
    profiles = sorted(df.profile.loc[df.profile.notna()].astype(str).unique())
    for candidate in BASELINE_NAMES:
        if candidate in profiles:
            return df.loc[df.profile.astype(str).eq(candidate)].copy(), candidate
    starts = [p for p in profiles if p.lower().startswith("baseline")]
    if starts:
        return df.loc[df.profile.astype(str).eq(starts[0])].copy(), starts[0]
    if len(profiles) == 1:
        return df.loc[df.profile.astype(str).eq(profiles[0])].copy(), profiles[0]
    return df.iloc[0:0].copy(), ""


def suppress_design_intervals(df: pd.DataFrame, diagnostic: bool) -> pd.DataFrame:
    """Diagnostic pilot origins are not a probability sample; suppress inherited CIs."""
    if df.empty or not diagnostic:
        return df
    out = df.copy()
    for col in ("design_se", "design_ci_lower", "design_ci_upper"):
        if col in out.columns:
            out[col] = np.nan
    if "sampling_design" in out.columns:
        out["sampling_design"] = "diagnostic_pilot_not_probability_sample"
    out["design_intervals_suppressed"] = True
    return out


def paired_difference(times: pd.DataFrame, city: str, scope: str, mode_a: str, mode_b: str, threshold: float, profile: str = "") -> dict[str, Any]:
    d = times.copy()
    if not {"city", "scope", "mode", "uid"}.issubset(d.columns):
        return {}
    d = d.loc[d.city.astype(str).eq(city) & d.scope.astype(str).eq(scope)]
    if profile and "profile" in d:
        d = d.loc[d.profile.astype(str).eq(profile)]
    left = d.loc[d["mode"].astype(str).eq(mode_a)].drop_duplicates("uid").set_index("uid")
    right = d.loc[d["mode"].astype(str).eq(mode_b)].drop_duplicates("uid").set_index("uid")
    if left.empty or right.empty or set(left.index) != set(right.index):
        return {}
    right = right.loc[left.index]
    lo_a, hi_a = coverage_flags(left.reset_index(), threshold)
    lo_b, hi_b = coverage_flags(right.reset_index(), threshold)
    base = left.reset_index()
    est = design_estimate(base, lo_a.astype(float) - lo_b.astype(float))
    bound_lo = design_estimate(base, lo_a.astype(float) - hi_b.astype(float))
    bound_hi = design_estimate(base, hi_a.astype(float) - lo_b.astype(float))
    return {
        "city": city, "scope": scope, "mode_a": mode_a, "mode_b": mode_b,
        "profile": profile, "threshold": threshold, "difference_pp": est["estimate"],
        "design_se_pp": est["se"], "design_ci_lower_pp": est["lower"],
        "design_ci_upper_pp": est["upper"], "technical_bound_lower_pp": bound_lo["estimate"],
        "technical_bound_upper_pp": bound_hi["estimate"], "n_paired_origins": est["n"],
        "n_input_origins": est["n_input"], "n_excluded_for_estimate": est["n_excluded_input"],
        "sampling_design": est["design"], "technical_bounds_are_not_ci": True,
    }


def paired_difference_table(times: pd.DataFrame, profile: str, use_design_ci: bool = True) -> pd.DataFrame:
    rows = []
    for city in CITIES:
        for scope in SCOPES:
            for tau in THRESHOLDS:
                row = paired_difference(times, city, scope, "wt", "foot", tau, profile)
                if row:
                    row["comparison"] = "WT minus direct walking"
                    if not use_design_ci:
                        row["design_se_pp"] = row["design_ci_lower_pp"] = row["design_ci_upper_pp"] = np.nan
                        row["sampling_design"] = "diagnostic_pilot_not_probability_sample"
                    rows.append(row)
                for mode in CAP_MODES:
                    row = paired_difference(times, city, scope, mode, "wt", tau, profile)
                    if row:
                        row["comparison"] = MODE_LABELS[mode] + " minus unbounded WT"
                        if not use_design_ci:
                            row["design_se_pp"] = row["design_ci_lower_pp"] = row["design_ci_upper_pp"] = np.nan
                            row["sampling_design"] = "diagnostic_pilot_not_probability_sample"
                        rows.append(row)
    return pd.DataFrame(rows)


def read_frame(path: Path) -> pd.DataFrame:
    d = read_csv(path)
    for col in ("pop", "design_weight", "N_h", "n_h"):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def _label_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def scope_name(city: str, label: Any, profile: Any = "") -> str:
    """Reader-facing facility label; the source scope key is retained separately."""
    raw = _label_token(label)
    token = raw.casefold()
    context = _label_token(profile).casefold()
    city_key = _label_token(city).casefold()
    if not raw:
        return "Not applicable"
    if "->" in raw or "→" in raw:
        left, right = raw.replace("→", "->").split("->", 1)
        return f"Activity {left.strip()} → Activity {right.strip()}"
    if "∩" in raw:
        left, right = raw.split("∩", 1)
        return f"{scope_name(city, left.strip(), profile)} intersection {scope_name(city, right.strip(), profile)}"
    if "−" in raw and raw.count("−") == 1:
        left, right = raw.split("−", 1)
        return f"{scope_name(city, left.strip(), profile)} minus {scope_name(city, right.strip(), profile)}"
    if token in {"156", "161", "162"}:
        return f"Activity {raw}"
    if token == "p_public_admin":
        return "Public administration"
    if token in {"s_rd04_ge1", "s_rd04_ge1.0"}:
        return "Annual claims"
    if token == "s_rd04_ge100":
        return "Annual claims ≥100"
    if token == "r_municipal":
        return "Municipal registry subset"
    if city_key in {"sao paulo", "são paulo", "sp"}:
        if raw == "R":
            return "Registry (municipal subset)" if "municipal_legacy" in context else "Registry"
        if raw == "E":
            if "public_rd04" in context:
                return "Public administration"
            return "Equipment-SUS (municipal subset)" if "municipal_legacy" in context else "Equipment-SUS"
        if raw == "A":
            if "public_rd04" in context:
                return "Annual claims"
            if "rd04_ge100" in context:
                return "Annual claims ≥100"
            if "raw_activity" in context:
                return "Annual claims (all activity)"
            return "Retained historical activity-flag inventory (municipal subset)" if "municipal_legacy" in context else "Retained historical activity-flag inventory"
    if city_key in {"bogota", "bogotá", "bo"}:
        if raw == "R":
            if "classification_mean_30_epsg4326" in context:
                return "Registry-linked primary set (142 sites)"
            return "Historical-capacity registry set" if "historical_capacity" in context else "Registry"
        if raw == "E":
            return "Public-provider signal"
    return f"Project-defined facility scenario ({raw})"


def scope_definition(city: str, label: Any, profile: Any = "") -> str:
    """Human-readable operational definition paired with every raw scope key."""
    raw = _label_token(label)
    context = _label_token(profile).casefold()
    city_key = _label_token(city).casefold()
    if not raw:
        return "No facility scope applies to this population-frame or readiness row."
    if "->" in raw or "→" in raw:
        left, right = raw.replace("→", "->").split("->", 1)
        return f"Paired São Paulo annual-activity destination-set comparison: {left.strip()} sites versus {right.strip()} sites; these are activity-set IDs, not the main-figure R/E/A scopes."
    if "∩" in raw:
        left, right = raw.split("∩", 1)
        return f"Facility-set intersection: {scope_definition(city, left.strip(), profile)}; and {scope_definition(city, right.strip(), profile)}"
    if "−" in raw and raw.count("−") == 1:
        left, right = raw.split("−", 1)
        return f"Facility-scope contrast: {scope_definition(city, left.strip(), profile)} minus {scope_definition(city, right.strip(), profile)}"
    if city_key in {"sao paulo", "são paulo", "sp"}:
        if raw == "R":
            if "municipal_legacy" in context:
                return "Municipal-only São Paulo registry subset, using the same project registry criteria within the municipality."
            return "São Paulo registry scenario: active CNES records with project criteria for surgical-centre and hospital-care markers and facility type in {5, 7, 15, 20, 21}; not every surgical-centre record."
        if raw == "E" and "public_rd04" in context:
            return "Public-administration legal-nature scenario (115 R-qualified sites) paired with the annual group-04 claims sensitivity; distinct from the primary equipment marker."
        if raw == "E":
            suffix = " Municipal-only membership is used in this sensitivity." if "municipal_legacy" in context else ""
            return "Equipment-SUS scenario: São Paulo R-qualified sites with CNES code 0335 (laparoscope/video) recorded in at least one of 12 monthly 2024 equipment files; not patient eligibility or a complete surgical-capability inventory." + suffix
        if raw == "A" and "public_rd04" in context:
            return "Annual claims scenario: R-qualified sites with a 2024 annual group-04 surgical-claim record (135-site set); distinct from the retained historical activity flag."
        if raw == "A" and "rd04_ge100" in context:
            return "Higher-threshold annual-claims scenario: R-qualified sites with at least 100 2024 group-04 claim rows; distinct from the retained historical activity flag."
        if raw == "A" and "raw_activity" in context:
            return "Annual-claims sensitivity using all recorded 2024 group-04 activity under the raw activity frame; not the retained historical activity flag."
        if raw == "A" and "municipal_legacy" in context:
            return "Municipal-only retained historical activity-flag inventory subset; it is not the annual claims scenario."
        if raw == "A":
            return "Retained historical activity-flag inventory (118 R-qualified sites); it is not a current capacity or complete annual-claims inventory."
        if raw == "P_public_admin":
            return "Public-administration legal-nature destination scenario, separate from the primary equipment-SUS set."
        if raw == "S_RD04_ge1":
            return "R-qualified annual group-04 surgical-claims scenario (2024); distinct from the retained historical activity-flag inventory."
        if raw == "S_RD04_ge100":
            return "R-qualified 2024 annual group-04 surgical-claims scenario with a threshold of at least 100 claim rows."
        if raw == "R_municipal":
            return "Municipality-only registry subset; it is a separate spatial-membership sensitivity."
    if city_key in {"bogota", "bogotá", "bo"}:
        if raw == "R" and "historical_capacity" in context:
            return "Historical Bogotá capacity set: coordinate-linked 2022 surgical-capacity sites without requiring a match in the later provider register (158-site sensitivity)."
        if raw == "R":
            return "Bogotá registry-linked scenario: positive 2022 surgical-room capacity, match to the 2026 REPS site register, and accepted unique physical-site coordinates (142 sites)."
        if raw == "E":
            return "Bogotá public-provider signal: REPS-linked provider ESE=SI plus a consistent public legal-nature signal among accepted physical surgical sites (16 sites); not patient entitlement or proof of service use."
    return f"Project-defined scenario identified by raw key {raw!r}; retain this key and consult its named sensitivity/source manifest."


def profile_definition(profile: Any, city: Any = "", scenario: Any = "", period: Any = "") -> str:
    raw = _label_token(profile)
    scenario_raw = _label_token(scenario)
    period_raw = _label_token(period)
    key = (raw or period_raw).casefold()
    definitions = {
        "baseline_fixed_departure": "Primary fixed-departure route profile: 11 November 2026 at 07:30 local time; results remain conditional on the frozen routing inputs.",
        "baseline": "Primary baseline route profile on the corresponding fixed origin frame.",
        "validation_weekday": "Additional weekday request: 12 November 2026 at 08:30 local time; date and departure time change together.",
        "validation-weekday": "Additional weekday request: 12 November 2026 at 08:30 local time; date and departure time change together.",
        "validation_weekend": "Additional weekend request: 14 November 2026 at 12:00 local time; date and departure time change together.",
        "validation-weekend": "Additional weekend request: 14 November 2026 at 12:00 local time; date and departure time change together.",
        "sensitivity_bo_historical_capacity": "Bogotá historical-capacity destination-set sensitivity, using the same eligible origin frame and routing setup.",
        "sensitivity_annual_activity_expansion": "São Paulo 1,100-origin comparison of annual-activity destination sets containing 156, 161 or 162 sites.",
        "sensitivity_public_rd04": "São Paulo public-administration plus annual-claims destination sensitivity; registry R is retained, E denotes public administration and A denotes annual claims in this profile.",
        "sensitivity_municipal_legacy": "São Paulo municipality-only facility-membership sensitivity; scope labels refer to the municipal subsets.",
        "sensitivity_rd04_ge100": "São Paulo sensitivity using the ≥100 annual group-04 claim-row threshold; the equipment-SUS set remains distinct from annual claims.",
        "sensitivity_raw_activity": "São Paulo all-recorded annual-activity sensitivity; annual claims are not the retained historical activity flag.",
        "classification_mean_30_epsg4326": "Corrected primary Bogotá socioeconomic grouping: area-weighted mapped-block mean estrato in EPSG:4326, requiring >30% mapped cell area; values within 1e-12 of 2, 4 or 6 were snapped to the cut before classification. This changes group labels only; routing, overall coverage and D are unchanged.",
        "baseline_public_rd04_validation_frame": "São Paulo 220-origin validation-frame baseline using public-administration and annual-claims destinations.",
        "planning_half_headway": "Nominal half-headway planning scenario paired to the public-administration/annual-claims baseline on the same validation frame; not observed service frequency or passenger waiting time.",
        "main_factor_1_over_0_6": "Primary car-time scaling scenario; native road duration divided by 0.60.",
        "free_flow_factor_1_0": "Car-time sensitivity with factor 1.0 before the primary 0.60 scaling convention.",
        "intermediate_factor_1_over_0_8": "Intermediate car-time scaling sensitivity using factor 1/0.8.",
        "reconstructed_base_plus_5_min": "Reconstructed native car time with five minutes added before the primary scaling convention.",
        "main_adjusted_plus_5_min": "Primary adjusted car time with five additional minutes.",
    }
    text = definitions.get(key, "No separate routing profile applies to this frame/definition row." if not key else f"Scenario profile {raw or period_raw!r}; the raw identifier is retained in the adjacent profile/scenario field.")
    if scenario_raw and scenario_raw.casefold() != key:
        scenario_key = scenario_raw.casefold()
        if scenario_key in definitions:
            text += " Car-time scenario: " + definitions[scenario_key]
        elif scenario_raw:
            text += f" Scenario identifier {scenario_raw!r} is retained in the source row."
    return text


def enrich_source_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """Add semantic scope/profile columns while preserving every raw key/value."""
    out = df.copy()
    names, definitions, profiles = [], [], []
    for _, row in out.iterrows():
        city = _label_token(row.get("city", ""))
        profile = row.get("profile", row.get("analysis_profile", ""))
        scenario = row.get("scenario", row.get("activity_set_pair", ""))
        period = row.get("period", "")
        context = _label_token(profile) or _label_token(scenario) or _label_token(period)
        raw_scope = row.get("scope", row.get("scope_label", ""))
        if not _label_token(raw_scope) and _label_token(row.get("base_scope", "")) and _label_token(row.get("extra_scope", "")):
            base_n = _label_token(row.get("base_site_count", "")) or _label_token(row.get("base_scope", ""))
            extra_n = _label_token(row.get("extra_site_count", "")) or _label_token(row.get("extra_scope", ""))
            raw_scope = f"{base_n}->{extra_n}"
        if not _label_token(raw_scope) and _label_token(row.get("activity_set_pair", "")):
            raw_scope = row.get("activity_set_pair")
        if not _label_token(raw_scope) and _label_token(row.get("scope1", "")) and _label_token(row.get("scope2", "")):
            raw_scope = f"{row.get('scope1')}−{row.get('scope2')}"
        if not _label_token(raw_scope) and _label_token(row.get("base_scope", "")):
            raw_scope = row.get("base_scope")
        names.append(scope_name(city, raw_scope, context))
        definitions.append(scope_definition(city, raw_scope, context))
        profiles.append(profile_definition(profile, city, scenario, period))
    out["scope_name"] = names
    out["scope_definition"] = definitions
    out["profile_definition"] = profiles
    return out


def frame_summaries(root: Path, times: pd.DataFrame, pilot: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    facility_rows: list[dict[str, Any]] = []
    frame_info: dict[str, dict[str, Any]] = {}
    fixed = {
        "Sao Paulo": (1100, "Stratified probability sample"),
        "Bogota": (1057, "Complete grid-cell census"),
    }
    for city in CITIES:
        name = "main_sp.csv" if city == "Sao Paulo" else "main_bo.csv"
        frame = read_frame(root / "core" / "sets" / name)
        frozen_n, design = fixed[city]
        nrows = int(len(frame)) if not frame.empty else np.nan
        route_n = int(times.loc[times.city.astype(str).eq(city), "uid"].nunique()) if not times.empty and {"city", "uid"}.issubset(times.columns) else 0
        if frame.empty:
            unknown_n, unknown_pop = np.nan, np.nan
        else:
            stratum = frame.get("stratum", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
            unknown_mask = ~stratum.isin({"q1", "q2", "q3", "q4", "q5"}) if city == "Sao Paulo" else stratum.eq("unknown")
            unknown_n = int(unknown_mask.sum())
            unknown_pop = float(pd.to_numeric(frame.loc[unknown_mask, "pop"], errors="coerce").fillna(0).sum()) if "pop" in frame else np.nan
        frame_info[city] = {"frozen_n": frozen_n, "route_n": route_n, "design": design, "unknown_n": unknown_n, "unknown_pop": unknown_pop}
        income_note = (
            "Q1–Q5 are tract-income quintiles based on positive V06004 values, not equal-population person quintiles; unknown-income origins stay in the denominator and are sampled as a separate stratum."
            if city == "Sao Paulo"
            else "Residential estrato is a socioeconomic category, not monetary income; the unknown-estrato cells stay in the complete denominator."
        )
        rows.append({
            "record_type": "population_frame", "city": city, "scope": "",
            "measure": "fixed origin frame", "estimate": frozen_n, "unit": "origins",
            "source_rows": nrows, "route_origin_rows": route_n, "design": design,
            "unknown_income_origins": unknown_n, "unknown_income_population": unknown_pop,
            "unknown_income_population_share_pct": np.nan,
            "definition": income_note,
        })
    facilities = read_csv(root / "core" / "facilities_main.csv")
    if not facilities.empty and {"city", "R", "E", "A"}.issubset(facilities.columns):
        for city in CITIES:
            part = facilities.loc[facilities.city.astype(str).eq(city)]
            flag = {c: pd.to_numeric(part[c], errors="coerce").fillna(0).astype(bool) for c in ("R", "E", "A")}
            raw_e, raw_a = int(flag["E"].sum()), int(flag["A"].sum())
            if city == "Sao Paulo":
                scenarios = [("R", flag["R"]), ("E", flag["R"] & flag["E"]), ("A", flag["R"] & flag["A"])]
                flag_note = f"Core primary flags: raw E={raw_e}, A={raw_a}; source intersections are listed separately."
            else:
                scenarios = [("R", flag["R"]), ("E", flag["R"] & flag["E"])]
                flag_note = f"BO E public-provider signal; not equivalent to SP E. Raw BO E={raw_e}."
            for label, mask in scenarios:
                n = int(mask.sum())
                facility_rows.append({"city": city, "scope_label": label, "facility_count": n, "raw_E_count": raw_e, "raw_A_count": raw_a if city == "Sao Paulo" else np.nan})
                rows.append({
                    "record_type": "facility_definition", "city": city, "scope": label,
                    "measure": "facility count", "estimate": n, "unit": "facilities",
                    "source_rows": int(len(part)), "route_origin_rows": np.nan, "design": "scenario count",
                    "unknown_income_origins": np.nan, "unknown_income_population": np.nan,
                    "definition": scope_definition(city, label) + " " + flag_note,
                })
    # Preserve the separate expanded-facility sensitivity audit. Its labels
    # and intersections are not silently substituted for the frozen main arms.
    overlap_json = root / "facility" / "final_scope_overlap.json"
    overlap_csv = root / "facility" / "final_scope_overlap.csv"
    if overlap_json.exists():
        audit = json.loads(overlap_json.read_text(encoding="utf-8"))
        scope_counts = audit.get("scope_counts", {})
        for label, count in scope_counts.items():
            rows.append({
                "record_type": "facility_sensitivity_definition",
                "city": "Sao Paulo",
                "scope": label,
                "measure": "expanded facility provenance count",
                "estimate": count,
                "unit": "facilities",
                "source_rows": np.nan,
                "route_origin_rows": np.nan,
                "design": "separate sensitivity audit",
                "unknown_income_origins": np.nan,
                "unknown_income_population": np.nan,
                "definition": "Expanded SP facility provenance/sensitivity count; not substituted for the frozen primary route scenarios.",
            })
    overlap = read_csv(overlap_csv)
    if not overlap.empty and {"scope1", "scope2", "intersection"}.issubset(overlap.columns):
        for _, r in overlap.iterrows():
            rows.append({
                "record_type": "facility_overlap",
                "city": "Sao Paulo",
                "scope": str(r.scope1) + " ∩ " + str(r.scope2),
                "measure": "pairwise facility intersection",
                "estimate": r.get("intersection", np.nan),
                "unit": "facilities",
                "source_rows": np.nan,
                "route_origin_rows": np.nan,
                "design": "set overlap; not a nested funnel",
                "unknown_income_origins": np.nan,
                "unknown_income_population": np.nan,
                "definition": (
                    f"n1={r.get('n1')}; n2={r.get('n2')}; only1={r.get('only1')}; "
                    f"only2={r.get('only2')}; union={r.get('union')}; Jaccard={r.get('jaccard')}."
                ),
            })
    rows.append({
        "record_type": "municipal_legacy_sensitivity",
        "city": "Sao Paulo",
        "scope": "municipal_legacy",
        "measure": "raw provider universe",
        "estimate": 245,
        "unit": "facilities",
        "source_rows": np.nan,
        "route_origin_rows": np.nan,
        "design": "separate sensitivity frame",
        "unknown_income_origins": np.nan,
        "unknown_income_population": np.nan,
        "definition": "Raw flags E=51, A=61; municipal R=185; within R, E=50 and A=56. This legacy sensitivity is separate from wider registry provenance counts and primary route arms.",
    })
    if pilot:
        rows.append({
            "record_type": "pilot_status", "city": "Both", "scope": "",
            "measure": "route pilot", "estimate": int(times.uid.nunique()) if not times.empty and "uid" in times else 0,
            "unit": "origins", "source_rows": len(times),
            "route_origin_rows": int(times.uid.nunique()) if not times.empty and "uid" in times else 0,
            "design": "Observed elapsed-time pilot; not manuscript estimates",
            "unknown_income_origins": np.nan, "unknown_income_population": np.nan,
            "definition": "Code-path preview only; no synthetic observations were added.",
        })
    return pd.DataFrame(rows), pd.DataFrame(facility_rows), frame_info


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_bogota_classification_reanalysis(
    gradients: pd.DataFrame, table1: pd.DataFrame, analysis_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replace Bogotá group gradients and unknown-group metadata with the corrected primary classification."""
    gradient_path = analysis_dir / "classification_low_high_gradients.csv"
    population_path = analysis_dir / "classification_population_by_group.csv"
    summary_path = analysis_dir / "summary.json"
    for path in (gradient_path, population_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Corrected Bogotá classification source is absent: {path}")

    corrected = pd.read_csv(gradient_path)
    required = {"scenario", "scope", "mode", "threshold", "low_population", "high_population", "low_coverage", "high_coverage", "low_minus_high"}
    if not required.issubset(corrected.columns):
        raise ValueError(f"Corrected Bogotá gradients lack columns: {sorted(required - set(corrected.columns))}")
    corrected = corrected.loc[corrected.scenario.astype(str).eq("mean_30_epsg4326")].copy()
    corrected["threshold"] = pd.to_numeric(corrected.threshold, errors="coerce")
    expected_thresholds = set(THRESHOLDS)
    expected_keys = {(scope, mode, threshold) for scope in ("R", "E") for mode in ("car", "wt") for threshold in expected_thresholds}
    observed_keys = set(zip(corrected.scope.astype(str), corrected["mode"].astype(str), corrected.threshold))
    if observed_keys != expected_keys or corrected.duplicated(["scope", "mode", "threshold"]).any():
        raise ValueError("Corrected Bogotá primary gradients are not a complete unique R/E × car/WT × 10–60 min grid.")

    bogota = corrected.rename(columns={
        "low_minus_high": "gradient",
        "low_population": "population_low",
        "high_population": "population_high",
        "low_coverage": "coverage_low",
        "high_coverage": "coverage_high",
    })
    bogota["city"] = "Bogota"
    bogota["profile"] = "classification_mean_30_epsg4326"
    bogota["analysis_profile"] = "classification_mean_30_epsg4326"
    bogota["design_ci_lower"] = np.nan
    bogota["design_ci_upper"] = np.nan
    bogota["missing_bound_lower"] = np.nan
    bogota["missing_bound_upper"] = np.nan
    bogota["n"] = 1057
    bogota["sampling_design"] = "complete-frame census; no sampling confidence interval"
    bogota["classification_source"] = str(gradient_path)
    bogota["classification_source_sha256"] = file_sha256(gradient_path)
    original = gradients.loc[~gradients.city.astype(str).eq("Bogota")].copy() if "city" in gradients else gradients.copy()
    combined = pd.concat([original, bogota], ignore_index=True, sort=False)

    groups = pd.read_csv(population_path)
    required_groups = {"scenario", "group", "cells", "population", "population_share_pct"}
    if not required_groups.issubset(groups.columns):
        raise ValueError(f"Corrected Bogotá population table lacks columns: {sorted(required_groups - set(groups.columns))}")
    unknown = groups.loc[
        groups.scenario.astype(str).eq("mean_30_epsg4326") & groups.group.astype(str).eq("unknown")
    ]
    if len(unknown) != 1:
        raise ValueError("Corrected Bogotá population table must have one unknown-group row for the primary scenario.")
    u = unknown.iloc[0]
    t1 = table1.copy()
    for col in ("unknown_income_population_share_pct", "classification_scenario", "classification_source", "classification_source_sha256"):
        if col not in t1.columns:
            t1[col] = np.nan
        t1[col] = t1[col].astype("object")
    mask = t1.record_type.astype(str).eq("population_frame") & t1.city.astype(str).eq("Bogota")
    if int(mask.sum()) != 1:
        raise ValueError("Table 1 must contain exactly one Bogotá population-frame row before classification update.")
    t1.loc[mask, "unknown_income_origins"] = int(u.cells)
    t1.loc[mask, "unknown_income_population"] = float(u.population)
    t1.loc[mask, "unknown_income_population_share_pct"] = float(u.population_share_pct)
    t1.loc[mask, "classification_scenario"] = "mean_30_epsg4326"
    t1.loc[mask, "classification_source"] = str(population_path)
    t1.loc[mask, "classification_source_sha256"] = file_sha256(population_path)
    note = (
        " Corrected primary classification: area-weighted mapped-block mean estrato in EPSG:4326, "
        "mapped-area fraction >30%; values within 1e-12 of 2, 4 or 6 were snapped to the boundary. "
        "Two cells (40,902.0584 population) change groups versus the frozen legacy classification; "
        "overall coverage and D are unchanged."
    )
    t1.loc[mask, "definition"] = t1.loc[mask, "definition"].astype(str) + note

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "done":
        raise ValueError("Corrected Bogotá summary is not marked done.")
    repair = summary.get("numerical_boundary_repair", {})
    metadata = {
        "scenario": "mean_30_epsg4326",
        "classification_source": str(gradient_path),
        "classification_sha256": file_sha256(gradient_path),
        "population_source": str(population_path),
        "population_sha256": file_sha256(population_path),
        "summary_source": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "corrected_gradient_rows": int(len(bogota)),
        "corrected_unknown_cells": int(u.cells),
        "corrected_unknown_population": float(u.population),
        "corrected_unknown_population_share_pct": float(u.population_share_pct),
        "changed_cells_from_legacy": repair.get("changed_cells_from_legacy"),
        "changed_population_from_legacy": repair.get("changed_population_from_legacy"),
        "global_coverage_and_D_unchanged": repair.get("global_coverage_and_D_are_unchanged"),
        "census_sampling_ci": "not applicable",
    }
    return combined, t1, metadata


def classification_group_definition(group: Any) -> str:
    key = _label_token(group).casefold()
    return {
        "low": "Residential estrato group (0, 2].",
        "mid": "Residential estrato group (2, 4].",
        "high": "Residential estrato group (4, 6].",
        "unknown": "Unknown classification: the mapped-area requirement or a valid mapped estrato value was not met.",
        "all": "All Bogotá origins, including unknown group assignments.",
    }.get(key, f"Raw group key {group!r} retained from the classification output.")


def classification_scenario_definition(scenario: Any) -> str:
    raw = _label_token(scenario)
    key = raw.casefold()
    if key == "legacy_original30_frozen":
        return "Frozen original Bogotá classification retained as a historical comparison; it is not the corrected primary grouping."
    if key == "dominant_area_epsg4326":
        return "Dominant-area Bogotá estrato classification in EPSG:4326 as specified in frozen sensitivity plan v4."
    if key.startswith("mean_") and ("epsg4326" in key or "epsg3116" in key):
        threshold = key.split("_", 2)[1]
        crs = "EPSG:4326" if "epsg4326" in key else "EPSG:3116"
        return f"Area-weighted mapped-block mean estrato in {crs}, using the {threshold.replace('_', ' ')} mapped-area threshold."
    return f"Scenario key {raw!r} is retained; see the frozen Bogotá classification plan for its operational definition."


def attach_bogota_classification_groups(map_data: pd.DataFrame, analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = analysis_dir / "classification_origin_scenarios.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Corrected Bogotá origin classifications are absent: {path}")
    origins = pd.read_csv(path, dtype={"uid": str})
    required = {"uid", "scenario", "group", "base_group", "mapped_area_fraction", "mapped_area_mean"}
    if not required.issubset(origins.columns):
        raise ValueError(f"Bogotá origin classifications lack columns: {sorted(required - set(origins.columns))}")
    primary = origins.loc[origins.scenario.astype(str).eq("mean_30_epsg4326")].copy()
    if primary.uid.duplicated().any() or primary.uid.nunique() != 1057:
        raise ValueError("Corrected primary Bogotá origin classifications must contain 1,057 unique cells.")
    primary = primary.rename(columns={
        "group": "socioeconomic_group",
        "base_group": "legacy_original_group",
        "mapped_area_mean": "mapped_area_weighted_mean_estrato",
    })
    primary["classification_scenario"] = "mean_30_epsg4326"
    primary["classification_scenario_definition"] = primary.classification_scenario.map(classification_scenario_definition)
    primary["socioeconomic_group_definition"] = primary.socioeconomic_group.map(classification_group_definition)
    primary["classification_source"] = str(path)
    primary["classification_source_sha256"] = file_sha256(path)
    if map_data.empty:
        return map_data, primary
    result = map_data.copy()
    result["socioeconomic_group"] = ""
    sp_mask = result.city.astype(str).eq("Sao Paulo")
    if "stratum" in result:
        result.loc[sp_mask, "socioeconomic_group"] = result.loc[sp_mask, "stratum"].fillna("Unknown").astype(str)
    result.loc[sp_mask, "classification_scenario"] = "main_sp_fixed_strata"
    result.loc[sp_mask, "classification_scenario_definition"] = "Existing São Paulo tract-income quintiles Q1–Q5; unknown-income origins remain in the denominator."
    bo_mask = result.city.astype(str).eq("Bogota")
    if int(bo_mask.sum()) != 2 * 1057:
        raise ValueError("Figure 4 Bogotá origin-classification source does not contain the expected R/E paired maps.")
    add = primary[[
        "uid", "socioeconomic_group", "legacy_original_group", "mapped_area_fraction",
        "mapped_area_weighted_mean_estrato", "classification_scenario",
        "classification_scenario_definition", "socioeconomic_group_definition",
        "classification_source", "classification_source_sha256",
    ]]
    result = result.merge(add, how="left", on="uid", suffixes=("", "_corrected"), validate="many_to_one")
    for col in ("socioeconomic_group", "classification_scenario", "classification_scenario_definition"):
        corrected_col = f"{col}_corrected"
        if corrected_col in result:
            result.loc[bo_mask, col] = result.loc[bo_mask, corrected_col]
            result = result.drop(columns=corrected_col)
    result.loc[sp_mask, "socioeconomic_group_definition"] = result.loc[sp_mask, "socioeconomic_group"].map(
        lambda group: "Existing São Paulo tract-income quintile; Q1–Q5 are not equal-population person quintiles." if group not in ("Unknown", "") else "Unknown income group retained in the denominator."
    )
    result.loc[sp_mask, "legacy_original_group"] = ""
    result.loc[sp_mask, "mapped_area_fraction"] = np.nan
    result.loc[sp_mask, "mapped_area_weighted_mean_estrato"] = np.nan
    result.loc[sp_mask, "classification_source"] = "core/sets/main_sp.csv stratum"
    result.loc[sp_mask, "classification_source_sha256"] = ""
    result = result.drop(columns=[c for c in result.columns if c.endswith("_corrected")], errors="ignore")
    return result, primary


def prepare_bogota_classification_coverage(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "classification_coverage_curves.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Corrected Bogotá classification coverage source is absent: {path}")
    source = pd.read_csv(path)
    required = {"scenario", "facility_frame", "group", "scope", "mode", "threshold", "cells", "population", "covered_population", "coverage"}
    if not required.issubset(source.columns):
        raise ValueError(f"Corrected Bogotá classification coverage lacks columns: {sorted(required - set(source.columns))}")
    expected_scenarios = {
        "legacy_original30_frozen", "mean_30_epsg4326", "mean_50_epsg4326", "mean_80_epsg4326",
        "dominant_area_epsg4326", "mean_30_epsg3116", "mean_50_epsg3116", "mean_80_epsg3116",
    }
    threshold_values = pd.to_numeric(source.threshold, errors="raise")
    if not np.isfinite(threshold_values).all():
        raise ValueError("Bogota classification source contains a missing or nonfinite threshold")
    if set(source.scenario.astype(str)) != expected_scenarios or set(threshold_values) != set(THRESHOLDS):
        raise ValueError("Corrected Bogotá classification coverage must retain all eight schemes and 10–60 min thresholds.")
    source["city"] = "Bogota"
    source["profile"] = "classification_" + source.scenario.astype(str)
    source["classification_scenario_definition"] = source.scenario.map(classification_scenario_definition)
    source["socioeconomic_group_definition"] = source.group.map(classification_group_definition)
    source["source_file"] = str(path)
    source["source_sha256"] = file_sha256(path)
    return source


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.09, 1.025, label, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")


def placeholder(ax: plt.Axes, reason: str, label: str | None = None) -> None:
    ax.set_facecolor("#F5F5F5")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")
    if label:
        panel(ax, label)
    ax.text(0.5, 0.53, "INCOMPLETE", transform=ax.transAxes, ha="center", va="center", fontsize=9, fontweight="bold", color="#666666")
    ax.text(0.5, 0.40, reason, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color="#666666", wrap=True)


def not_applicable(ax: plt.Axes, reason: str, label: str | None = None) -> None:
    ax.set_facecolor("#FAFAFA")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#E0E0E0")
    if label:
        panel(ax, label)
    ax.text(0.5, 0.54, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=9, fontweight="bold", color="#767676")
    ax.text(0.5, 0.40, reason, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color="#767676", wrap=True)


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    svg_path = str(base) + ".svg"
    pdf_path = str(base) + ".pdf"
    png_path = str(base) + ".png"
    tiff_path = str(base) + ".tiff"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return [Path(p).name for p in (svg_path, pdf_path, png_path, tiff_path)]


def figure1(root: Path, out: Path, facility_data: pd.DataFrame, frame_info: dict[str, dict[str, Any]], times: pd.DataFrame, pilot: bool) -> list[str]:
    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.92, 1.35), wspace=0.36,
                            left=0.12,right=0.97,bottom=0.34,top=0.86)
    ax0, ax1 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    panel(ax0, "a")
    panel(ax1, "b")
    cities = ["Sao Paulo", "Bogota"]
    city_labels = {"Sao Paulo": "São Paulo", "Bogota": "Bogotá"}
    sizes = [frame_info.get(c, {}).get("frozen_n", 0) for c in cities]
    yy = np.arange(2)
    ax0.barh(yy, sizes, color=[PALETTE["E"], "#767676"], height=0.55, edgecolor="white")
    for y, n, city in zip(yy, sizes, cities):
        kind = "sampled origins" if city == "Sao Paulo" else "census origins"
        ax0.text(n + max(sizes) * 0.02, y, f"{n:,}  {kind}", va="center", fontsize=6.5)
    ax0.set_yticks(yy, [city_labels[c] for c in cities])
    ax0.invert_yaxis()
    ax0.set_xlim(0, max(sizes) * 1.65)
    ax0.set_xlabel("Origins", fontsize=6.5)
    ax0.set_title("Analysis-origin counts", loc="left", fontsize=7.5, fontweight="bold")
    ax0.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    ax0.spines["left"].set_visible(False)
    ax0.tick_params(axis="both", labelsize=6.2, length=2.2)
    if facility_data.empty:
        placeholder(ax1, "core facility flags are unavailable", "b")
    else:
        display = facility_data.copy()
        display = display.loc[display.scope_label.isin(["R", "E", "A"])]
        sp_label = {"R": "São Paulo\nRegistry", "E": "São Paulo\nEquipment-SUS", "A": "São Paulo\nRetained historical\nactivity-flag inventory"}
        bo_label = {"R": "Bogotá\nRegistry", "E": "Bogotá\nPublic-provider signal"}
        display["label"] = [
            (sp_label if c == "Sao Paulo" else bo_label).get(s, c + " · " + s)
            for c, s in zip(display.city, display.scope_label)
        ]
        y = np.arange(len(display))
        val = display.facility_count.to_numpy(float)
        colors = [PALETTE.get(x, "#767676") for x in display.scope_label]
        ax1.barh(y, val, color=colors, height=0.62, edgecolor="white")
        for yi, v in zip(y, val):
            ax1.text(v + max(1, val.max()) * 0.025, yi, f"{int(v):,}", va="center", fontsize=6.5)
        ax1.set_yticks(y, display.label)
        ax1.invert_yaxis()
        ax1.set_xlim(0, max(val.max(), 1) * 1.3)
        ax1.set_xlabel("Facilities", fontsize=6.5)
        ax1.set_title("Scenario counts (not a nested funnel)", loc="left", fontsize=7.5, fontweight="bold")
        ax1.grid(axis="x", color="#E8E8E8", linewidth=0.5)
        ax1.tick_params(axis="both", labelsize=6.2, length=2.2)
    fig.suptitle(("PILOT ONLY PREVIEW · " if pilot else "") + "Analysis origins and facility definitions", x=0.02, y=0.995, ha="left", fontsize=8.5, fontweight="bold")
    fig.text(
        0.02, 0.018,
        "São Paulo: Registry=339, Equipment-SUS=89, retained historical activity-flag inventory=118; the latter two overlap at 78 sites but are not nested.\n"
        "SP: 1,100 sampled origins (stratified probability sample); BO: 1,057 complete-grid census origins.\n"
        "Equipment-SUS is not patient eligibility; Bogotá's 16-site Public-provider signal is a different construct. Full overlaps are in Table 1.",
        fontsize=6.0, va="bottom",
    )
    return save_figure(fig, out / "Figure1_population_facilities")


def line_with_bounds(ax: plt.Axes, df: pd.DataFrame, color: str, label: str, linestyle: Any = "-", linewidth: float = 1.15, markersize: float = 2.4, show_ci: bool = True) -> None:
    d = df.sort_values("threshold")
    x = pd.to_numeric(d.threshold, errors="coerce").to_numpy(float)
    y = pd.to_numeric(d.coverage_lower, errors="coerce").to_numpy(float)
    upper = pd.to_numeric(d.get("coverage_upper", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    ci_lo = pd.to_numeric(d.get("design_ci_lower", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    ci_hi = pd.to_numeric(d.get("design_ci_upper", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    ax.plot(x, y, color=color, linestyle=linestyle, marker="o", ms=markersize, lw=linewidth, label=label)
    ci_mask = show_ci & np.isin(x, DISPLAY_THRESHOLDS) & np.isfinite(ci_lo) & np.isfinite(ci_hi) & np.isfinite(y)
    if ci_mask.any():
        ax.errorbar(
            x[ci_mask], y[ci_mask],
            yerr=np.vstack((np.maximum(0, y[ci_mask] - ci_lo[ci_mask]), np.maximum(0, ci_hi[ci_mask] - y[ci_mask]))),
            fmt="none", ecolor=color, elinewidth=0.65, capsize=1.6, alpha=0.8,
        )
    if np.isfinite(upper).any() and np.nanmax(np.abs(upper - y)) > 1e-8:
        ax.plot(x, upper, color=color, linestyle=(0, (1, 2)), lw=0.75, alpha=0.68)


def figure2(curves: pd.DataFrame, contrasts: pd.DataFrame, out: Path, pilot: bool) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.95), sharex=True, sharey="row")
    scope_styles = {"R": "-", "E": "--", "A": ":"}
    technical_style = "-."
    cities = ("Sao Paulo", "Bogota")
    city_labels = {
        "Sao Paulo": ("São Paulo", "probability sample"),
        "Bogota": ("Bogotá", "census frame"),
    }
    contrast_scopes = ("E", "A")

    if curves.empty:
        for ax, label in zip(axes.flat, "abcd"):
            placeholder(ax, "main coverage curves are unavailable", label)
        fig.suptitle(("PILOT ONLY PREVIEW · " if pilot else "") + "Mode, facility scope and threshold", x=0.01, y=0.99, ha="left", fontsize=8.5, fontweight="bold")
        fig.subplots_adjust(left=0.095, right=0.99, bottom=0.18, top=0.80, hspace=0.53, wspace=0.20)
        return save_figure(fig, out / "Figure2_mode_scope_threshold")

    def finite_values(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
        values: list[np.ndarray] = []
        for column in columns:
            if column in df:
                current = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
                values.append(current[np.isfinite(current)])
        return np.concatenate(values) if values else np.array([], dtype=float)

    display_curves = curves.loc[
        curves.city.astype(str).isin(cities)
        & curves.scope.astype(str).isin(scope_styles)
        & curves["mode"].astype(str).isin(("car", "wt"))
        & ~(curves.city.astype(str).eq("Bogota") & curves.scope.astype(str).eq("A"))
    ].copy()
    coverage_bounds_differ = False
    if {"coverage_lower", "coverage_upper"}.issubset(display_curves.columns):
        lower = pd.to_numeric(display_curves.coverage_lower, errors="coerce").to_numpy(float)
        upper = pd.to_numeric(display_curves.coverage_upper, errors="coerce").to_numpy(float)
        valid = np.isfinite(lower) & np.isfinite(upper)
        coverage_bounds_differ = bool(valid.any() and not np.allclose(lower[valid], upper[valid], rtol=0, atol=1e-8))

    if not contrasts.empty:
        contrasts = contrasts.loc[
            contrasts.city.astype(str).isin(cities)
            & contrasts.get("scope1", pd.Series("R", index=contrasts.index)).astype(str).eq("R")
            & contrasts.get("scope2", pd.Series(index=contrasts.index, dtype=str)).astype(str).isin(contrast_scopes)
        ].copy()
    contrast_technical_bounds_differ = False
    if {"contrast", "missing_bound_lower", "missing_bound_upper"}.issubset(contrasts.columns):
        point = pd.to_numeric(contrasts.contrast, errors="coerce").to_numpy(float)
        bound_lo = pd.to_numeric(contrasts.missing_bound_lower, errors="coerce").to_numpy(float)
        bound_hi = pd.to_numeric(contrasts.missing_bound_upper, errors="coerce").to_numpy(float)
        valid_lo = np.isfinite(point) & np.isfinite(bound_lo)
        valid_hi = np.isfinite(point) & np.isfinite(bound_hi)
        contrast_technical_bounds_differ = bool(
            (valid_lo.any() and not np.allclose(point[valid_lo], bound_lo[valid_lo], rtol=0, atol=1e-8))
            or (valid_hi.any() and not np.allclose(point[valid_hi], bound_hi[valid_hi], rtol=0, atol=1e-8))
        )

    letters = iter("abcd")
    for ci, city in enumerate(cities):
        ax = axes[0, ci]
        panel(ax, next(letters))
        city_curves = display_curves.loc[display_curves.city.astype(str).eq(city)]
        n_values = pd.to_numeric(city_curves.get("n", pd.Series(dtype=float)), errors="coerce")
        n_values = n_values.loc[n_values.notna()]
        n_text = f"n={int(n_values.iloc[0]):,}" if len(n_values) else "n unavailable"
        city_name, frame_label = city_labels[city]
        if pilot:
            title = f"{city_name} · pilot subset ({n_text})"
        else:
            title = f"{city_name} · {frame_label} ({n_text})"
        ax.set_title(title, loc="left", fontsize=6.7, fontweight="bold", pad=4)

        for scope in SCOPES:
            if city == "Bogota" and scope == "A":
                continue
            scope_curves = city_curves.loc[city_curves.scope.astype(str).eq(scope)]
            for mode, color in (("car", PALETTE["car"]), ("wt", PALETTE["wt"])):
                d = scope_curves.loc[scope_curves["mode"].astype(str).eq(mode)].sort_values("threshold")
                if d.empty:
                    continue
                x = pd.to_numeric(d.threshold, errors="coerce").to_numpy(float)
                y = pd.to_numeric(d.coverage_lower, errors="coerce").to_numpy(float)
                ax.plot(x, y, color=color, linestyle=scope_styles[scope], marker="o", ms=2.2, lw=1.0, zorder=3)

                upper = pd.to_numeric(d.get("coverage_upper", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
                upper_differs = np.isfinite(upper) & np.isfinite(y) & (np.abs(upper - y) > 1e-8)
                if upper_differs.any():
                    ax.plot(x, upper, color=color, linestyle=technical_style, lw=0.58, alpha=0.72, zorder=2)

                ci_lo = pd.to_numeric(d.get("design_ci_lower", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
                ci_hi = pd.to_numeric(d.get("design_ci_upper", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
                ci_mask = np.zeros(x.shape, dtype=bool)
                if not pilot and city == "Sao Paulo":
                    ci_mask = np.isin(x, DISPLAY_THRESHOLDS) & np.isfinite(ci_lo) & np.isfinite(ci_hi) & np.isfinite(y)
                if ci_mask.any():
                    ax.errorbar(
                        x[ci_mask], y[ci_mask],
                        yerr=np.vstack((np.maximum(0, y[ci_mask] - ci_lo[ci_mask]), np.maximum(0, ci_hi[ci_mask] - y[ci_mask]))),
                        fmt="none", ecolor=color, elinewidth=0.58, capsize=1.3, alpha=0.84, zorder=4,
                    )

        ax.set_ylim(0, 103)
        ax.set_xlim(9, 61)
        ax.set_xticks((15, 30, 45, 60))
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_yticks((0, 50, 100))
        ax.grid(color="#E8E8E8", linewidth=0.45)
        ax.set_axisbelow(True)
        if ci == 0:
            ax.set_ylabel("Coverage (%)", fontsize=6.3)

    contrast_values = finite_values(
        contrasts,
        ("contrast", "design_ci_lower", "design_ci_upper", "missing_bound_lower", "missing_bound_upper"),
    )
    if len(contrast_values):
        span = max(float(np.max(contrast_values) - np.min(contrast_values)), 1.0)
        pad = max(1.0, span * 0.04)
        y_min = min(-30.0, math.floor((float(np.min(contrast_values)) - pad) / 5) * 5)
        y_max = max(45.0, math.ceil((float(np.max(contrast_values)) + pad) / 5) * 5)
    else:
        y_min, y_max = -30.0, 45.0

    for ci, city in enumerate(cities):
        ax = axes[1, ci]
        panel(ax, next(letters))
        city_name = city_labels[city][0]
        ax.set_title(f"{city_name} · descriptive additive contrasts", loc="left", fontsize=6.7, fontweight="bold", pad=4)
        city_contrasts = contrasts.loc[contrasts.city.astype(str).eq(city)] if not contrasts.empty else pd.DataFrame()
        for scope2 in contrast_scopes:
            d = city_contrasts.loc[city_contrasts.scope2.astype(str).eq(scope2)].sort_values("threshold") if not city_contrasts.empty else pd.DataFrame()
            if d.empty:
                continue
            x = pd.to_numeric(d.threshold, errors="coerce").to_numpy(float)
            y = pd.to_numeric(d.contrast, errors="coerce").to_numpy(float)
            ax.plot(x, y, color=PALETTE[scope2], marker="o", ms=2.25, lw=1.05, zorder=3)

            bound_lo = pd.to_numeric(d.get("missing_bound_lower", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
            bound_hi = pd.to_numeric(d.get("missing_bound_upper", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
            lower_differs = np.isfinite(bound_lo) & np.isfinite(y) & (np.abs(bound_lo - y) > 1e-8)
            upper_differs = np.isfinite(bound_hi) & np.isfinite(y) & (np.abs(bound_hi - y) > 1e-8)
            if lower_differs.any():
                ax.plot(x, bound_lo, color=PALETTE[scope2], linestyle=technical_style, lw=0.58, alpha=0.72, zorder=2)
            if upper_differs.any():
                ax.plot(x, bound_hi, color=PALETTE[scope2], linestyle=technical_style, lw=0.58, alpha=0.72, zorder=2)

            ci_lo = pd.to_numeric(d.get("design_ci_lower", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
            ci_hi = pd.to_numeric(d.get("design_ci_upper", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
            ci_mask = np.zeros(x.shape, dtype=bool)
            if not pilot and city == "Sao Paulo":
                ci_mask = np.isin(x, DISPLAY_THRESHOLDS) & np.isfinite(ci_lo) & np.isfinite(ci_hi) & np.isfinite(y)
            if ci_mask.any():
                ax.errorbar(
                    x[ci_mask], y[ci_mask],
                    yerr=np.vstack((np.maximum(0, y[ci_mask] - ci_lo[ci_mask]), np.maximum(0, ci_hi[ci_mask] - y[ci_mask]))),
                    fmt="none", ecolor=PALETTE[scope2], elinewidth=0.62, capsize=1.4, alpha=0.88, zorder=4,
                )

        ax.axhline(0, color="#767676", lw=0.65, linestyle="--", zorder=1)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(9, 61)
        ax.set_xticks((15, 30, 45, 60))
        ax.set_xlabel("Threshold (min)", fontsize=6.3, labelpad=2)
        ax.grid(color="#E8E8E8", linewidth=0.45)
        ax.set_axisbelow(True)
        if ci == 0:
            ax.set_ylabel("D (percentage points)", fontsize=6.3)

    mode_scope_handles = [
        Line2D([0], [0], color=PALETTE["car"], lw=1.25, marker="o", ms=2.3, label="Car"),
        Line2D([0], [0], color=PALETTE["wt"], lw=1.25, marker="o", ms=2.3, label="WT (transit + walking)"),
        Line2D([0], [0], color="#3F3F3F", lw=1.1, linestyle=scope_styles["R"], label="Registry (city-specific)"),
        Line2D([0], [0], color="#3F3F3F", lw=1.1, linestyle=scope_styles["E"], label="Equipment-SUS (São Paulo)"),
        Line2D([0], [0], color="#3F3F3F", lw=1.1, linestyle=scope_styles["E"], label="Public-provider signal (Bogotá)"),
        Line2D([0], [0], color="#3F3F3F", lw=1.1, linestyle=scope_styles["A"], label="Retained historical activity-flag inventory (São Paulo)"),
    ]
    fig.legend(
        handles=mode_scope_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=3, fontsize=6.0, handlelength=1.4, handletextpad=0.28, columnspacing=0.68,
        borderaxespad=0.0,
    )

    contrast_handles = [
        Line2D([0], [0], color=PALETTE["E"], lw=1.1, marker="o", ms=2.2,
               label="D: Registry − Equipment-SUS (SP) / Public-provider signal (Bogotá)"),
        Line2D([0], [0], color=PALETTE["A"], lw=1.1, marker="o", ms=2.2,
               label="D: Registry − Legacy activity (SP only)"),
    ]
    if not pilot:
        contrast_handles.append(Line2D([0], [0], color="#555555", lw=0.65, marker="|", ms=4.5, label="95% design CI at 15/30/45 (SP only)"))
    if coverage_bounds_differ or contrast_technical_bounds_differ:
        contrast_handles.append(Line2D([0], [0], color="#555555", lw=0.65, linestyle=technical_style, label="Technical bound (not CI)"))
    fig.legend(
        handles=contrast_handles, loc="upper center", bbox_to_anchor=(0.5, 0.858),
        ncol=2, fontsize=5.4, handlelength=1.4, handletextpad=0.28, columnspacing=0.68,
        borderaxespad=0.0,
    )

    notes = ["D is descriptive, not causal."]
    if coverage_bounds_differ:
        notes.append("Dash-dot lines show technical upper bounds, not CIs.")
    else:
        notes.append("Technical bounds coincide with coverage.")
    if contrast_technical_bounds_differ and not coverage_bounds_differ:
        notes.append("Dash-dot D lines show technical bounds, not CIs.")
    fig.suptitle(("PILOT ONLY PREVIEW · " if pilot else "") + "Mode, facility scope and threshold", x=0.01, y=0.995, ha="left", fontsize=8.5, fontweight="bold")
    fig.text(0.01, 0.025, " ".join(notes), fontsize=6.2, va="bottom")
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.18, top=0.76, hspace=0.53, wspace=0.20)
    return save_figure(fig, out / "Figure2_mode_scope_threshold")


def figure3(curves: pd.DataFrame, differences: pd.DataFrame, out: Path, pilot: bool) -> list[str]:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.7), sharex=True, sharey=True)
    styles = {
        "foot": (PALETTE["foot"], (0, (3, 1.5))),
        "wt": (PALETTE["wt"], "-"),
        "wt_cap1km": (PALETTE["cap1"], "--"),
        "wt_cap2km": (PALETTE["cap2"], "-."),
        "wt_cap3km": (PALETTE["cap3"], ":"),
    }
    display_curves = curves.loc[
        curves.city.astype(str).isin(CITIES)
        & curves.scope.astype(str).isin(SCOPES)
        & curves["mode"].astype(str).isin(styles)
        & ~(curves.city.astype(str).eq("Bogota") & curves.scope.astype(str).eq("A"))
    ] if not curves.empty else pd.DataFrame()
    technical_bounds_differ = False
    if {"coverage_lower", "coverage_upper"}.issubset(display_curves.columns):
        lower = pd.to_numeric(display_curves.coverage_lower, errors="coerce").to_numpy(float)
        upper = pd.to_numeric(display_curves.coverage_upper, errors="coerce").to_numpy(float)
        valid = np.isfinite(lower) & np.isfinite(upper)
        technical_bounds_differ = bool(valid.any() and not np.allclose(lower[valid], upper[valid], rtol=0, atol=1e-8))
    handles = [Line2D([0], [0], color=c, linestyle=s, lw=1.15, label=MODE_LABELS[m]) for m, (c, s) in styles.items()]

    def city_n(city: str) -> str:
        if curves.empty or "n" not in curves:
            return "n unavailable"
        n_values = pd.to_numeric(curves.loc[curves.city.astype(str).eq(city), "n"], errors="coerce").to_numpy(float)
        n_values = n_values[np.isfinite(n_values)]
        return f"{int(n_values[0]):,}" if len(n_values) else "n unavailable"

    letters = iter("abcdef")
    for ri, scope in enumerate(SCOPES):
        for ci, city in enumerate(CITIES):
            ax = axes[ri, ci]
            letter = next(letters)
            if city == "Bogota" and scope == "A":
                ax.set_axis_off()
                panel(ax, letter)
                ax.text(0.02, 0.94, "Mode key and definitions", transform=ax.transAxes, ha="left", va="top", fontsize=6.6, fontweight="bold")
                ax.legend(
                    handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.78),
                    ncol=2, fontsize=6.1, handlelength=1.5, handletextpad=0.35,
                    columnspacing=0.7, labelspacing=0.55, borderaxespad=0.0,
                )
                if pilot:
                    sample_note = f"Pilot subset: SP n={city_n('Sao Paulo')}; BO n={city_n('Bogota')}."
                    ci_note = "Pilot design intervals are suppressed."
                    delta_note = "Panel annotations give paired 30-min WT−walking Δ."
                else:
                    sample_note = f"SP probability sample n={city_n('Sao Paulo')}; BO census n={city_n('Bogota')}."
                    ci_note = "Whiskers: 95% design CI at 15/30/45 min (SP only)."
                    delta_note = "Panel annotations give paired 30-min WT−walking Δ; brackets are SP CIs."
                notes = [
                    sample_note,
                    ci_note,
                    delta_note,
                    "All walking caps filter the same returned-route pool.",
                    "Technical bounds coincide with coverage." if not technical_bounds_differ else "Dotted lines show technical upper bounds, not CIs.",
                ]
                ax.text(0.02, 0.34, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=6.0, linespacing=1.15)
                continue
            panel(ax, letter)
            sub = curves.loc[curves.city.astype(str).eq(city) & curves.scope.astype(str).eq(scope)] if not curves.empty else pd.DataFrame()
            if sub.empty:
                placeholder(ax, "main curves are unavailable")
                continue
            for mode, (color, ls) in styles.items():
                d = sub.loc[sub["mode"].astype(str).eq(mode)]
                if not d.empty:
                    line_with_bounds(ax, d, color, MODE_LABELS[mode], linestyle=ls, linewidth=1.0, markersize=2.0, show_ci=not pilot)
            d30 = differences.loc[
                differences.city.astype(str).eq(city)
                & differences.scope.astype(str).eq(scope)
                & differences.comparison.astype(str).eq("WT minus direct walking")
                & pd.to_numeric(differences.threshold, errors="coerce").eq(30)
            ] if not differences.empty else pd.DataFrame()
            if not d30.empty:
                r = d30.iloc[0]
                value = pd.to_numeric(pd.Series([r.get("difference_pp")]), errors="coerce").iloc[0]
                lo = pd.to_numeric(pd.Series([r.get("design_ci_lower_pp")]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([r.get("design_ci_upper_pp")]), errors="coerce").iloc[0]
                text = f"Δ30 WT−foot {value:+.1f} pp"
                if (not pilot) and np.isfinite(lo) and np.isfinite(hi):
                    text += f" [{lo:+.1f}, {hi:+.1f}]"
                ax.text(0.98, 0.05, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=PALETTE["wt"])
            city_label = "São Paulo" if city == "Sao Paulo" else "Bogotá"
            ax.set_title(f"{city_label} · {scope_name(city, scope)}", loc="left", fontsize=6.8, fontweight="bold", pad=3)
            ax.set_ylim(0, 103)
            ax.set_xlim(9, 61)
            ax.set_xticks((15, 30, 45, 60))
            ax.tick_params(axis="both", labelsize=6.0, length=2.0)
            ax.grid(color="#E8E8E8", linewidth=0.45)
            if ri == 2 and ci == 0:
                ax.set_xlabel("Threshold (min)", fontsize=6.3, labelpad=2)
            elif ri == 1 and ci == 1:
                ax.set_xlabel("Threshold (min)", fontsize=6.3, labelpad=2)
                ax.tick_params(axis="x", labelbottom=True)
    fig.supylabel("Coverage (%)", x=0.018, fontsize=6.5)
    fig.suptitle(("PILOT ONLY PREVIEW · " if pilot else "") + "Direct walking, WT net coverage and common-pool walking caps", x=0.02, y=0.995, ha="left", fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.085, top=0.90, hspace=0.42, wspace=0.21)
    return save_figure(fig, out / "Figure3_walk_caps")


def map_classifications(times: pd.DataFrame, profile: str, threshold: float = 30.0) -> pd.DataFrame:
    need = {"city", "uid", "scope", "mode", "lat", "lon"}
    if times.empty or not need.issubset(times.columns):
        return pd.DataFrame()
    d = times.loc[times.profile.astype(str).eq(profile)].copy() if profile and "profile" in times else times.copy()
    rows = []
    for city in CITIES:
        for scope in SCOPES:
            part = d.loc[d.city.astype(str).eq(city) & d.scope.astype(str).eq(scope)]
            car = part.loc[part["mode"].astype(str).eq("car")].drop_duplicates("uid").set_index("uid")
            wt = part.loc[part["mode"].astype(str).eq("wt")].drop_duplicates("uid").set_index("uid")
            if car.empty or wt.empty or set(car.index) != set(wt.index):
                continue
            wt = wt.loc[car.index]
            clo, chi = coverage_flags(car.reset_index(), threshold)
            wlo, whi = coverage_flags(wt.reset_index(), threshold)
            classes = []
            for c0, c1, w0, w1 in zip(clo, chi, wlo, whi):
                if c0 and w0:
                    classes.append("both")
                elif c0 and not w1:
                    classes.append("car_only")
                elif w0 and not c1:
                    classes.append("wt_only")
                elif not c1 and not w1:
                    classes.append("neither")
                else:
                    classes.append("indeterminate")
            base = car.reset_index()
            keep = [c for c in ("uid", "lat", "lon", "stratum", "pop", "design_weight") if c in base.columns]
            result = base[keep].copy()
            result["city"], result["scope"] = city, scope
            result["threshold"], result["classification"] = threshold, classes
            rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def figure4(gradients: pd.DataFrame, map_data: pd.DataFrame, out: Path, pilot: bool) -> list[str]:
    fig = plt.figure(figsize=(7.2, 6.55))
    gs = fig.add_gridspec(4, 2, height_ratios=(1.0, 1.1, 1.1, 1.1), hspace=0.52, wspace=0.30,
                         left=0.10,right=0.98,bottom=0.06,top=0.87)
    letters = iter("abcdefgh")
    city_labels = {"Sao Paulo": "São Paulo", "Bogota": "Bogotá"}
    gradient_data = gradients.loc[
        gradients.city.astype(str).isin(CITIES)
        & pd.to_numeric(gradients.threshold, errors="coerce").eq(30)
        & gradients.scope.astype(str).isin(SCOPES)
        & gradients["mode"].astype(str).isin(("car", "wt"))
        & ~(gradients.city.astype(str).eq("Bogota") & gradients.scope.astype(str).eq("A"))
    ].copy() if not gradients.empty and {"city", "threshold", "scope", "mode"}.issubset(gradients.columns) else pd.DataFrame()

    gradient_values: list[np.ndarray] = []
    for col in ("gradient", "design_ci_lower", "design_ci_upper", "missing_bound_lower", "missing_bound_upper"):
        if col in gradient_data:
            vals = pd.to_numeric(gradient_data[col], errors="coerce").to_numpy(float)
            gradient_values.append(vals[np.isfinite(vals)])
    all_gradient_values = np.concatenate(gradient_values) if gradient_values else np.array([], dtype=float)
    if len(all_gradient_values):
        spread = max(float(np.max(all_gradient_values) - np.min(all_gradient_values)), 1.0)
        pad = max(2.0, spread * 0.05)
        gradient_ymin = math.floor((float(np.min(all_gradient_values)) - pad) / 5) * 5
        gradient_ymax = math.ceil((float(np.max(all_gradient_values)) + pad) / 5) * 5
    else:
        gradient_ymin, gradient_ymax = -40.0, 40.0

    technical_bounds_differ = False
    required_bound_cols = {"gradient", "missing_bound_lower", "missing_bound_upper"}
    if required_bound_cols.issubset(gradient_data.columns):
        estimate = pd.to_numeric(gradient_data.gradient, errors="coerce").to_numpy(float)
        bound_lo = pd.to_numeric(gradient_data.missing_bound_lower, errors="coerce").to_numpy(float)
        bound_hi = pd.to_numeric(gradient_data.missing_bound_upper, errors="coerce").to_numpy(float)
        valid_lo = np.isfinite(estimate) & np.isfinite(bound_lo)
        valid_hi = np.isfinite(estimate) & np.isfinite(bound_hi)
        technical_bounds_differ = bool(
            (valid_lo.any() and not np.allclose(estimate[valid_lo], bound_lo[valid_lo], rtol=0, atol=1e-8))
            or (valid_hi.any() and not np.allclose(estimate[valid_hi], bound_hi[valid_hi], rtol=0, atol=1e-8))
        )

    for ci, city in enumerate(CITIES):
        ax = fig.add_subplot(gs[0, ci])
        panel(ax, next(letters))
        sub = gradient_data.loc[gradient_data.city.astype(str).eq(city)]
        if sub.empty:
            placeholder(ax, "30-min socioeconomic-gradient estimates are unavailable")
            ax.set_title(city_labels[city], loc="left", fontsize=6.8, fontweight="bold")
            continue
        xmap = {scope: i for i, scope in enumerate(SCOPES)}
        for mode, off, color in (("car", -0.09, PALETTE["car"]), ("wt", 0.09, PALETTE["wt"])):
            dm = sub.loc[sub["mode"].astype(str).eq(mode)]
            for _, row in dm.iterrows():
                scope = str(row.get("scope", ""))
                if scope not in xmap:
                    continue
                x = xmap[scope] + off
                est = pd.to_numeric(pd.Series([row.get("gradient")]), errors="coerce").iloc[0]
                lo = pd.to_numeric(pd.Series([row.get("design_ci_lower")]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([row.get("design_ci_upper")]), errors="coerce").iloc[0]
                blo = pd.to_numeric(pd.Series([row.get("missing_bound_lower")]), errors="coerce").iloc[0]
                bhi = pd.to_numeric(pd.Series([row.get("missing_bound_upper")]), errors="coerce").iloc[0]
                if technical_bounds_differ and np.isfinite(est) and np.isfinite(blo) and np.isfinite(bhi) and (abs(est - blo) > 1e-8 or abs(est - bhi) > 1e-8):
                    ax.vlines(x, blo, bhi, color=color, alpha=0.3, lw=2.4, zorder=1)
                if (not pilot) and np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar(x, est, yerr=[[max(0, est-lo)], [max(0, hi-est)]], fmt="o", ms=3.1, color=color, capsize=1.8, lw=0.75, zorder=2)
                else:
                    ax.plot(x, est, "o", ms=3.1, color=color, zorder=2)
        ax.axhline(0, color="#777777", lw=0.55, linestyle="--")
        if city == "Sao Paulo":
            ax.set_xticks(range(3), ["Registry", "Equipment-SUS", "Retained historical\nactivity-flag inventory"], fontsize=5.5)
            ax.set_xlim(-0.4, 2.4)
        else:
            ax.set_xticks((0, 1), ["Registry", "Public-provider\nsignal"], fontsize=5.5)
            ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(gradient_ymin, gradient_ymax)
        ax.set_title(city_labels[city], loc="left", fontsize=6.8, fontweight="bold", pad=3)
        if ci == 0:
            ax.set_ylabel("Lower − higher group (pp)", fontsize=6.1)
        ax.tick_params(axis="both", labelsize=6.0, length=2.0)
        ax.grid(axis="y", color="#E8E8E8", lw=0.45)

    gradient_handles = [
        Line2D([0], [0], color=PALETTE["car"], marker="o", lw=0, ms=3.2, label="Car"),
        Line2D([0], [0], color=PALETTE["wt"], marker="o", lw=0, ms=3.2, label="WT"),
    ]
    if not pilot:
        gradient_handles.append(Line2D([0], [0], color="#555555", lw=0.65, marker="|", ms=4.5, label="95% design CI at 30 min (SP only)"))
    if technical_bounds_differ:
        gradient_handles.append(Patch(facecolor="#888888", alpha=0.3, label="Technical bound (not CI)"))
    fig.legend(
        handles=gradient_handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
        ncol=len(gradient_handles), fontsize=6.2, handlelength=1.3, handletextpad=0.35,
        columnspacing=0.9, borderaxespad=0.0,
    )

    def map_origin_n(city: str) -> str:
        if map_data.empty or not {"city", "uid"}.issubset(map_data.columns):
            return "n unavailable"
        return f"{int(map_data.loc[map_data.city.astype(str).eq(city), 'uid'].nunique()):,}"

    for ri, scope in enumerate(SCOPES, start=1):
        for ci, city in enumerate(CITIES):
            letter = next(letters)
            ax = fig.add_subplot(gs[ri, ci])
            if city == "Bogota" and scope == "A":
                ax.set_axis_off()
                panel(ax, letter)
                ax.text(0.02, 0.94, "Map classes and definitions", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, fontweight="bold")
                map_handles = [Line2D([0], [0], marker="o", linestyle="", color=color, markersize=3.7, label=label) for _, label, color in MAP_CLASSES]
                ax.legend(
                    handles=map_handles, loc="upper left", bbox_to_anchor=(0.02, 0.82),
                    ncol=2, fontsize=5.9, handlelength=0.8, handletextpad=0.3,
                    columnspacing=0.6, labelspacing=0.28, borderaxespad=0.0,
                )
                if pilot:
                    map_note = f"Pilot map subset: SP n={map_origin_n('Sao Paulo')}; BO n={map_origin_n('Bogota')}."
                else:
                    map_note = f"30-min maps include all paired origins: SP n={map_origin_n('Sao Paulo')}; BO n={map_origin_n('Bogota')}."
                notes = [
                    map_note,
                    "Unknown SES remains in each denominator; no unsampled area is filled.",
                    "Q1−Q5: SP tract-income groups; low−high: BO residential estrato.",
                    "Technical bounds coincide with estimates." if not technical_bounds_differ else "Thick bars show technical bounds, not CIs.",
                ]
                ax.text(0.02, 0.34, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=5.9, linespacing=1.15)
                continue

            panel(ax, letter)
            sub = map_data.loc[map_data.city.astype(str).eq(city) & map_data.scope.astype(str).eq(scope)] if not map_data.empty else pd.DataFrame()
            if sub.empty:
                placeholder(ax, "paired 30-min origin classifications are unavailable")
            else:
                for key, label, color in MAP_CLASSES:
                    pts = sub.loc[sub.classification.astype(str).eq(key)]
                    if not pts.empty:
                        ax.scatter(pts.lon, pts.lat, s=7 if len(pts) < 300 else 4, color=color, alpha=0.72, linewidths=0, rasterized=True, label=label)
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xlabel("Longitude", fontsize=6.0, labelpad=1.5)
                ax.set_ylabel("Latitude", fontsize=6.0, labelpad=1.5)
                ax.tick_params(labelsize=5.8, length=1.8, pad=1.5)
                ax.grid(color="#EEEEEE", linewidth=0.35)
            ax.set_title(f"{city_labels[city]} · {scope_name(city, scope)} · 30 min", loc="left", fontsize=6.3, fontweight="bold", pad=3)
    fig.suptitle(("PILOT ONLY PREVIEW · " if pilot else "") + "Socioeconomic differences and 30-min car/WT origin classification", x=0.02, y=0.995, ha="left", fontsize=8.5, fontweight="bold")
    return save_figure(fig, out / "Figure4_income_geography")


def validation_paths(root: Path, explicit_out: Path | None) -> tuple[Path | None, Path | None]:
    if explicit_out:
        outdir = explicit_out
        if (outdir / "validation_sameframe_base").exists():
            base = outdir / "validation_sameframe_base"
        elif outdir.name == "validation_sameframe_base":
            base, outdir = outdir, outdir.parent
        else:
            base = outdir / "validation_sameframe_base"
        return (outdir, base) if (outdir / "validation_sameframe_times.csv").exists() or base.exists() else (None, None)
    search = root / "extensions"
    matches = list(search.rglob("validation_sameframe_times.csv")) if search.exists() else []
    if not matches:
        return None, None
    times_path = sorted(matches, key=lambda p: str(p))[0]
    outdir = times_path.parent
    base = outdir / "validation_sameframe_base"
    return outdir, base if base.exists() else None


def profile_token(profiles: list[str], name: str) -> str | None:
    lowered = {p: p.lower() for p in profiles}
    exact = [p for p in profiles if lowered[p] in {name, "validation_" + name, "validation-" + name}]
    if exact:
        return sorted(exact)[0]
    contains = [p for p in profiles if name in lowered[p]]
    return sorted(contains)[0] if contains else None


def validate_sameframe(times: pd.DataFrame, base_curves: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    required = {"city", "uid", "scope", "mode", "profile"}
    if times.empty or not required.issubset(times.columns):
        return False, "same-frame time file or required columns are missing", pd.DataFrame()
    profiles = sorted(times.profile.loc[times.profile.notna()].astype(str).unique())
    chosen = {name: profile_token(profiles, name) for name in ("baseline", "weekday", "weekend")}
    missing = [name for name, value in chosen.items() if not value]
    if missing:
        return False, "same-frame route profiles absent: " + ", ".join(missing), pd.DataFrame()
    group_cols = ["city", "scope", "mode"]
    keysets = {}
    for name, prof in chosen.items():
        d = times.loc[times.profile.astype(str).eq(prof)]
        keysets[name] = set(tuple(row) for row in d[group_cols].drop_duplicates().itertuples(index=False, name=None))
    if any(keysets[name] != keysets["baseline"] for name in ("weekday", "weekend")):
        return False, "city/scope/mode arms differ across period profiles", pd.DataFrame()
    meta = [c for c in ("stratum", "pop", "design_weight", "N_h", "n_h", "inclusion_prob", "lat", "lon") if c in times.columns]
    records = []
    for key in sorted(keysets["baseline"]):
        city, scope, mode = key
        subset = times.loc[
            times.city.astype(str).eq(city)
            & times.scope.astype(str).eq(scope)
            & times["mode"].astype(str).eq(mode)
            & times.profile.astype(str).isin(chosen.values())
        ]
        frames = {}
        for label, prof in chosen.items():
            part = subset.loc[subset.profile.astype(str).eq(prof)].copy()
            if part.uid.duplicated().any():
                return False, f"duplicate origin arms for {city}/{scope}/{mode}/{label}", pd.DataFrame()
            frames[label] = part.set_index("uid").sort_index()
        base_uids = set(frames["baseline"].index.astype(str))
        if any(set(frames[label].index.astype(str)) != base_uids for label in ("weekday", "weekend")):
            return False, f"origin UID set differs for {city}/{scope}/{mode}", pd.DataFrame()
        for label in ("weekday", "weekend"):
            for col in meta:
                a, b = frames["baseline"][col], frames[label][col]
                if col == "stratum":
                    equal = a.fillna("").astype(str).equals(b.fillna("").astype(str))
                else:
                    av = pd.to_numeric(a, errors="coerce").to_numpy(float)
                    bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
                    equal = bool(np.allclose(av, bv, equal_nan=True, rtol=0, atol=1e-10))
                if not equal:
                    return False, f"denominator metadata differs for {city}/{scope}/{mode}/{label}", pd.DataFrame()
        for label in ("baseline", "weekday", "weekend"):
            part = frames[label].reset_index()
            lower, upper = coverage_flags(part, 30)
            lo = design_estimate(part, lower.astype(float))
            hi = design_estimate(part, upper.astype(float))
            records.append({
                "city": city, "scope": scope, "mode": mode, "period": label,
                "profile": chosen[label], "threshold": 30, "coverage_lower": lo["estimate"],
                "coverage_upper": hi["estimate"], "design_se": lo["se"],
                "design_ci_lower": lo["lower"], "design_ci_upper": lo["upper"],
                "n": lo["n"], "sampling_design": lo["design"],
                "sampled_population": float(pd.to_numeric(part.get("pop", pd.Series(0, index=part.index)), errors="coerce").fillna(0).sum()),
                "estimated_population": float(pd.to_numeric(part.get("design_weight", pd.Series(0, index=part.index)), errors="coerce").fillna(0).sum()),
                "same_uid_frame_verified": True, "denominator_metadata_verified": True,
            })
    result = pd.DataFrame(records)
    if result.empty:
        return False, "no complete city × scope × mode validation arms", result
    if base_curves.empty:
        return False, "unified-denominator validation_sameframe_base curves are missing", pd.DataFrame()
    base, _ = choose_baseline(base_curves)
    base30 = base.loc[pd.to_numeric(base.threshold, errors="coerce").eq(30)]
    for idx, row in result.loc[result.period.eq("baseline")].iterrows():
        match = base30.loc[
            base30.city.astype(str).eq(str(row.city))
            & base30.scope.astype(str).eq(str(row.scope))
            & base30["mode"].astype(str).eq(str(row["mode"]))
        ]
        if match.empty:
            return False, "same-frame baseline summary misses a time-file arm", pd.DataFrame()
        val = pd.to_numeric(match.coverage_lower, errors="coerce").iloc[0]
        if not np.isclose(val, row.coverage_lower, rtol=0, atol=0.05, equal_nan=True):
            return False, "same-frame baseline summary does not match its time rows", pd.DataFrame()
        result.loc[idx, "coverage_lower"] = val
        for col in ("coverage_upper", "design_se", "design_ci_lower", "design_ci_upper", "n", "sampling_design"):
            if col in match.columns:
                result.loc[idx, col] = match.iloc[0].get(col, result.loc[idx, col])
        result.loc[idx, "baseline_from_sameframe_analysis"] = True
    return True, "UID sets and denominator metadata match for baseline, weekday and weekend", result


def coverage_curves_path(source: Path) -> Path:
    """Accept either a sensitivity directory or its explicit coverage CSV."""
    return source / "coverage_curves.csv" if source.is_dir() else source


COMMONPOOL_PROFILE_SCOPES = {
    "Sao Paulo": {
        "baseline_fixed_departure": {"R", "E", "A"},
        "sensitivity_public_rd04": {"R", "E", "A"},
        "sensitivity_municipal_legacy": {"R", "E", "A"},
        "sensitivity_raw_activity": {"R", "E", "A"},
        "sensitivity_rd04_ge100": {"R", "E", "A"},
    },
    "Bogota": {
        "baseline_fixed_departure": {"R", "E"},
        "sensitivity_bo_historical_capacity": {"R", "E"},
    },
}


def commonpool_arm_semantics(city: str, scope: str, profile: str) -> tuple[str, int, str]:
    """Return a scope-specific reader label, facility count and definition.

    Labels are deliberately resolved from (city, scope, profile) together:
    public-administration and annual-claims meanings attach to E and A in the
    public/claims profile only, never to every row with that scope code.
    """
    arms: dict[tuple[str, str, str], tuple[str, int, str]] = {
        ("Sao Paulo", "R", "baseline_fixed_departure"): ("Registry · primary (339)", 339, "Primary São Paulo R-qualified registry set."),
        ("Sao Paulo", "E", "baseline_fixed_departure"): ("Equipment-SUS · primary (89)", 89, "R-qualified sites with CNES code 0335 recorded in at least one monthly 2024 equipment file."),
        ("Sao Paulo", "A", "baseline_fixed_departure"): ("Legacy activity · primary (118)", 118, "Retained historical activity-flag inventory; not a current annual-claims set."),
        ("Sao Paulo", "R", "sensitivity_public_rd04"): ("Registry · public/claims (339)", 339, "R-qualified registry set in the public-administration/annual-claims profile."),
        ("Sao Paulo", "E", "sensitivity_public_rd04"): ("Public administration (115)", 115, "R-qualified public-administration legal-nature set."),
        ("Sao Paulo", "A", "sensitivity_public_rd04"): ("Annual claims ≥1 (135)", 135, "R-qualified sites with at least one 2024 annual group-04 claim row."),
        ("Sao Paulo", "R", "sensitivity_rd04_ge100"): ("Registry · claims ≥100 (339)", 339, "R-qualified registry set in the higher-threshold annual-claims profile."),
        ("Sao Paulo", "E", "sensitivity_rd04_ge100"): ("Equipment-SUS · claims ≥100 arm (89)", 89, "R-qualified equipment-SUS marker set; unchanged within this claims-threshold profile."),
        ("Sao Paulo", "A", "sensitivity_rd04_ge100"): ("Claims ≥100 (133)", 133, "Annual group-04 claims set requiring at least 100 claim rows."),
        ("Sao Paulo", "R", "sensitivity_raw_activity"): ("Registry · all-activity arm (339)", 339, "R-qualified registry set in the all-recorded-activity profile."),
        ("Sao Paulo", "E", "sensitivity_raw_activity"): ("Equipment-SUS · raw profile (90)", 90, "Legacy equipment flag under the raw activity source definition."),
        ("Sao Paulo", "A", "sensitivity_raw_activity"): ("All activity (156; 21 outside R)", 156, "All recorded annual group-04 activity; includes 21 sites outside R."),
        ("Sao Paulo", "R", "sensitivity_municipal_legacy"): ("Municipal registry (185)", 185, "Municipal-only subset of the project registry scenario."),
        ("Sao Paulo", "E", "sensitivity_municipal_legacy"): ("Municipal Equipment-SUS (50)", 50, "Municipal-only equipment-SUS subset within R."),
        ("Sao Paulo", "A", "sensitivity_municipal_legacy"): ("Municipal legacy activity (56)", 56, "Municipal-only subset of the retained historical activity flag."),
        ("Bogota", "R", "baseline_fixed_departure"): ("Registry-linked primary (142)", 142, "142 coordinate-linked registry sites in the primary Bogotá frame."),
        ("Bogota", "E", "baseline_fixed_departure"): ("Public-provider signal · primary (16)", 16, "Public-provider signal among primary accepted physical sites; not patient entitlement."),
        ("Bogota", "R", "sensitivity_bo_historical_capacity"): ("Historical capacity (158)", 158, "158 coordinate-linked 2022 capacity sites, including sites without later provider-register matches."),
        ("Bogota", "E", "sensitivity_bo_historical_capacity"): ("Public-provider signal · historical (16)", 16, "Public-provider signal in the historical-capacity profile; not patient entitlement."),
    }
    key = (str(city), str(scope), str(profile))
    if key not in arms:
        raise ValueError(f"No audited common-pool label/membership definition for {key}")
    return arms[key]


def expected_commonpool_keys() -> set[tuple[str, str, str, str]]:
    return {
        (city, scope, mode, profile)
        for city, profiles in COMMONPOOL_PROFILE_SCOPES.items()
        for profile, scopes in profiles.items()
        for scope in scopes
        for mode in ("car", "wt")
    }


def sensitivity_points(sensitivity_source: Path | None) -> pd.DataFrame:
    if sensitivity_source is None:
        return pd.DataFrame()
    path = coverage_curves_path(sensitivity_source)
    if not path.is_file():
        raise FileNotFoundError(f"Common-pool coverage source is missing: {path}")
    d = read_csv(path)
    required = {"threshold", "city", "scope", "mode", "profile", "coverage_lower", "coverage_upper", "n"}
    if d.empty or not required.issubset(d.columns):
        raise ValueError(f"Common-pool coverage source lacks required columns: {sorted(required - set(d.columns))}")
    d = d.loc[pd.to_numeric(d.threshold, errors="coerce").eq(30) & d["mode"].astype(str).isin(["car", "wt"])].copy()
    keys = ["city", "scope", "mode", "profile"]
    if d.duplicated(keys + ["threshold"]).any():
        raise ValueError("Common-pool source has duplicate 30-min city/scope/mode/profile points")
    observed = set(zip(d.city.astype(str), d.scope.astype(str), d["mode"].astype(str), d.profile.astype(str)))
    expected = expected_commonpool_keys()
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(f"Common-pool 30-min car/WT arms differ from the audited schema; missing={missing}; unexpected={unexpected}")
    lower = pd.to_numeric(d.coverage_lower, errors="coerce").to_numpy(float)
    upper = pd.to_numeric(d.coverage_upper, errors="coerce").to_numpy(float)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all() and ((lower >= 0) & (lower <= upper) & (upper <= 100)).all()):
        raise ValueError("Common-pool 30-min coverage bounds are missing or outside [0, 100]")
    if pd.to_numeric(d.n, errors="coerce").isna().any():
        raise ValueError("Common-pool 30-min rows must retain origin-frame n")
    semantic_rows = [commonpool_arm_semantics(row.city, row.scope, row.profile) for row in d.itertuples()]
    d["arm_label"] = [row[0] for row in semantic_rows]
    d["destination_membership_n"] = [row[1] for row in semantic_rows]
    d["scope_name"] = [row[0].split(" (")[0].split(" · ")[0] for row in semantic_rows]
    d["arm_definition"] = [row[2] for row in semantic_rows]
    d["source_file"] = str(path)
    d["source_sha256"] = file_sha256(path)
    d["source_profile_key"] = d.profile.astype(str)
    return d


def validate_figure5_completeness(same_ok: bool, same_data: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    if not same_ok or same_data.empty:
        raise ValueError("Figure 5 cannot be marked ready without validated same-frame period data")
    timing = same_data.loc[same_data["mode"].astype(str).isin(["car", "wt"])].copy()
    timing_key = ["city", "scope", "mode", "period"]
    expected_timing = {
        (city, scope, mode, period)
        for city, scopes in (("Sao Paulo", ("R", "E", "A")), ("Bogota", ("R", "E")))
        for scope in scopes
        for mode in ("car", "wt")
        for period in ("baseline", "weekday", "weekend")
    }
    observed_timing = set(zip(timing.city.astype(str), timing.scope.astype(str), timing["mode"].astype(str), timing.period.astype(str)))
    if timing.duplicated(timing_key).any() or observed_timing != expected_timing:
        raise ValueError("Figure 5 period panels require every unique SP R/E/A and BO R/E car/WT arm for baseline, weekday and weekend")
    for city, expected_n in (("Sao Paulo", 220), ("Bogota", 200)):
        city_n = pd.to_numeric(timing.loc[timing.city.astype(str).eq(city), "n"], errors="coerce")
        if city_n.isna().any() or not city_n.eq(expected_n).all():
            raise ValueError(f"Figure 5 {city} validation panels must retain n={expected_n} across all modes/scopes/periods")
    if sensitivity.empty:
        raise ValueError("Figure 5 facility panels require the complete common-pool source")
    keys = ["city", "scope", "mode", "profile"]
    if set(zip(sensitivity.city.astype(str), sensitivity.scope.astype(str), sensitivity["mode"].astype(str), sensitivity.profile.astype(str))) != expected_commonpool_keys():
        raise ValueError("Figure 5 facility panels are missing a canonical city/scope/mode/profile arm")
    if sensitivity.duplicated(keys + ["threshold"]).any() or sensitivity.arm_label.isna().any() or sensitivity.arm_definition.isna().any():
        raise ValueError("Figure 5 facility points require unique keys and explicit membership labels/definitions")
    if sensitivity.destination_membership_n.isna().any() or sensitivity.source_file.isna().any() or sensitivity.source_sha256.isna().any():
        raise ValueError("Figure 5 facility source rows must retain membership counts and traceable source metadata")


def figure5(same_ok: bool, reason: str, same_data: pd.DataFrame, sensitivity: pd.DataFrame, out: Path, pilot: bool, require_complete: bool = False) -> tuple[list[str], list[dict[str, str]], pd.DataFrame]:
    if require_complete:
        validate_figure5_completeness(same_ok, same_data, sensitivity)
    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(4, 2, height_ratios=(1, 1, 1, 3), hspace=0.38, wspace=0.42)
    letters = iter("abcdefgh")
    readiness: list[dict[str, str]] = []
    if not same_ok:
        ax = fig.add_subplot(gs[0:3, :])
        placeholder(ax, reason or "same-frame baseline/weekday/weekend data are not ready", next(letters))
        readiness.append({"component": "same-frame timing comparison", "status": "INCOMPLETE", "reason": reason})
    else:
        profile_colors = {"baseline": "#4D4D4D", "weekday": "#0F4D92", "weekend": "#42949E"}
        for ri, scope in enumerate(SCOPES):
            for ci, city in enumerate(CITIES):
                ax = fig.add_subplot(gs[ri, ci])
                panel(ax, next(letters))
                sub = same_data.loc[
                    same_data.city.astype(str).eq(city)
                    & same_data.scope.astype(str).eq(scope)
                    & same_data["mode"].astype(str).isin(["car", "wt"])
                ]
                if sub.empty:
                    if city == "Bogota" and scope == "A":
                        ax.set_axis_off()
                        ax.text(0.02, 0.94, "Reading the panels", transform=ax.transAxes,
                                fontsize=6.8, fontweight="bold", va="top")
                        handles = [
                            Line2D([0], [0], marker="o", color="#555555", lw=0, label="Car"),
                            Line2D([0], [0], marker="s", color="#555555", lw=0, label="Walking + transit (WT)"),
                            Line2D([0], [0], marker="o", color=profile_colors["baseline"], lw=0, label="Baseline"),
                            Line2D([0], [0], marker="o", color=profile_colors["weekday"], lw=0, label="Weekday"),
                            Line2D([0], [0], marker="o", color=profile_colors["weekend"], lw=0, label="Weekend"),
                        ]
                        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.82),
                                  fontsize=5.8, ncol=2, labelspacing=0.35, columnspacing=0.8,
                                  borderaxespad=0, handletextpad=0.35)
                        ax.text(0.02, 0.32,
                                "Validation frames: SP n=220; BO n=200.\nIntervals are pointwise 95% design CIs.\nBO has no primary legacy-activity arm.",
                                transform=ax.transAxes, va="top", fontsize=5.8, linespacing=1.2)
                    else:
                        placeholder(ax, "validated period comparison absent")
                    continue
                xpos = {"baseline": 0, "weekday": 1, "weekend": 2}
                for mode, off, marker in (("car", -0.07, "o"), ("wt", 0.07, "s")):
                    for period in ("baseline", "weekday", "weekend"):
                        dm = sub.loc[sub["mode"].astype(str).eq(mode) & sub.period.astype(str).eq(period)]
                        if dm.empty:
                            continue
                        r = dm.iloc[0]
                        x, y, color = xpos[period] + off, float(r.coverage_lower), profile_colors[period]
                        lo = pd.to_numeric(pd.Series([r.get("design_ci_lower")]), errors="coerce").iloc[0]
                        hi = pd.to_numeric(pd.Series([r.get("design_ci_upper")]), errors="coerce").iloc[0]
                        if np.isfinite(lo) and np.isfinite(hi):
                            ax.errorbar(x, y, yerr=[[max(0, y-lo)], [max(0, hi-y)]], fmt=marker, ms=3, color=color, capsize=1.8, lw=0.75)
                        else:
                            ax.plot(x, y, marker=marker, ms=3, color=color)
                ax.set_xticks([0, 1, 2], ["Baseline", "Weekday", "Weekend"])
                ax.set_ylim(-2, 102)
                ax.grid(axis="y", color="#E8E8E8", lw=0.45)
                city_display = "São Paulo" if city == "Sao Paulo" else "Bogotá"
                short_scope = {"Sao Paulo": {"R": "Registry", "E": "Equipment-SUS", "A": "Legacy activity"},
                               "Bogota": {"R": "Registry", "E": "Public-provider signal"}}[city][scope]
                ax.set_title(f"{city_display} · {short_scope}", loc="left", fontsize=6.8, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel("Covered at 30 min (%)")
        readiness.append({"component": "same-frame timing comparison", "status": "READY", "reason": reason})
    if sensitivity.empty:
        for ci, city in enumerate(CITIES):
            ax = fig.add_subplot(gs[3, ci])
            placeholder(ax, "results_sensitivity has no 30-min car/WT coverage rows", next(letters))
        readiness.append({"component": "facility/definition sensitivity", "status": "INCOMPLETE", "reason": "No sensitivity coverage rows were available."})
    else:
        for ci, city in enumerate(CITIES):
            ax = fig.add_subplot(gs[3, ci])
            panel(ax, next(letters))
            sub = sensitivity.loc[sensitivity.city.astype(str).eq(city)].copy()
            if sub.empty:
                placeholder(ax, "no sensitivity arms for this city")
                continue
            labels = sorted(sub.arm_label.astype(str).unique())
            label_map = sub[["scope", "profile", "arm_label"]].drop_duplicates()
            if label_map.duplicated(["scope", "profile"]).any() or label_map.arm_label.duplicated().any():
                raise ValueError(f"{city} Figure 5 facility labels do not map uniquely to source arms")
            ypos = {name: i for i, name in enumerate(labels)}
            for mode, off, color in (("car", -0.1, PALETTE["car"]), ("wt", 0.1, PALETTE["wt"])):
                for _, row in sub.loc[sub["mode"].astype(str).eq(mode)].iterrows():
                    y, x = ypos[str(row.arm_label)] + off, float(row.coverage_lower)
                    lo = pd.to_numeric(pd.Series([row.get("design_ci_lower")]), errors="coerce").iloc[0]
                    hi = pd.to_numeric(pd.Series([row.get("design_ci_upper")]), errors="coerce").iloc[0]
                    if np.isfinite(lo) and np.isfinite(hi):
                        ax.errorbar(x, y, xerr=[[max(0, x-lo)], [max(0, hi-x)]], fmt="o", ms=2.6, color=color, capsize=1.3, lw=0.65)
                    else:
                        ax.plot(x, y, "o", ms=2.6, color=color)
            display_labels = labels
            if city == "Bogota":
                compact = {
                    "Registry-linked primary (142)": "Registry-linked\nprimary (142)",
                    "Public-provider signal · primary (16)": "Public-provider\nprimary (16)",
                    "Public-provider signal · historical (16)": "Public-provider\nhistorical (16)",
                    "Historical capacity (158)": "Historical\ncapacity (158)",
                }
                display_labels = [compact.get(label, label) for label in labels]
            ax.set_yticks(range(len(labels)), display_labels, fontsize=6.6)
            ax.set_xlim(-2, 102)
            ax.set_xlabel("30-min modeled lower coverage (%)")
            city_display = "São Paulo" if city == "Sao Paulo" else "Bogotá"
            title = "destination-set scenarios" if city == "Sao Paulo" else "142/158-site frames"
            ax.set_title(f"{city_display} · {title}", loc="left", fontsize=6.5, fontweight="bold")
            ax.grid(axis="x", color="#E8E8E8", lw=0.45)
        readiness.append({"component": "facility/definition sensitivity", "status": "READY", "reason": "Scenario labels are preserved; meaning is not inferred from profile names."})
    fig.suptitle(("PILOT ONLY PREVIEW · " if pilot else "") + "Departure-time and destination-set sensitivity", x=0.02, y=0.995, ha="left", fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.31, right=0.985, top=0.94, bottom=0.09, hspace=0.38, wspace=0.42)
    source = same_data.copy()
    if not sensitivity.empty:
        s = sensitivity.copy()
        s["period"] = "sensitivity"
        source = pd.concat([source, s], ignore_index=True, sort=False)
    return save_figure(fig, out / "Figure5_validation_sensitivity"), readiness, source


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "No source rows available.\n"
    d = df[columns] if columns else df
    if max_rows is not None:
        d = d.head(max_rows)
    names = [str(c) for c in d.columns]
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    for row in d.itertuples(index=False, name=None):
        out.append("| " + " | ".join(cell(v) for v in row) + " |")
    if max_rows is not None and len(df) > max_rows:
        out.append("")
        out.append(f"CSV contains all {len(df)} rows.")
    return "\n".join(out) + "\n"


def table1_markdown(table: pd.DataFrame, pilot: bool) -> str:
    table = enrich_source_semantics(table)
    text = "# Table 1. Population frames and facility definitions\n\n"
    if pilot:
        text += "Pilot-only preview. Route records are a diagnostic first-two-by-UID-within-stratum subset, not a probability sample or manuscript estimate.\n\n"
    text += "SP uses the fixed 1,100-origin probability sample; BO uses the complete 1,057-origin frame. Unknown income/estrato origins remain in the denominator. The corrected Bogotá group counts update unknown cells and population share without changing the full frame. SP raw E/A flags are distinct from R-qualified intersections. SP E is a SUS marker, not patient eligibility; BO E is a public-provider signal, not entitlement.\n\n"
    cols = ["record_type", "city", "scope", "scope_name", "measure", "estimate", "unit", "source_rows", "route_origin_rows", "design", "unknown_income_origins", "unknown_income_population", "unknown_income_population_share_pct", "classification_scenario", "definition"]
    return text + markdown_table(table, [c for c in cols if c in table.columns])


def table2_markdown(table: pd.DataFrame, pilot: bool) -> str:
    table = enrich_source_semantics(table)
    text = "# Table 2. Main 15-, 30- and 45-minute estimates\n\n"
    if pilot:
        text += "Pilot-only preview; the diagnostic first-two-by-UID-within-stratum subset is not a probability sample. No design interval is reported for pilot rows.\n\n"
    text += "Coverage is conditional on queried returned routes and the modeled network. coverage_lower/upper are technical-failure bounds; design_ci_lower/upper are pointwise stratified-SRS intervals for the lower estimate and apply only to SP. BO census estimates have no pseudo-sampling CI.\n\n"
    cols = ["city", "scope", "scope_name", "mode", "profile", "profile_definition", "threshold", "n", "sampling_design", "sampled_population", "estimated_population", "coverage_lower", "coverage_upper", "design_se", "design_ci_lower", "design_ci_upper"]
    return text + markdown_table(table, [c for c in cols if c in table.columns])


def make_table3(results_dir: Path, sensitivity_dir: Path | None, readiness: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    status = read_csv(results_dir / "route_status_population.csv")
    for _, r in status.iterrows():
        rows.append({
            "row_type": "route_status", "city": r.get("city", ""), "scope": r.get("scope", ""),
            "mode": r.get("mode", ""), "profile": r.get("profile", ""), "threshold": np.nan,
            "metric": "origin-arm status", "estimate": r.get("design_population", np.nan),
            "lower": np.nan, "upper": np.nan, "n": r.get("n", np.nan),
            "population": r.get("population", np.nan), "status": r.get("status", ""),
            "source": "results/route_status_population.csv",
        })
    auc = read_csv(results_dir / "threshold_area_summaries.csv")
    for _, r in auc.iterrows():
        rows.append({
            "row_type": "threshold_area", "city": r.get("city", ""), "scope": r.get("scope", ""),
            "mode": r.get("mode", ""), "profile": r.get("profile", ""), "threshold": r.get("range_min", ""),
            "metric": "mean modeled coverage over full curve", "estimate": r.get("average_coverage_lower", np.nan),
            "lower": np.nan, "upper": r.get("average_coverage_upper", np.nan), "n": r.get("n", np.nan),
            "population": r.get("estimated_population", np.nan), "status": "technical upper bound is in upper column",
            "source": "results/threshold_area_summaries.csv",
        })
    contrast = read_csv(results_dir / "mode_scope_contrasts.csv")
    for _, r in contrast.iterrows():
        rows.append({
            "row_type": "paired_scope_contrast", "city": r.get("city", ""),
            "scope": str(r.get("scope1", "")) + "−" + str(r.get("scope2", "")),
            "mode": "car/WT interaction", "profile": r.get("profile", ""), "threshold": r.get("threshold", np.nan),
            "metric": "descriptive additive D", "estimate": r.get("contrast", np.nan),
            "lower": r.get("design_ci_lower", np.nan), "upper": r.get("design_ci_upper", np.nan),
            "n": np.nan, "population": np.nan, "status": "design interval; bounds in source file",
            "source": "results/mode_scope_contrasts.csv",
        })
    if sensitivity_dir:
        sens = read_csv(coverage_curves_path(sensitivity_dir))
        if not sens.empty and "threshold" in sens:
            sens = sens.loc[pd.to_numeric(sens.threshold, errors="coerce").isin(DISPLAY_THRESHOLDS)]
        for _, r in sens.iterrows():
            rows.append({
                "row_type": "sensitivity_curve", "city": r.get("city", ""), "scope": r.get("scope", ""),
                "mode": r.get("mode", ""), "profile": r.get("profile", ""), "threshold": r.get("threshold", np.nan),
                "metric": "modeled lower coverage at prespecified node", "estimate": r.get("coverage_lower", np.nan),
                "lower": r.get("design_ci_lower", np.nan), "upper": r.get("design_ci_upper", np.nan),
                "n": r.get("n", np.nan), "population": r.get("estimated_population", np.nan),
                "status": "technical upper bound remains separate in source", "source": str(coverage_curves_path(sensitivity_dir)),
            })
    for r in readiness:
        rows.append({
            "row_type": "readiness", "city": "Both", "scope": "", "mode": "", "profile": "",
            "threshold": np.nan, "metric": r["component"], "estimate": np.nan, "lower": np.nan,
            "upper": np.nan, "n": np.nan, "population": np.nan,
            "status": r["status"] + ": " + r["reason"], "source": "figure readiness check",
        })
    return pd.DataFrame(rows)


def table3_markdown(table: pd.DataFrame, readiness: list[dict[str, str]], pilot: bool) -> str:
    table = enrich_source_semantics(table)
    text = "# Table 3. Route diagnostics and robustness outputs\n\n"
    if pilot:
        text += "Pilot-only diagnostic preview. Formal validation components remain incomplete until their source files are available.\n\n"
    text += "Route statuses remain distinct: no-pattern outcomes are not technical failures. Technical failures contribute bounds, not sampling intervals.\n\n"
    text += "## Figure 5 readiness\n\n| Component | Status | Reason |\n| --- | --- | --- |\n"
    for r in readiness:
        text += f"| {r['component']} | {r['status']} | {r['reason']} |\n"
    text += "\n"
    cols = ["row_type", "city", "scope", "scope_name", "mode", "profile", "profile_definition", "threshold", "metric", "estimate", "lower", "upper", "n", "population", "status", "source"]
    return text + markdown_table(table, [c for c in cols if c in table.columns], 100)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enrich_source_semantics(df).to_csv(path, index=False)


def contact_sheet(figures_dir: Path, path: Path) -> bool:
    stems = [
        "Figure1_population_facilities", "Figure2_mode_scope_threshold", "Figure3_walk_caps",
        "Figure4_income_geography", "Figure5_validation_sensitivity",
    ]
    pngs = [figures_dir / (stem + ".png") for stem in stems if (figures_dir / (stem + ".png")).exists()]
    if not pngs:
        return False
    fig, axes = plt.subplots(len(pngs), 1, figsize=(7.2, 2.0 * len(pngs)), squeeze=False)
    for ax, image_path in zip(axes.flat, pngs):
        ax.imshow(plt.imread(image_path))
        ax.set_title(image_path.stem.replace("_", " "), loc="left", fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def build_pilot_mockroot(root: Path) -> tuple[Path, int]:
    source = root / "core" / "pilot_elapsed_times.csv"
    if not source.exists():
        raise FileNotFoundError("pilot mode requires core/pilot_elapsed_times.csv")
    mock = root / "validation" / "figure_pilot" / "mockroot"
    (mock / "core" / "sets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, mock / "core" / "times_official.csv")
    for name in ("facilities_main.csv",):
        item = root / "core" / name
        if item.exists():
            shutil.copy2(item, mock / "core" / name)
    for name in ("main_sp.csv", "main_bo.csv"):
        item = root / "core" / "sets" / name
        if item.exists():
            shutil.copy2(item, mock / "core" / "sets" / name)
    for name in ("final_scope_overlap.csv", "final_scope_overlap.json"):
        item = root / "facility" / name
        if item.exists():
            (mock / "facility").mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, mock / "facility" / name)
    analyzer = root / "core" / "v13_analyse.py"
    if not analyzer.exists():
        raise FileNotFoundError("pilot mode requires the project's core analysis script")
    subprocess.run(
        [sys.executable, str(analyzer), "--times", str(mock / "core" / "times_official.csv"), "--out", str(mock / "results")],
        check=True,
    )
    times = read_csv(mock / "core" / "times_official.csv")
    metadata = {
        "status": "PILOT_ONLY_NOT_FOR_MANUSCRIPT",
        "source_type": "observed elapsed-route pilot records",
        "source_filename": "pilot_elapsed_times.csv",
        "n_origin": int(times.uid.nunique()) if not times.empty and "uid" in times else 0,
        "n_arm_rows": int(len(times)),
        "profiles": sorted(times.profile.loc[times.profile.notna()].astype(str).unique()) if not times.empty and "profile" in times else [],
        "selection": "diagnostic first two UIDs within each stratum; not a probability sample",
        "analysis": "Aggregated with the project's existing analysis program inside this mock root.",
        "limitations": [
            "Pilot subset is a code-path preview, not a publication estimate.",
            "No synthetic observations were added.",
            "Full frame and facility data are copied only to test figure 1 and table 1.",
        ],
    }
    (mock / "PILOT_ONLY.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return mock, int(metadata["n_origin"])


def compare_f5_to_golden_source(source: pd.DataFrame, golden_path: Path) -> dict[str, Any]:
    golden = read_csv(golden_path)
    required = {"city", "scope", "mode", "profile", "threshold", "coverage_lower"}
    if not required.issubset(golden.columns):
        raise ValueError(f"Golden Figure 5 source lacks columns: {sorted(required - set(golden.columns))}")
    golden = golden.loc[pd.to_numeric(golden.threshold, errors="coerce").eq(30)
                         & golden["mode"].astype(str).isin(["car", "wt"])].copy()
    actual = source.loc[source.period.astype(str).eq("sensitivity")].copy()
    keys = ["city", "scope", "mode", "profile", "threshold"]
    if golden.duplicated(keys).any() or actual.duplicated(keys).any():
        raise ValueError("Golden/current Figure 5 source contains duplicate comparison keys")
    joined = actual.merge(golden, on=keys, how="outer", suffixes=("_current", "_golden"), indicator=True, validate="one_to_one")
    if not joined._merge.eq("both").all():
        raise ValueError("Current common-pool Figure 5 keys differ from the golden final source")
    columns = [c for c in ("coverage_lower", "coverage_upper", "design_ci_lower", "design_ci_upper")
               if f"{c}_current" in joined and f"{c}_golden" in joined]
    for col in columns:
        a = pd.to_numeric(joined[f"{col}_current"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(joined[f"{col}_golden"], errors="coerce").to_numpy(float)
        if not np.allclose(a, b, rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError(f"Figure 5 adapter changed canonical {col} values")
    wt_rows = joined.loc[joined["mode"].astype(str).eq("wt")]
    def value(city: str, scope: str, profile: str) -> float:
        row = wt_rows.loc[wt_rows.city.astype(str).eq(city) & wt_rows.scope.astype(str).eq(scope)
                          & wt_rows.profile.astype(str).eq(profile)]
        if len(row) != 1:
            raise ValueError(f"Golden regression point is absent: {city}/{scope}/{profile}/WT30")
        return float(row.coverage_lower_current.iloc[0])
    return {
        "golden_source": str(golden_path),
        "golden_source_sha256": file_sha256(golden_path),
        "matched_points": int(len(joined)),
        "numeric_columns_compared": columns,
        "numeric_identity": True,
        "reference_wt30_points_from_source": {
            "Bogota R primary142": value("Bogota", "R", "baseline_fixed_departure"),
            "Bogota R historical158": value("Bogota", "R", "sensitivity_bo_historical_capacity"),
            "Bogota E primary142": value("Bogota", "E", "baseline_fixed_departure"),
            "Sao Paulo R primary": value("Sao Paulo", "R", "baseline_fixed_departure"),
            "Sao Paulo R municipal": value("Sao Paulo", "R", "sensitivity_municipal_legacy"),
            "Sao Paulo E public administration": value("Sao Paulo", "E", "sensitivity_public_rd04"),
            "Sao Paulo A annual claims": value("Sao Paulo", "A", "sensitivity_public_rd04"),
            "Sao Paulo A claims >=100": value("Sao Paulo", "A", "sensitivity_rd04_ge100"),
        },
    }


def render_figure5_adapter_only(
    root: Path, out: Path, sameframe_out: Path | None,
    commonpool_analysis: Path, golden_sensitivity: Path | None,
    provisional: bool,
) -> dict[str, Any]:
    """Render only Figure 5 into an isolated adapter-test directory."""
    if not commonpool_analysis.is_file():
        raise FileNotFoundError(f"Explicit common-pool CSV does not exist: {commonpool_analysis}")
    val_out, base_dir = validation_paths(root, sameframe_out)
    if val_out is None or base_dir is None or not (val_out / "validation_sameframe_times.csv").is_file():
        raise FileNotFoundError("Verified same-frame timing inputs were not found")
    base_path = base_dir / "coverage_curves.csv"
    if not base_path.is_file():
        raise FileNotFoundError(f"Same-frame baseline coverage source is missing: {base_path}")
    same_times = read_csv(val_out / "validation_sameframe_times.csv")
    base_curves = read_csv(base_path)
    same_ok, same_reason, same_data = validate_sameframe(same_times, base_curves)
    if not same_ok:
        raise ValueError(f"Same-frame validation failed; refusing an incomplete Figure 5: {same_reason}")
    sensitivity = sensitivity_points(commonpool_analysis)
    validate_figure5_completeness(same_ok, same_data, sensitivity)
    out.mkdir(parents=True, exist_ok=True)
    figure_paths, readiness, source = figure5(
        same_ok, same_reason, same_data, sensitivity, out, pilot=False, require_complete=True
    )
    if any(row["status"] != "READY" for row in readiness):
        raise ValueError(f"Figure 5 readiness did not pass: {readiness}")
    golden_check = compare_f5_to_golden_source(source, golden_sensitivity) if golden_sensitivity else None
    source_path = out / "Figure5_validation_sensitivity_source.csv"
    write_csv(source, source_path)
    manifest = {
        "status": "PROVISIONAL_M04_PENDING" if provisional else "READY_FOR_REVIEW",
        "isolated_figure5_adapter_test": True,
        "figure_files": figure_paths,
        "source_file": source_path.name,
        "source_rows": int(len(source)),
        "facility_points": int(len(sensitivity)),
        "facility_points_by_city": sensitivity.groupby("city").size().to_dict(),
        "facility_profiles": sorted(sensitivity.profile.astype(str).unique().tolist()),
        "commonpool_analysis": str(commonpool_analysis),
        "commonpool_analysis_sha256": file_sha256(commonpool_analysis),
        "sameframe_times": str(val_out / "validation_sameframe_times.csv"),
        "sameframe_times_sha256": file_sha256(val_out / "validation_sameframe_times.csv"),
        "sameframe_base_curves": str(base_path),
        "sameframe_base_curves_sha256": file_sha256(base_path),
        "sameframe_arms": int(len(same_data.loc[same_data["mode"].astype(str).isin(["car", "wt"])])),
        "sameframe_validation": same_reason,
        "readiness": readiness,
        "golden_numeric_comparison": golden_check,
        "producer_sha256": file_sha256(Path(__file__)),
    }
    (out / "FIG5_ADAPTER_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = [
        "# Figure 5 adapter test",
        "",
        f"Status: **{manifest['status']}**",
        "",
        f"Common-pool 30-min points: {len(sensitivity)} ({manifest['facility_points_by_city']}).",
        "Same-frame timing inputs passed UID and denominator checks; SP n=220 and BO n=200 across baseline, weekday and weekend.",
        "Labels are resolved by city, scope and profile. Site counts and semantic definitions are included per point in the source CSV.",
        "",
        f"Golden comparison: {golden_check['matched_points']} points matched with numeric identity." if golden_check else "Golden comparison: not supplied.",
        "M04 routing correction is still pending; this isolated test does not replace the provisional publication assets.",
        "",
    ]
    (out / "FIG5_ADAPTER_AUDIT.md").write_text("\n".join(audit), encoding="utf-8")
    return manifest


def render_bundle(
    root: Path, out: Path, pilot: bool, sameframe_out: Path | None,
    sensitivity_arg: Path | None, bogota_reanalysis_arg: Path | None,
    provisional: bool = False, commonpool_analysis_arg: Path | None = None,
) -> dict[str, Any]:
    figs, source, tables_dir = out / "figures", out / "source_data", out / "tables"
    display_tables_dir = out / "display_tables"
    for directory in (figs, source, tables_dir, display_tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    all_times = read_csv(root / "core" / "times_official.csv")
    times, time_profile = choose_baseline(all_times)
    results_dir = root / "results"
    all_curves = read_csv(results_dir / "coverage_curves.csv")
    all_curves = suppress_design_intervals(all_curves, pilot)
    curves, curve_profile = choose_baseline(all_curves)
    contrasts, _ = choose_baseline(read_csv(results_dir / "mode_scope_contrasts.csv"))
    gradients, _ = choose_baseline(read_csv(results_dir / "income_gradients.csv"))
    contrasts = suppress_design_intervals(contrasts, pilot)
    gradients = suppress_design_intervals(gradients, pilot)
    sensitivity_source = commonpool_analysis_arg or sensitivity_arg or (root / "results_sensitivity")
    sensitivity_source = sensitivity_source if sensitivity_source.exists() else None
    sens_points = sensitivity_points(sensitivity_source)
    table1, facility_data, frames = frame_summaries(root, times, pilot)
    classification_metadata = None
    analysis_dir = None
    if not pilot:
        analysis_dir = bogota_reanalysis_arg or (root / "hp_revision_20261116" / "bogota")
        gradients, table1, classification_metadata = apply_bogota_classification_reanalysis(
            gradients, table1, analysis_dir
        )
    map_data = map_classifications(times, time_profile, 30)
    bogota_classification_coverage = pd.DataFrame()
    bogota_origin_classification = pd.DataFrame()
    if not pilot and analysis_dir is not None:
        map_data, bogota_origin_classification = attach_bogota_classification_groups(map_data, analysis_dir)
        bogota_classification_coverage = prepare_bogota_classification_coverage(analysis_dir)
    paired = paired_difference_table(times, time_profile, use_design_ci=not pilot)
    if not curves.empty:
        curves = curves.loc[pd.to_numeric(curves.threshold, errors="coerce").isin(THRESHOLDS)].copy()
    table2 = curves.loc[pd.to_numeric(curves.threshold, errors="coerce").isin(DISPLAY_THRESHOLDS)].copy() if not curves.empty else pd.DataFrame()
    write_csv(table1, source / "Table1_population_facility_source.csv")
    write_csv(table2, source / "Table2_main_15_30_45_source.csv")
    write_csv(curves, source / "Figure2_main_curve_source.csv")
    write_csv(all_curves, source / "results_coverage_curves_all_profiles_source.csv")
    write_csv(contrasts, source / "Figure2_paired_scope_contrast_source.csv")
    write_csv(paired, source / "Figure3_paired_mode_source.csv")
    write_csv(map_data, source / "Figure4_origin_classification_source.csv")
    write_csv(gradients, source / "Figure4_income_gradient_source.csv")
    if not bogota_origin_classification.empty:
        write_csv(bogota_origin_classification, source / "Bogota_primary_origin_classification_source.csv")
    if not bogota_classification_coverage.empty:
        write_csv(bogota_classification_coverage, source / "Bogota_classification_coverage_curves_source.csv")
    write_csv(sens_points, source / "Figure5_sensitivity_source.csv")
    write_csv(facility_data, source / "Figure1_facility_scenario_counts.csv")
    figure1(root, figs, facility_data, frames, times, pilot)
    figure2(curves, contrasts, figs, pilot)
    figure3(curves, paired, figs, pilot)
    figure4(gradients, map_data, figs, pilot)

    val_out, base_dir = validation_paths(root, sameframe_out)
    same_times = read_csv(val_out / "validation_sameframe_times.csv") if val_out else pd.DataFrame()
    base_curves = read_csv(base_dir / "coverage_curves.csv") if base_dir else pd.DataFrame()
    same_ok, same_reason, same_data = False, "same-frame extension output is not present", pd.DataFrame()
    notes = base_dir / "ANALYSIS_NOTES.json" if base_dir else None
    if notes and notes.exists():
        try:
            if json.loads(notes.read_text(encoding="utf-8")).get("status") == "validation_periods_absent":
                same_reason = "same-frame weekday/weekend route profiles are absent"
        except Exception:
            pass
    if not same_times.empty:
        same_ok, same_reason, same_data = validate_sameframe(same_times, base_curves)
    fig5_paths, readiness, fig5_source = figure5(
        same_ok, same_reason, same_data, sens_points, figs, pilot,
        require_complete=commonpool_analysis_arg is not None,
    )
    write_csv(fig5_source, source / "Figure5_validation_sensitivity_source.csv")

    table3 = make_table3(results_dir, sensitivity_source, readiness)
    write_csv(table3, source / "Table3_diagnostics_robustness_source.csv")
    write_csv(table1, tables_dir / "Table1_population_facilities.csv")
    write_csv(table2, tables_dir / "Table2_main_estimates.csv")
    write_csv(table3, tables_dir / "Table3_diagnostics_robustness.csv")
    (tables_dir / "Table1_population_facilities.md").write_text(table1_markdown(table1, pilot), encoding="utf-8")
    (tables_dir / "Table2_main_estimates.md").write_text(table2_markdown(table2, pilot), encoding="utf-8")
    (tables_dir / "Table3_diagnostics_robustness.md").write_text(table3_markdown(table3, readiness, pilot), encoding="utf-8")
    for name, table in (
        ("table1", table1),
        ("table2", table2),
        ("table3", table3),
    ):
        write_csv(table, display_tables_dir / f"{name}.csv")
    for name, source_name in (
        ("table1", "Table1_population_facilities.md"),
        ("table2", "Table2_main_estimates.md"),
        ("table3", "Table3_diagnostics_robustness.md"),
    ):
        (display_tables_dir / f"{name}.md").write_text((tables_dir / source_name).read_text(encoding="utf-8"), encoding="utf-8")
    sheet = contact_sheet(figs, out / "contact_sheet.png")

    incomplete = [r for r in readiness if r["status"] != "READY"]
    if curves.empty:
        incomplete.append({"component": "main coverage figures/tables", "status": "INCOMPLETE", "reason": "results/coverage_curves.csv is missing"})
    if times.empty:
        incomplete.append({"component": "origin map and paired mode panels", "status": "INCOMPLETE", "reason": "core/times_official.csv is missing"})
    qa = [
        "# Figure and table QA", "",
        f"Status: {'PILOT ONLY' if pilot else ('PROVISIONAL — pending metro routing review' if provisional else ('INCOMPLETE' if incomplete else 'READY'))}.", "",
        "## Figure contract", "",
        "- Core conclusion: transport mode, destination scenario and time threshold jointly alter modeled potential geographic coverage.",
        "- Archetype: quantitative grid for figures 1–3 and 5; asymmetric mixed-modality for figure 4.",
        "- Backend: Python/matplotlib produced all figure outputs, previews, exports and the contact sheet.",
        "- Final width: 183 mm; white background; editable SVG/PDF text; 600-dpi TIFF.",
        "- Coverage is conditional on queried returned routes and the modeled network, not observed clinical access.", "",
        *( ["- Provisional baseline export: the route service-mode mapping is under correction; do not treat the current network estimates as final until rerun with the metro-inclusive route output.", ""] if provisional else [] ),
        "## Statistical and interpretation notes", "",
        "- SP main-result design intervals apply to its probability sample. The full BO main frame is a census and has no pseudo-sampling CI.",
        "- The pilot preview uses a diagnostic first-two-by-UID-within-stratum set with inherited weights; all pilot design intervals are suppressed and no pilot population inference is intended.",
        "- In Figure 5, the validation-frame probability sample can have design intervals in either city; this is distinct from the full-frame BO census.",
        "- Technical-failure bounds are distinct from sampling intervals.",
        "- D is a descriptive paired additive mode-by-scope contrast, not a causal difference-in-differences.",
        "- SP Q1–Q5 are tract-income quintiles among positive V06004 tracts, not equal-population person quintiles. Unknown-income origins stay in the overall denominator.",
        "- BO estrato is a residential socioeconomic category, not monetary income; corrected unknown estrato counts/population share use the `mean_30_epsg4326` classification and stay in the full frame.",
        "- SP E is the 2024 any-month laparoscopic/video SUS marker, not individual eligibility; raw E/A counts are separate from R-qualified intersections.",
        "- BO E is a public-provider signal and is not interpreted as service entitlement or as equivalent to SP E.",
        "- Maps mark observed origins only; no unsampled area is filled.",
        "- Bogotá low/high gradients in Figure 4 use the corrected `mean_30_epsg4326` grouping over all 10–60 min thresholds; no sampling CI is reported for the complete-frame census.",
        "- Figure 4 origin-map source rows include the corrected Bogotá socioeconomic group for each mapped origin; the current full Bogotá subgroup curves are provided separately from overall routing curves.",
        "- Every exported source CSV keeps the machine scope/profile key and adds reader-facing `scope_name`, `scope_definition` and `profile_definition` columns.",
        "- Table 1 preserves the historical classification only as a comparison; two cells (40,902.0584 population) changed group after boundary snapping, while overall coverage and D remain fixed.",
        "- WT caps filter a common returned candidate pool and do not establish a globally constrained optimum.", "",
        "## Row selection and exclusions", "",
        f"- Primary times profile: {time_profile or 'none selected'}; retained {len(times):,} of {len(all_times):,} time rows for baseline map and paired panels.",
        f"- Primary curve profile: {curve_profile or 'none selected'}; retained {len(curves):,} baseline rows in the prespecified 10–60 min range from {len(all_curves):,} input rows.",
        "- Non-baseline curve rows are retained verbatim in source_data/results_coverage_curves_all_profiles_source.csv; only the named primary profile is drawn in main-result panels.",
        "- Paired ratio estimators record input, retained and excluded row counts in the source CSV. Only non-finite outcomes and non-positive/non-finite design weights are omitted from an estimate; route status rows remain available.",
        "- Missing N_h values are ignored only while retrieving a stratum finite-population size for variance calculation; no origin rows are removed by that lookup.", "",
        "## Readiness", "",
    ]
    qa += [f"- {r['status']}: {r['component']} — {r['reason']}" for r in readiness]
    qa += [f"- {r['status']}: {r['component']} — {r['reason']}" for r in incomplete]
    qa += ["", "## Bundle files", ""]
    qa += [f"- figures/{p.name}" for p in sorted(figs.iterdir()) if p.is_file()]
    qa += [f"- source_data/{p.name}" for p in sorted(source.glob("*.csv"))]
    qa += [f"- display_tables/{p.name}" for p in sorted(display_tables_dir.iterdir()) if p.is_file()]
    qa += ["- contact_sheet.png (Python-generated).", "- Tables are provided as CSV and Markdown.", ""]
    (out / "FIGURE_QA.md").write_text("\n".join(qa), encoding="utf-8")
    manifest = {
        "status": "PILOT_ONLY" if pilot else ("PROVISIONAL_M04_PENDING" if provisional else ("INCOMPLETE" if incomplete else "READY")),
        "provisional": bool(provisional),
        "backend": "python/matplotlib", "pilot_only": bool(pilot),
        "input_roles": {
            "times": "core/times_official.csv", "main_results": "results/*.csv",
            "sensitivity_results": str(sensitivity_source) if sensitivity_source else None,
            "sameframe": "extensions/**/validation_sameframe_times.csv and validation_sameframe_base/*.csv",
        },
        "time_profile": time_profile, "curve_profile": curve_profile,
        "bogota_classification_reanalysis": classification_metadata,
        "bogota_classification_subgroup_source": "source_data/Bogota_classification_coverage_curves_source.csv",
        "bogota_origin_group_source": "source_data/Bogota_primary_origin_classification_source.csv",
        "figures": ["Figure1_population_facilities", "Figure2_mode_scope_threshold", "Figure3_walk_caps", "Figure4_income_geography", "Figure5_validation_sensitivity"],
        "tables": ["Table1_population_facilities", "Table2_main_estimates", "Table3_diagnostics_robustness"],
        "display_tables": ["display_tables/table1.csv", "display_tables/table2.csv", "display_tables/table3.csv"],
        "incomplete_components": incomplete, "figure5_readiness": readiness,
        "contact_sheet_written": sheet, "exports": ["PNG preview", "editable SVG", "editable PDF", "600-dpi TIFF"],
    }
    (out / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the v13 five-figure / three-table bundle.")
    parser.add_argument("--root", required=True, type=Path, help="Project root B")
    parser.add_argument("--out", required=True, type=Path, help="Publication bundle output directory")
    parser.add_argument("--pilot", action="store_true", help="Aggregate the real pilot in a disposable mock root and label outputs pilot-only")
    parser.add_argument("--sameframe-out", type=Path, default=None, help="Extension OUTDIR with validation_sameframe_times.csv and validation_sameframe_base/")
    parser.add_argument("--sensitivity-dir", type=Path, default=None, help="Override results_sensitivity directory")
    parser.add_argument("--commonpool-analysis", type=Path, default=None, help="Explicit complete common-pool coverage_curves.csv used for Figure 5")
    parser.add_argument("--figure5-only", action="store_true", help="Render only Figure 5 into the requested output directory")
    parser.add_argument("--golden-sensitivity", type=Path, default=None, help="Optional prior Figure5_sensitivity_source.csv for exact numeric regression comparison")
    parser.add_argument("--bogota-reanalysis", type=Path, default=None, help="Corrected Bogotá classification output directory")
    parser.add_argument("--provisional", action="store_true", help="Mark this baseline export provisional pending an external routing correction")
    args = parser.parse_args()
    project_root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    commonpool_analysis = args.commonpool_analysis.expanduser().resolve() if args.commonpool_analysis else None
    sameframe_out = args.sameframe_out.expanduser().resolve() if args.sameframe_out else None
    golden_sensitivity = args.golden_sensitivity.expanduser().resolve() if args.golden_sensitivity else None
    if args.figure5_only:
        if args.pilot:
            parser.error("--pilot cannot be combined with --figure5-only")
        if commonpool_analysis is None:
            parser.error("--figure5-only requires --commonpool-analysis <coverage_curves.csv>")
        manifest = render_figure5_adapter_only(
            project_root, out, sameframe_out, commonpool_analysis, golden_sensitivity, args.provisional
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        return 0
    input_root = project_root
    if args.pilot:
        input_root, origins = build_pilot_mockroot(project_root)
        print(f"PILOT_ONLY mockroot ready; origins={origins}", flush=True)
    bogota_reanalysis = args.bogota_reanalysis.expanduser().resolve() if args.bogota_reanalysis else None
    manifest = render_bundle(
        input_root, out, args.pilot, sameframe_out, args.sensitivity_dir,
        bogota_reanalysis, args.provisional, commonpool_analysis,
    )
    print(f"BUNDLE_STATUS={manifest['status']}", flush=True)
    print(f"OUT={out}", flush=True)
    for item in manifest["incomplete_components"]:
        print(f"INCOMPLETE={item['component']}: {item['reason']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
