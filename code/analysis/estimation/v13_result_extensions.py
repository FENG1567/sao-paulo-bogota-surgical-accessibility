#!/usr/bin/env python3
"""Design-aware extensions for the fixed-origin V13 route analysis.

This script reads a materialized times CSV only. It does not route, download,
or modify its inputs. Sampling variance and coverage bounds are delegated to
the existing v13_analyse.ratio_estimate and coverage_vector functions.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v13_analyse import coverage_vector, ratio_estimate


THRESHOLDS = list(range(10, 61, 5))
REQUIRED_COLUMNS = [
    "city", "uid", "scope", "mode", "profile", "stratum", "pop",
    "design_weight", "N_h", "n_h", "inclusion_prob", "time_min",
    "status", "finite_unknown", "lat", "lon",
]
REQUIRED_SCOPES = {"R", "E"}
FIXTURE_SCOPES = {"R", "E", "A"}
CORE_MODES = {"car", "wt", "foot"}
META_COLUMNS = [
    "stratum", "pop", "design_weight", "N_h", "n_h", "inclusion_prob",
    "lat", "lon",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    values = series.fillna(False)
    truthy = {True, 1, "1", "true", "True", "TRUE", "yes", "Yes"}
    falsy = {False, 0, "0", "false", "False", "FALSE", "no", "No", ""}
    normalized = []
    for value in values:
        if value in truthy:
            normalized.append(True)
        elif value in falsy:
            normalized.append(False)
        else:
            raise ValueError(f"invalid finite_unknown value: {value!r}")
    return pd.Series(normalized, index=series.index, dtype=bool)


def validate_times(df: pd.DataFrame, require_complete_arms: bool = True) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"times file is missing required columns: {missing}")
    out = df.copy()
    for col in ("city", "uid", "scope", "mode", "profile", "status"):
        out[col] = out[col].astype("string")
        if out[col].isna().any() or out[col].str.strip().eq("").any():
            raise ValueError(f"null/empty values are not allowed in {col}")
        out[col] = out[col].astype(str)
    out["stratum"] = out["stratum"].fillna("unknown").astype(str)
    out.loc[out.stratum.str.strip().eq(""), "stratum"] = "unknown"
    keys = ["city", "uid", "scope", "mode", "profile"]
    if out.duplicated(keys, keep=False).any():
        examples = out.loc[out.duplicated(keys, keep=False), keys].head(8).to_dict("records")
        raise ValueError(f"duplicate city/uid/scope/mode/profile arm rows: {examples}")
    for col in ("pop", "design_weight", "N_h", "n_h", "inclusion_prob"):
        out[col] = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(out[col].to_numpy(float)).all():
            raise ValueError(f"non-finite values in {col}")
    if (out["pop"] <= 0).any() or (out["design_weight"] <= 0).any():
        raise ValueError("pop and design_weight must be positive")
    if (out["N_h"] <= 0).any() or (out["n_h"] <= 0).any() or (out["n_h"] > out["N_h"]).any():
        raise ValueError("invalid stratum sample sizes N_h/n_h")
    if (out["inclusion_prob"] <= 0).any() or (out["inclusion_prob"] > 1).any():
        raise ValueError("inclusion_prob must be in (0, 1]")
    raw_time = out["time_min"]
    parsed_time = pd.to_numeric(raw_time, errors="coerce")
    invalid_text = raw_time.notna() & parsed_time.isna()
    if invalid_text.any():
        raise ValueError("time_min contains nonnumeric non-missing values")
    finite_time = np.isfinite(parsed_time.to_numpy(float))
    if (parsed_time[finite_time] < 0).any():
        raise ValueError("time_min must be nonnegative when returned")
    out["time_min"] = parsed_time.where(finite_time, np.nan)
    out["finite_unknown"] = _as_bool(out["finite_unknown"])

    # Every UID must carry identical design/geographic metadata in every arm.
    for (city, profile, uid), group in out.groupby(["city", "profile", "uid"], sort=False, dropna=False):
        for col in META_COLUMNS:
            if group[col].nunique(dropna=False) > 1:
                raise ValueError(f"arm metadata mismatch for {city}/{profile}/{uid}: {col}")

    if require_complete_arms:
        for (city, profile), group in out.groupby(["city", "profile"], sort=False):
            scopes = set(group.scope)
            modes = set(group["mode"])
            if not REQUIRED_SCOPES.issubset(scopes):
                raise ValueError(f"{city}/{profile} lacks required scopes {sorted(REQUIRED_SCOPES-scopes)}")
            if not CORE_MODES.issubset(modes):
                raise ValueError(f"{city}/{profile} lacks required modes {sorted(CORE_MODES-modes)}")
            expected = set(group.uid)
            # R and E are the required scopes. A is optional (for example,
            # Bogota's facility frame may not define an A scope).
            for scope in sorted(scopes):
                for mode in sorted(CORE_MODES):
                    arm = group[(group.scope == scope) & (group["mode"] == mode)]
                    if arm.empty:
                        raise ValueError(f"missing required arm {city}/{profile}/{scope}/{mode}")
                    actual = set(arm.uid)
                    if actual != expected:
                        absent = sorted(expected - actual)[:8]
                        extra = sorted(actual - expected)[:8]
                        raise ValueError(f"incomplete arm {city}/{profile}/{scope}/{mode}; missing={absent}; extra={extra}")
            # Any additional represented mode/scope arm is also required to be
            # complete; this prevents caps or sensitivity arms from silently
            # changing the origin denominator.
            for (scope, mode), arm in group.groupby(["scope", "mode"], sort=False):
                if set(arm.uid) != expected:
                    absent = sorted(expected - set(arm.uid))[:8]
                    raise ValueError(f"incomplete arm {city}/{profile}/{scope}/{mode}; missing={absent}")
    return out


def _arms(df: pd.DataFrame) -> dict[tuple[str, str, str, str], pd.DataFrame]:
    return {
        (str(city), str(profile), str(scope), str(mode)): group.sort_values("uid").reset_index(drop=True)
        for (city, profile, scope, mode), group in df.groupby(["city", "profile", "scope", "mode"], sort=True)
    }


def _design_ci(est: dict[str, float], clamp: tuple[float, float] | None = None) -> tuple[float, float]:
    lo, hi = est["lower"], est["upper"]
    if clamp is not None:
        lo = max(clamp[0], lo) if np.isfinite(lo) else np.nan
        hi = min(clamp[1], hi) if np.isfinite(hi) else np.nan
    return lo, hi


def _event_bounds(x_lo: np.ndarray, x_hi: np.ndarray,
                  y_lo: np.ndarray, y_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feasible bounds for the paired binary event x=1 and y=0."""
    lower = ((x_lo >= 1) & (y_hi <= 0)).astype(float)
    upper = ((x_hi >= 1) & (y_lo <= 0)).astype(float)
    return lower, upper


def _metric_record(base: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    low = ratio_estimate(base, lower)
    high = ratio_estimate(base, upper)
    ci_low, ci_high = _design_ci(low, (0.0, 100.0))
    weight = base.design_weight.to_numpy(float)
    return {
        "estimate_pct": low["estimate"],
        "technical_lower_pct": low["estimate"],
        "technical_upper_pct": high["estimate"],
        "design_se_pp": low["se"],
        "design_ci_lower_pct": ci_low,
        "design_ci_upper_pct": ci_high,
        "ht_population_lower": float(np.dot(weight, lower)),
        "ht_population_upper": float(np.dot(weight, upper)),
    }


def _indicators(d: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    return coverage_vector(d, threshold)


def build_discordances(df: pd.DataFrame) -> pd.DataFrame:
    arms = _arms(df)
    rows: list[dict[str, Any]] = []
    for (city, profile, scope, mode), d in sorted(arms.items()):
        if mode not in {"car", "wt", "foot"}:
            continue
        if mode != "car":
            continue
        wt = arms[(city, profile, scope, "wt")].set_index("uid").loc[d.uid].reset_index()
        foot = arms[(city, profile, scope, "foot")].set_index("uid").loc[d.uid].reset_index()
        base = d.reset_index(drop=True)
        for threshold in THRESHOLDS:
            car_lo, car_hi = _indicators(base, threshold)
            wt_lo, wt_hi = _indicators(wt, threshold)
            foot_lo, foot_hi = _indicators(foot, threshold)
            definitions = [
                ("car_covers_wt_does_not", car_lo, car_hi, wt_lo, wt_hi),
                ("wt_covers_car_does_not", wt_lo, wt_hi, car_lo, car_hi),
                ("wt_covers_foot_does_not", wt_lo, wt_hi, foot_lo, foot_hi),
            ]
            for event, xlo, xhi, ylo, yhi in definitions:
                lower, upper = _event_bounds(xlo, xhi, ylo, yhi)
                rows.append({
                    "city": city, "profile": profile, "scope": scope,
                    "threshold_min": threshold, "event": event,
                    "n_origins": len(base), **_metric_record(base, lower, upper),
                })
    return pd.DataFrame(rows)


def _income_label(city: str, stratum: str) -> tuple[str, str, str]:
    label = stratum if stratum else "unknown"
    if label.strip().lower() in {"unknown", "missing", "na", "nan", "none"}:
        return "missing_or_unknown", "missing_or_unknown", "unknown"
    if city == "Sao Paulo":
        return label, "income_quintile", label
    if city.lower() == "bogota":
        return f"estrato:{label}", "estrato_not_income", "not_available"
    return label, "observed_sampling_stratum", label


def build_coverage_curves_by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (city, profile, scope, mode), d0 in df.groupby(["city", "profile", "scope", "mode"], sort=True):
        d = d0.reset_index(drop=True)
        groups: list[tuple[str, str, str, pd.DataFrame]] = [
            ("overall", "overall", "all origins including missing/unknown", d)
        ]
        for stratum, subgroup in d.groupby("stratum", sort=True, dropna=False):
            display, kind, income = _income_label(str(city), str(stratum))
            groups.append((display, kind, income, subgroup.reset_index(drop=True)))
        for label, kind, income, subset in groups:
            for threshold in THRESHOLDS:
                lo, hi = _indicators(subset, threshold)
                est = ratio_estimate(subset, lo)
                upper = ratio_estimate(subset, hi)
                ci_low, ci_high = _design_ci(est, (0.0, 100.0))
                rows.append({
                    "city": city, "profile": profile, "scope": scope, "mode": mode,
                    "subgroup": label, "subgroup_type": kind, "income_group": income,
                    "sampling_stratum": "all" if label == "overall" else str(subset.stratum.iloc[0]),
                    "threshold_min": threshold, "n_origins": len(subset),
                    "coverage_lower_pct": est["estimate"],
                    "coverage_technical_upper_pct": upper["estimate"],
                    "design_se_pp": est["se"], "design_ci_lower_pct": ci_low,
                    "design_ci_upper_pct": ci_high,
                })
    return pd.DataFrame(rows)


def build_scope_losses(df: pd.DataFrame) -> pd.DataFrame:
    arms = _arms(df)
    rows: list[dict[str, Any]] = []
    cities_profiles = sorted(set((c, p) for c, p, _, _ in arms))
    for city, profile in cities_profiles:
        modes = sorted({mode for c, p, _, mode in arms if c == city and p == profile})
        for mode in modes:
            ref = arms[(city, profile, "R", mode)].reset_index(drop=True)
            available_scopes = {scope for c, p, scope, m in arms if c == city and p == profile and m == mode}
            for target in ("E", "A"):
                if target not in available_scopes:
                    continue
                dest = arms[(city, profile, target, mode)].set_index("uid").loc[ref.uid].reset_index()
                for threshold in THRESHOLDS:
                    rlo, rhi = _indicators(ref, threshold)
                    tlo, thi = _indicators(dest, threshold)
                    loss_lo, loss_hi = _event_bounds(rlo, rhi, tlo, thi)
                    groups: list[tuple[str, str, str, pd.DataFrame, np.ndarray, np.ndarray]] = [
                        ("overall", "overall", "all origins including missing/unknown", ref, loss_lo, loss_hi)
                    ]
                    for stratum, idx in ref.groupby("stratum", sort=True).groups.items():
                        ix = np.array(list(idx), dtype=int)
                        sub = ref.iloc[ix].reset_index(drop=True)
                        label, kind, income = _income_label(city, str(stratum))
                        groups.append((label, kind, income, sub, loss_lo[ix], loss_hi[ix]))
                    for label, kind, income, subset, lo, hi in groups:
                        row = _metric_record(subset, lo, hi)
                        rows.append({
                            "city": city, "profile": profile, "mode": mode,
                            "scope_from": "R", "scope_to": target,
                            "threshold_min": threshold, "subgroup": label,
                            "subgroup_type": kind, "income_group": income,
                            "sampling_stratum": "all" if label == "overall" else str(subset.stratum.iloc[0]),
                            "n_origins": len(subset),
                            "coverage_loss_pct": row["estimate_pct"],
                            "technical_lower_pct": row["technical_lower_pct"],
                            "technical_upper_pct": row["technical_upper_pct"],
                            "design_se_pp": row["design_se_pp"],
                            "design_ci_lower_pct": row["design_ci_lower_pct"],
                            "design_ci_upper_pct": row["design_ci_upper_pct"],
                            "ht_population_lost_lower": row["ht_population_lower"],
                            "ht_population_lost_upper": row["ht_population_upper"],
                        })
    return pd.DataFrame(rows)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cut = q * float(weights.sum())
    return float(values[min(int(np.searchsorted(np.cumsum(weights), cut, side="left")), len(values) - 1)])


def build_returned_route_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (city, profile, scope, mode), d0 in df.groupby(["city", "profile", "scope", "mode"], sort=True):
        d = d0.reset_index(drop=True)
        t = d.time_min.to_numpy(float)
        w = d.design_weight.to_numpy(float)
        has = np.isfinite(t)
        route_population = float(w[has].sum())
        no_route_population = float(w[~has].sum())
        total = float(w.sum())
        rows.append({
            "city": city, "profile": profile, "scope": scope, "mode": mode,
            "quantiles_condition": "finite returned candidate routes only",
            "p50_min": _weighted_quantile(t[has], w[has], .50) if has.any() else np.nan,
            "p90_min": _weighted_quantile(t[has], w[has], .90) if has.any() else np.nan,
            "n_returned_origins": int(has.sum()), "n_no_returned_origins": int((~has).sum()),
            "ht_population_with_returned_route": route_population,
            "ht_population_no_returned_route": no_route_population,
            "ht_population_total": total,
            "pct_population_no_returned_route": 100 * no_route_population / total if total else np.nan,
        })
    return pd.DataFrame(rows)


def build_car_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "main_factor_1_over_0_6": lambda t: t,
        "free_flow_factor_1_0": lambda t: t * 0.6,
        "intermediate_factor_1_over_0_8": lambda t: t * 0.6 / 0.8,
        "reconstructed_base_plus_5_min": lambda t: t * 0.6 + 5.0,
        "main_adjusted_plus_5_min": lambda t: t + 5.0,
    }
    arms = _arms(df)
    rows: list[dict[str, Any]] = []
    for (city, profile, scope, mode), car0 in sorted(arms.items()):
        if mode != "car":
            continue
        car = car0.reset_index(drop=True)
        wt = arms[(city, profile, scope, "wt")].set_index("uid").loc[car.uid].reset_index()
        base = car.copy()
        for scenario, transform in scenarios.items():
            scenario_car = car.copy()
            scenario_car["time_min"] = transform(car.time_min.to_numpy(float))
            for threshold in THRESHOLDS:
                car_lo, car_hi = _indicators(scenario_car, threshold)
                wt_lo, wt_hi = _indicators(wt, threshold)
                car_est = ratio_estimate(base, car_lo)
                wt_est = ratio_estimate(base, wt_lo)
                delta_lo = car_lo - wt_lo
                delta = ratio_estimate(base, delta_lo)
                delta_tech_lo = ratio_estimate(base, car_lo - wt_hi)["estimate"]
                delta_tech_hi = ratio_estimate(base, car_hi - wt_lo)["estimate"]
                ci_low, ci_high = _design_ci(delta)
                rows.append({
                    "city": city, "profile": profile, "scope": scope,
                    "scenario": scenario, "threshold_min": threshold,
                    "car_coverage_pct": car_est["estimate"],
                    "car_technical_upper_pct": ratio_estimate(base, car_hi)["estimate"],
                    "wt_coverage_pct": wt_est["estimate"],
                    "wt_technical_upper_pct": ratio_estimate(base, wt_hi)["estimate"],
                    "car_minus_wt_pp": delta["estimate"],
                    "technical_delta_lower_pp": delta_tech_lo,
                    "technical_delta_upper_pp": delta_tech_hi,
                    "design_se_pp": delta["se"],
                    "design_ci_lower_pp": ci_low,
                    "design_ci_upper_pp": ci_high,
                    "n_origins": len(base),
                })
    return pd.DataFrame(rows)


def build_spatial_paired(df: pd.DataFrame) -> pd.DataFrame:
    """One de-identified origin row with common design fields and wide paired arms."""
    output: list[pd.DataFrame] = []
    for (city, profile), group in df.groupby(["city", "profile"], sort=True):
        first = group.sort_values(["scope", "mode"]).drop_duplicates("uid").copy()
        result = first[["city", "profile", "uid", "lat", "lon", *META_COLUMNS]].copy()
        result = result.rename(columns={
            "stratum": "sampling_stratum_common", "pop": "pop_common",
            "design_weight": "design_weight_common", "N_h": "N_h_common",
            "n_h": "n_h_common", "inclusion_prob": "inclusion_prob_common",
        })
        result = result.loc[:, ~result.columns.duplicated()]
        for (scope, mode), arm in group.groupby(["scope", "mode"], sort=True):
            arm = arm.set_index("uid").loc[result.uid]
            prefix = f"{scope}_{mode}"
            result[f"{prefix}_time_min"] = pd.to_numeric(arm.time_min, errors="coerce").to_numpy()
            result[f"{prefix}_status"] = arm.status.astype(str).to_numpy()
            result[f"{prefix}_finite_unknown"] = arm.finite_unknown.astype(bool).to_numpy()
            result[f"{prefix}_covered30"] = (np.isfinite(arm.time_min.to_numpy(float)) & (arm.time_min.to_numpy(float) <= 30)).astype(int)
        output.append(result)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def build_validation_sameframe(df: pd.DataFrame, sets_dir: Path, out: Path,
                               analyzer: Path) -> dict[str, Any]:
    files = [sets_dir / "validation_sp.csv", sets_dir / "validation_bo.csv"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"--sets requires {path}")
    profile_lower = df.profile.str.lower()
    baseline_profiles = sorted(df.loc[profile_lower.str.startswith("baseline"), "profile"].unique().tolist())
    validation_profiles = {
        "weekday": sorted(df.loc[profile_lower.str.startswith("validation_weekday"), "profile"].unique().tolist()),
        "weekend": sorted(df.loc[profile_lower.str.startswith("validation_weekend"), "profile"].unique().tolist()),
    }
    if not baseline_profiles:
        raise ValueError("--sets was provided but no baseline* profiles occur in --times")
    frames = []
    replace_cols = ["city", "uid", "stratum", "pop", "design_weight", "N_h", "n_h", "inclusion_prob", "lat", "lon"]
    for path in files:
        frame = pd.read_csv(path, dtype={"uid": str})
        need = set(replace_cols)
        absent = sorted(need - set(frame.columns))
        if absent:
            raise ValueError(f"validation frame {path} misses columns {absent}")
        if frame.duplicated(["city", "uid"]).any():
            raise ValueError(f"duplicate city/uid in validation frame {path}")
        available_rows = df.loc[df.profile.isin(baseline_profiles) & df.city.isin(frame.city), ["city", "uid"]]
        available = set(map(tuple, available_rows.astype(str).drop_duplicates().to_numpy().tolist()))
        requested = set(map(tuple, frame[["city", "uid"]].astype(str).drop_duplicates().to_numpy().tolist()))
        unmatched = sorted(requested - available)
        if unmatched:
            raise ValueError(f"validation city/UIDs absent from baseline times in {path.name}: {unmatched[:8]}")
        frames.append(frame[replace_cols])
    validation = pd.concat(frames, ignore_index=True)
    if validation.duplicated(["city", "uid"]).any():
        raise ValueError("validation_sp and validation_bo contain duplicate city/uid rows")
    key = ["city", "uid"]
    expected_uids = {str(city): set(group.uid.astype(str)) for city, group in validation.groupby("city", sort=False)}

    for profile in baseline_profiles:
        for city, expected in expected_uids.items():
            observed = set(df.loc[(df.profile == profile) & (df.city == city), "uid"].astype(str))
            missing_uids = sorted(expected - observed)
            if missing_uids:
                raise ValueError(
                    f"{profile}/{city} is missing validation-frame UIDs: {missing_uids[:8]}"
                )

    # Each available date profile must contain the complete validation-frame UID
    # set in every city. validate_times separately verifies all scope/mode arms.
    for category, profiles in validation_profiles.items():
        for profile in profiles:
            for city, expected in expected_uids.items():
                observed = set(df.loc[(df.profile == profile) & (df.city == city), "uid"].astype(str))
                missing_uids = sorted(expected - observed)
                if missing_uids:
                    raise ValueError(
                        f"{profile}/{city} is missing validation-frame UIDs: {missing_uids[:8]}"
                    )

    baseline = df[df.profile.isin(baseline_profiles)].copy()
    baseline = baseline.merge(validation, on=key, how="inner", suffixes=("", "_validation"), validate="many_to_one")
    for col in replace_cols:
        if col not in key:
            baseline[col] = baseline[f"{col}_validation"]
            baseline.drop(columns=[f"{col}_validation"], inplace=True)

    validation_profile_names = validation_profiles["weekday"] + validation_profiles["weekend"]
    dates = df[df.profile.isin(validation_profile_names)].copy()
    if not dates.empty:
        dates = dates.merge(validation, on=key, how="inner", suffixes=("", "_validation"), validate="many_to_one")
        # Date profiles were already built on the validation frame. Check every
        # metadata field; do not silently overwrite a weight or stratum mismatch.
        for col in replace_cols:
            if col in key:
                continue
            actual = dates[col]
            expected = dates[f"{col}_validation"]
            if col == "stratum":
                equal = actual.fillna("unknown").astype(str).to_numpy() == expected.fillna("unknown").astype(str).to_numpy()
            else:
                a = pd.to_numeric(actual, errors="coerce").to_numpy(float)
                b = pd.to_numeric(expected, errors="coerce").to_numpy(float)
                equal = np.isclose(a, b, rtol=1e-9, atol=1e-8, equal_nan=True)
            if not bool(np.all(equal)):
                bad = int(np.flatnonzero(~equal)[0])
                raise ValueError(
                    f"{dates.iloc[bad].profile}/{dates.iloc[bad].city}/{dates.iloc[bad].uid} "
                    f"validation-frame metadata mismatch: {col}"
                )
        dates.drop(columns=[f"{col}_validation" for col in replace_cols if col not in key], inplace=True)

    selected = pd.concat([baseline, dates], ignore_index=True)
    selected = validate_times(selected, require_complete_arms=True)
    out.mkdir(parents=True, exist_ok=True)
    times_path = out / "validation_sameframe_times.csv"
    selected.to_csv(times_path, index=False)
    analysis_out = out / "validation_sameframe_base"
    subprocess.run(
        [sys.executable, str(analyzer), "--times", str(times_path), "--out", str(analysis_out)],
        check=True, text=True,
    )
    absent_periods = [name for name, profiles in validation_profiles.items() if not profiles]
    return {
        "status": "validation_periods_absent" if absent_periods else "completed",
        "validation_periods_absent": absent_periods,
        "times_csv": str(times_path),
        "analysis_directory": str(analysis_out), "n_times_rows": int(len(selected)),
        "n_origins": int(selected[["city", "uid"]].drop_duplicates().shape[0]),
        "baseline_profiles": baseline_profiles,
        "validation_profiles": validation_profiles,
        "frame_files": [str(p) for p in files],
        "denominator": "baseline uses validationframe pop/design_weight/N_h/n_h/inclusion_prob/stratum and coordinates; validation weekday/weekend metadata is checked field-by-field and retained; route time/status comes from each official profile",
    }


def write_outputs(df: pd.DataFrame, out: Path, times_path: Path,
                  sets_dir: Path | None, analyzer: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "same_point_discordance.csv": build_discordances(df),
        "paired_scope_coverage_losses.csv": build_scope_losses(df),
        "coverage_curves_by_income_group.csv": build_coverage_curves_by_group(df),
        "duration_quantiles_returned_routes.csv": build_returned_route_quantiles(df),
        "car_sensitivity_curves.csv": build_car_sensitivity(df),
        "spatial_paired_origins.csv": build_spatial_paired(df),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    validation = (
        build_validation_sameframe(df, sets_dir, out, analyzer)
        if sets_dir is not None else {"status": "skipped", "reason": "--sets not provided"}
    )
    notes = {
        "input_times": str(times_path), "n_rows": int(len(df)),
        "n_origins_city_profile": int(df[["city", "profile", "uid"]].drop_duplicates().shape[0]),
        "outputs": {name: int(len(table)) for name, table in tables.items()},
        "validation_sameframe": validation,
        "design_variance": "all ratio estimates use v13_analyse.ratio_estimate; route coverage and technical unknowns use v13_analyse.coverage_vector",
        "technical_bounds": "lower/upper endpoints reflect finite_unknown route coverage bounds; design CI is only for the lower modeled estimate, not for total-population HT point totals or the technical upper bound",
        "ht_population": "population totals are point estimates sum(design_weight * indicator); no CI is reported for HT totals",
        "discordance": "same-UID descriptive event rates in percentage points; no causal interpretation",
        "scope_losses": "paired same-UID R->E and R->A losses; subgroup results use the sampling stratum labels",
        "income_labels": "Sao Paulo Q1-Q5 are income quintiles; Bogota stratum/estrato labels are sampling strata and are explicitly not income groups; missing/unknown rows remain in overall and are also reported separately",
        "duration_quantiles": "weighted p50/p90 condition on finite returned candidate-route times only; no-returned-route HT population is reported separately",
        "car_scenarios": {
            "main_factor_1_over_0_6": "input car time unchanged; input main time already carries the 1/0.6 factor",
            "free_flow_factor_1_0": "main time * 0.6",
            "intermediate_factor_1_over_0_8": "main time * 0.6 / 0.8",
            "reconstructed_base_plus_5_min": "main time * 0.6 + 5 minutes",
            "main_adjusted_plus_5_min": "input main car time + 5 minutes",
            "comparison": "each scenario reports car coverage and paired car-minus-WT coverage at every threshold; descriptive model sensitivity only",
        },
        "thresholds_min": THRESHOLDS,
        "analysis_scope": "post-processing only; no route requests/downloads; source times file is read-only",
    }
    (out / "ANALYSIS_NOTES.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    return notes


def _fixture() -> pd.DataFrame:
    # Four binary car/WT coverage quadrants at 30 minutes, plus paired foot.
    # One sampled stratum (n=4, N=8) yields nonzero design SE and a CI.
    quadrants = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)]
    rows: list[dict[str, Any]] = []
    profiles = [
        ("Sao Paulo", "baseline_fixture", FIXTURE_SCOPES, "Q1", 2.0, 8, .5),
        ("Bogota", "baseline_fixture", REQUIRED_SCOPES, "low", 2.0, 8, .5),
        # The same Bogota UIDs are valid under a different profile-specific
        # sampling frame, with different design weights and sample sizes.
        ("Bogota", "validation_fixture", REQUIRED_SCOPES, "low", 4.0, 16, .25),
        ("Sao Paulo", "validation_weekday", FIXTURE_SCOPES, "Q1", 3.0, 12, 1/3),
        ("Sao Paulo", "validation_weekend", FIXTURE_SCOPES, "Q1", 3.0, 12, 1/3),
        ("Bogota", "validation_weekday", REQUIRED_SCOPES, "low", 3.0, 12, 1/3),
        ("Bogota", "validation_weekend", REQUIRED_SCOPES, "low", 3.0, 12, 1/3),
    ]
    for uid_i, (car, wt, foot) in enumerate(quadrants, start=1):
        for city, profile, scopes, stratum, weight, nh, pi in profiles:
            uid = f"fixture-{uid_i}"
            for scope in sorted(scopes):
                for mode, covered in (("car", car), ("wt", wt), ("foot", foot)):
                    rows.append({
                        "city": city, "uid": uid,
                        "lat": -23.0 - uid_i / 100 if city == "Sao Paulo" else 4.6 + uid_i / 100,
                        "lon": -46.0 - uid_i / 100 if city == "Sao Paulo" else -74.1 - uid_i / 100,
                        "stratum": stratum, "pop": 1.0,
                        "design_weight": weight, "N_h": nh, "n_h": 4,
                        "inclusion_prob": pi, "scope": scope, "mode": mode,
                        "profile": profile, "time_min": 20.0 if covered else 40.0,
                        "status": "success", "finite_unknown": False,
                    })
    return pd.DataFrame(rows)


def run_self_test(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    df = validate_times(_fixture())
    times_path = out / "fixture_times.csv"
    df.to_csv(times_path, index=False)
    discordance = build_discordances(df)
    row = discordance[(discordance.scope == "R") & (discordance.threshold_min == 30) &
                      (discordance.event == "car_covers_wt_does_not")].iloc[0]
    foot_row = discordance[(discordance.scope == "R") & (discordance.threshold_min == 30) &
                           (discordance.event == "wt_covers_foot_does_not")].iloc[0]
    pair = df[(df.city == "Sao Paulo") & (df.profile == "baseline_fixture") &
              (df.scope == "R") & df["mode"].isin(["car", "wt", "foot"])].pivot(
        index="uid", columns="mode", values="time_min")
    quadrant_counts = {
        "car_yes_wt_no": int(((pair.car <= 30) & ~(pair.wt <= 30)).sum()),
        "car_no_wt_yes": int((~(pair.car <= 30) & (pair.wt <= 30)).sum()),
        "both_yes": int(((pair.car <= 30) & (pair.wt <= 30)).sum()),
        "both_no": int((~(pair.car <= 30) & ~(pair.wt <= 30)).sum()),
    }
    expected_se = math.sqrt(2.0) / 8.0 * 100.0
    checks = {
        "four_quadrants_present_once": quadrant_counts == {
            "car_yes_wt_no": 1, "car_no_wt_yes": 1,
            "both_yes": 1, "both_no": 1,
        },
        "car_only_estimate_25pct": math.isclose(float(row.estimate_pct), 25.0, abs_tol=1e-8),
        "paired_design_se_matches_v13_analyse": math.isclose(float(row.design_se_pp), expected_se, rel_tol=1e-8),
        "design_ci_is_not_suppressed": np.isfinite(row.design_ci_lower_pct) and np.isfinite(row.design_ci_upper_pct),
        "wt_foot_discordance_50pct": math.isclose(float(foot_row.estimate_pct), 50.0, abs_tol=1e-8),
        "no_technical_bound_width_without_unknowns": math.isclose(float(row.technical_lower_pct), float(row.technical_upper_pct), abs_tol=1e-8),
        "main_adjusted_plus_five_scenario_is_available": "main_adjusted_plus_5_min" in set(
            build_car_sensitivity(df[(df.city == "Sao Paulo") & (df.profile == "baseline_fixture") &
                                     (df.scope == "R")]).scenario
        ),
        "Bogota_without_A_scope_is_valid": "A" not in set(df.loc[df.city == "Bogota", "scope"]),
        "same_UID_across_profiles_can_have_different_design_weights": (
            df[(df.city == "Bogota") & (df.uid == "fixture-1")]
            .groupby("profile").design_weight.first().nunique() > 1
        ),
    }
    incomplete = df[~((df.city == "Sao Paulo") & (df.profile == "baseline_fixture") &
                      (df.scope == "E") & (df["mode"] == "wt") & (df.uid == "fixture-4"))]
    try:
        validate_times(incomplete)
        checks["missing_arm_uid_rejected"] = False
    except ValueError:
        checks["missing_arm_uid_rejected"] = True
    try:
        validate_times(pd.concat([df, df.iloc[[0]]], ignore_index=True))
        checks["duplicate_arm_rejected"] = False
    except ValueError:
        checks["duplicate_arm_rejected"] = True
    validation_dir = out / "fixture_sets"
    validation_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = df[(df.profile == "baseline_fixture")].drop_duplicates(["city", "uid"])
    frame_cols = ["city", "uid", "lat", "lon", "stratum", "pop", "design_weight",
                  "N_h", "n_h", "inclusion_prob"]
    sp_frame = frame_rows[frame_rows.city == "Sao Paulo"][frame_cols].copy()
    bo_frame = frame_rows[frame_rows.city == "Bogota"][frame_cols].copy()
    # The baseline uses a different design; each validation date must already
    # match the frame metadata and is checked without replacing its weights.
    sp_frame["design_weight"] = 3.0
    sp_frame["N_h"] = 12
    sp_frame["inclusion_prob"] = 1 / 3
    bo_frame["design_weight"] = 3.0
    bo_frame["N_h"] = 12
    bo_frame["inclusion_prob"] = 1 / 3
    sp_frame.to_csv(validation_dir / "validation_sp.csv", index=False)
    bo_frame.to_csv(validation_dir / "validation_bo.csv", index=False)
    validation_result = build_validation_sameframe(
        df, validation_dir, out / "validation_fixture_test",
        Path(__file__).with_name("v13_analyse.py").resolve(),
    )
    validation_times = pd.read_csv(validation_result["times_csv"], dtype={"uid": str})
    bo_validation_weight = validation_times.loc[validation_times.city == "Bogota", "design_weight"].unique()
    profile_counts = validation_times.groupby(["city", "uid"]).profile.nunique()
    checks["validation_sameframe_calls_analyzer_and_replaces_weights"] = (
        validation_result["status"] == "completed" and len(bo_validation_weight) == 1 and
        math.isclose(float(bo_validation_weight[0]), 3.0) and
        profile_counts.eq(3).all() and
        set(validation_result["validation_profiles"]["weekday"]) == {"validation_weekday"} and
        set(validation_result["validation_profiles"]["weekend"]) == {"validation_weekend"} and
        (Path(validation_result["analysis_directory"]) / "coverage_curves.csv").is_file()
    )
    baseline_only_result = build_validation_sameframe(
        df[df.profile == "baseline_fixture"].copy(), validation_dir,
        out / "validation_fixture_periods_absent_test",
        Path(__file__).with_name("v13_analyse.py").resolve(),
    )
    checks["absent_validation_dates_are_explicitly_marked"] = (
        baseline_only_result["status"] == "validation_periods_absent" and
        set(baseline_only_result["validation_periods_absent"]) == {"weekday", "weekend"}
    )
    bad_date_metadata = df.copy()
    bad_date_metadata.loc[
        (bad_date_metadata.city == "Sao Paulo") &
        (bad_date_metadata.profile == "validation_weekend") &
        (bad_date_metadata.uid == "fixture-1"), "design_weight"
    ] = 99.0
    try:
        build_validation_sameframe(
            bad_date_metadata, validation_dir, out / "validation_fixture_bad_metadata_test",
            Path(__file__).with_name("v13_analyse.py").resolve(),
        )
        checks["validation_date_metadata_mismatch_rejected"] = False
    except ValueError:
        checks["validation_date_metadata_mismatch_rejected"] = True
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "quadrant_counts": quadrant_counts,
        "expected_design_se_pp": expected_se,
        "observed_design_se_pp": float(row.design_se_pp),
        "observed_design_ci_pct": [float(row.design_ci_lower_pct), float(row.design_ci_upper_pct)],
        "input_fixture": str(times_path),
    }
    (out / "SELF_TEST_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", type=Path, help="input official or pilot times CSV")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--sets", type=Path, help="directory containing validation_sp.csv and validation_bo.csv")
    parser.add_argument("--self-test", action="store_true", help="run deterministic four-quadrant/design-CI fixture checks")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test(args.out)
    if args.times is None:
        parser.error("--times is required unless --self-test is used")
    analyzer = Path(__file__).with_name("v13_analyse.py").resolve()
    if not analyzer.is_file():
        raise FileNotFoundError(f"existing analyzer is required: {analyzer}")
    times_path = args.times.resolve()
    df = validate_times(pd.read_csv(times_path, dtype={"uid": str}))
    notes = write_outputs(df, args.out.resolve(), times_path, args.sets.resolve() if args.sets else None, analyzer)
    print(json.dumps({"status": "DONE", "out": str(args.out.resolve()), "outputs": notes["outputs"],
                      "validation_sameframe": notes["validation_sameframe"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
