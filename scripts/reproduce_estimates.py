#!/usr/bin/env python3
"""Reproduce primary origin-weighted route coverage and paired contrasts.

Run from any directory:
    python scripts/reproduce_estimates.py
Optional --times and --out paths override the repository-relative defaults.
Requires Python, numpy, and pandas.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = list(range(10, 61, 5))
REQUIRED = {
    "city", "uid", "stratum", "pop", "design_weight", "N_h", "n_h",
    "inclusion_prob", "scope", "mode", "profile", "time_min", "status",
    "finite_unknown",
}


def _as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).to_numpy(dtype=bool)
    truthy = {"1", "true", "yes"}
    falsy = {"", "0", "false", "no", "nan", "none"}
    out = []
    for value in series:
        if pd.isna(value):
            out.append(False)
            continue
        normalized = str(value).strip().lower()
        if normalized in truthy:
            out.append(True)
        elif normalized in falsy:
            out.append(False)
        else:
            raise ValueError(f"invalid finite_unknown value: {value!r}")
    return np.asarray(out, dtype=bool)


def ratio_estimate(d: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    w = d.design_weight.to_numpy(float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(w) & np.isfinite(y) & (w > 0)
    d = d.loc[good].reset_index(drop=True)
    w, y = w[good], y[good]
    if not len(w):
        return dict(estimate=np.nan, se=np.nan, lower=np.nan, upper=np.nan)
    den = w.sum()
    # Preserve exact constants, including all-zero or all-one outcomes.
    mu = float(y[0]) if np.all(y == y[0]) else np.dot(w, y) / den
    var = 0.0
    for _, ix in d.groupby("stratum", observed=True).groups.items():
        ix = np.asarray(list(ix))
        sub = d.iloc[ix]
        nh = len(ix)
        Nh = float(sub.N_h.iloc[0])
        if nh >= Nh:
            continue
        if nh < 2:
            return dict(estimate=100 * mu, se=np.nan, lower=np.nan, upper=np.nan)
        e = sub["pop"].to_numpy(float) * (y[ix] - mu)
        var += Nh**2 * (1 - nh / Nh) * np.var(e, ddof=1) / nh
    se = math.sqrt(max(0, var)) / den
    census = all(
        len(ix) >= float(d.loc[list(ix), "N_h"].iloc[0])
        for ix in d.groupby("stratum", observed=True).groups.values()
    )
    suppress_interval = census or se == 0
    return dict(
        estimate=100 * mu,
        se=100 * se,
        lower=np.nan if suppress_interval else 100 * (mu - 1.96 * se),
        upper=np.nan if suppress_interval else 100 * (mu + 1.96 * se),
    )


def coverage_vector(d: pd.DataFrame, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    t = pd.to_numeric(d.time_min, errors="coerce").to_numpy(float)
    covered = (np.isfinite(t) & (t <= threshold)).astype(float)
    if "finite_unknown" in d:
        unknown = _as_bool(d.finite_unknown)
    else:
        unknown = d.status.isin(
            ["technical_error", "partial_error", "not_queried"]
        ).to_numpy()
    upper = np.maximum(covered, unknown.astype(float))
    return covered, upper


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    out = df.copy()
    for col in ("city", "uid", "scope", "mode", "profile", "status"):
        out[col] = out[col].astype("string")
        if out[col].isna().any() or out[col].str.strip().eq("").any():
            raise ValueError(f"null/empty values are not allowed in {col}")
        out[col] = out[col].astype(str)
    out["stratum"] = out["stratum"].fillna("unknown").astype(str)
    keys = ["city", "uid", "scope", "mode", "profile"]
    if out.duplicated(keys).any():
        raise ValueError("duplicate origin-arm rows")
    for col in ("pop", "design_weight", "N_h", "n_h", "inclusion_prob"):
        out[col] = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(out[col].to_numpy(float)).all():
            raise ValueError(f"non-finite values in {col}")
    if (out["pop"] <= 0).any() or (out["design_weight"] <= 0).any():
        raise ValueError("pop and design_weight must be positive")
    if (out["N_h"] <= 0).any() or (out["n_h"] <= 0).any():
        raise ValueError("N_h and n_h must be positive")
    if (out["inclusion_prob"] <= 0).any() or (out["inclusion_prob"] > 1).any():
        raise ValueError("inclusion_prob must be in (0, 1]")
    parsed = pd.to_numeric(out.time_min, errors="coerce")
    if (out.time_min.notna() & parsed.isna()).any():
        raise ValueError("time_min contains nonnumeric non-missing values")
    if (parsed.dropna() < 0).any():
        raise ValueError("time_min must be nonnegative when present")
    out["time_min"] = parsed
    out["finite_unknown"] = _as_bool(out.finite_unknown)
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    default_data = repo / "data" / "estimation" / "times_primary.csv"
    default_out = repo / "artifacts" / "estimation"
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", default=str(default_data))
    ap.add_argument("--output-dir", "--out", dest="out", default=str(default_out))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = validate(pd.read_csv(args.times, dtype={"uid": str}, low_memory=False))
    curves = []
    interactions = []
    for arm, d0 in df.groupby(["city", "scope", "mode", "profile"], sort=True):
        d = d0.reset_index(drop=True)
        city, scope, mode, profile = arm
        census = all(
            len(g) >= float(g.N_h.iloc[0])
            for _, g in d.groupby("stratum", observed=True)
        )
        base = dict(
            city=city,
            scope=scope,
            mode=mode,
            profile=profile,
            n=len(d),
            sampling_design="census" if census else "stratified_SRS",
            sampled_population=float(d["pop"].sum()),
            estimated_population=float(d.design_weight.sum()),
        )
        for threshold in THRESHOLDS:
            lower, upper = coverage_vector(d, threshold)
            point = ratio_estimate(d, lower)
            upper_point = ratio_estimate(d, upper)["estimate"]
            curves.append(
                dict(
                    **base,
                    threshold=threshold,
                    coverage_lower=point["estimate"],
                    coverage_upper=upper_point,
                    design_se=point["se"],
                    design_ci_lower=max(0, point["lower"])
                    if np.isfinite(point["lower"]) else np.nan,
                    design_ci_upper=min(100, point["upper"])
                    if np.isfinite(point["upper"]) else np.nan,
                )
            )

    # Paired additive interaction: (R-E)_WT - (R-E)_car, evaluated on
    # the same sampled origins and with the original stratified design CI.
    for (city, profile), dd in df.groupby(["city", "profile"], sort=True):
        for scope2 in ("E", "A"):
            parts = []
            for scope, mode, sign in (
                ("R", "wt", 1), (scope2, "wt", -1),
                ("R", "car", -1), (scope2, "car", 1),
            ):
                arm = dd[(dd.scope == scope) & (dd["mode"] == mode)].copy()
                if arm.empty:
                    break
                parts.append((arm.set_index("uid"), sign))
            if len(parts) != 4:
                continue
            idx = parts[0][0].index
            if any(set(z.index) != set(idx) for z, _ in parts):
                raise ValueError("paired contrast arms have different origin sets")
            base = parts[0][0].loc[idx].reset_index()
            is_census = all(
                len(g) >= float(g.N_h.iloc[0])
                for _, g in base.groupby("stratum", observed=True)
            )
            for threshold in THRESHOLDS:
                y = np.zeros(len(idx))
                bound_lower = np.zeros(len(idx))
                bound_upper = np.zeros(len(idx))
                for z, sign in parts:
                    lo, hi = coverage_vector(z.loc[idx].reset_index(), threshold)
                    y += sign * lo
                    bound_lower += sign * (lo if sign > 0 else hi)
                    bound_upper += sign * (hi if sign > 0 else lo)
                est = ratio_estimate(base, y)
                interactions.append(
                    dict(
                        city=city,
                        profile=profile,
                        scope1="R",
                        scope2=scope2,
                        sampling_design="census" if is_census else "stratified_SRS",
                        threshold=threshold,
                        contrast=est["estimate"],
                        design_se=est["se"],
                        design_ci_lower=est["lower"],
                        design_ci_upper=est["upper"],
                        missing_bound_lower=ratio_estimate(base, bound_lower)["estimate"],
                        missing_bound_upper=ratio_estimate(base, bound_upper)["estimate"],
                    )
                )

    pd.DataFrame(curves).to_csv(out / "coverage_curves.csv", index=False)
    pd.DataFrame(interactions).to_csv(out / "mode_scope_contrasts.csv", index=False)
    (out / "ANALYSIS_NOTES.json").write_text(
        json.dumps(
            {
                "independent_unit": "sampled geographic origin",
                "coverage": "conditional on queried candidate routes and frozen input network",
                "design": "stratified SRS ratio linearization with finite-population correction",
                "ci": "pointwise design interval; not a model/input uncertainty interval",
                "thresholds_min": THRESHOLDS,
                "paired_contrast": "descriptive additive mode-by-scope contrast, not causal",
                "n_origin_arm_rows": int(len(df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"DONE: {len(curves)} coverage rows; {len(interactions)} paired contrasts; "
        f"{len(df)} origin-arm inputs",
        flush=True,
    )


if __name__ == "__main__":
    main()
