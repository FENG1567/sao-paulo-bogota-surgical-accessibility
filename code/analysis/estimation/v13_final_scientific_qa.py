"""Strict release gate for completed V13 production estimates; run on server."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from v13_analyse import coverage_vector, ratio_estimate


CITIES = ("Sao Paulo", "Bogota")
PROFILES = {
    "baseline_fixed_departure": ("main", {"Sao Paulo": 1100, "Bogota": 1057}),
    "validation_weekday": ("validation", {"Sao Paulo": 220, "Bogota": 200}),
    "validation_weekend": ("validation", {"Sao Paulo": 220, "Bogota": 200}),
}
SCOPES = {"Sao Paulo": {"R", "E", "A"}, "Bogota": {"R", "E"}}
MODES = {"car", "foot", "wt", "wt_cap1km", "wt_cap2km", "wt_cap3km"}
THRESHOLDS = tuple(range(10, 61, 5))
ARM_KEY = ["city", "uid", "profile", "scope", "mode"]
CURVE_KEY = ["city", "profile", "scope", "mode", "threshold"]
DESIGN_NUMERIC = ["pop", "design_weight", "N_h", "n_h", "inclusion_prob"]
ROUTE_STATUSES = {
    "ok", "no_returned_feasible_candidate", "technical_error", "partial_error", "not_queried"
}
TECHNICAL_STATUSES = {"technical_error", "partial_error", "not_queried"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise ValueError(message)


def _numeric(frame: pd.DataFrame, column: str, *, allow_missing: bool = False) -> np.ndarray:
    if column not in frame:
        _fail(f"missing required numeric column: {column}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric column: {column}") from exc
    if not np.isfinite(values[~np.isnan(values)]).all():
        _fail(f"nonfinite values in {column}")
    if not allow_missing and not np.isfinite(values).all():
        _fail(f"missing/nonfinite values in {column}")
    return values


def _strict_bool(series: pd.Series, column: str) -> np.ndarray:
    result = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif isinstance(value, (int, np.integer, float, np.floating)) and value in (0, 1):
            result.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            result.append(value.strip().lower() == "true")
        else:
            _fail(f"{column} must contain only true/false values; got {value!r}")
    return np.asarray(result, dtype=bool)


def _validate_design_frame(frame: pd.DataFrame, city: str, name: str,
                           expected_n: int | None = None) -> pd.DataFrame:
    required = {"city", "uid", "stratum", *DESIGN_NUMERIC}
    absent = sorted(required - set(frame.columns))
    if absent:
        _fail(f"{name}/{city} frame misses columns {absent}")
    out = frame.copy()
    if out.empty or out.city.isna().any() or set(out.city.astype(str)) != {city}:
        _fail(f"{name}/{city} frame has empty or unexpected city rows")
    if out.uid.isna().any() or out.uid.astype(str).str.strip().eq("").any():
        _fail(f"{name}/{city} frame has missing UID")
    out["uid"] = out.uid.astype(str)
    out["stratum"] = out.stratum.fillna("unknown").astype(str)
    if out.uid.duplicated().any():
        _fail(f"{name}/{city} frame has duplicate UID")
    if expected_n is not None and len(out) != expected_n:
        _fail(f"{name}/{city} frame has {len(out)} rows; expected {expected_n}")
    for col in DESIGN_NUMERIC:
        out[col] = _numeric(out, col)
    if ((out[["pop", "design_weight", "N_h", "n_h", "inclusion_prob"]] <= 0).any().any()):
        _fail(f"{name}/{city} frame has nonpositive design metadata")
    for col in ("N_h", "n_h"):
        values = out[col].to_numpy(float)
        if not np.equal(values, np.floor(values)).all():
            _fail(f"{name}/{city} frame has noninteger {col}")
    if (out.n_h > out.N_h).any():
        _fail(f"{name}/{city} frame has n_h > N_h")
    for stratum, group in out.groupby("stratum", dropna=False, sort=False):
        if group.N_h.nunique() != 1 or group.n_h.nunique() != 1:
            _fail(f"{name}/{city}/{stratum} has inconsistent N_h/n_h")
        expected_n_h = int(group.n_h.iloc[0])
        if len(group) != expected_n_h:
            _fail(f"{name}/{city}/{stratum} has {len(group)} observed rows; n_h={expected_n_h}")
    expected_pi = out.n_h.to_numpy(float) / out.N_h.to_numpy(float)
    if not np.allclose(out.inclusion_prob.to_numpy(float), expected_pi, rtol=1e-9, atol=1e-12):
        _fail(f"{name}/{city} inclusion_prob != n_h/N_h")
    expected_weight = out["pop"].to_numpy(float) / out.inclusion_prob.to_numpy(float)
    if not np.allclose(out.design_weight.to_numpy(float), expected_weight, rtol=1e-9, atol=1e-8):
        _fail(f"{name}/{city} design_weight != pop/inclusion_prob")
    for col in ("lat", "lon"):
        if col in out:
            out[col] = _numeric(out, col)
    return out


def validate_times(data: pd.DataFrame, *, frozen_frames: dict[tuple[str, str], pd.DataFrame] | None = None,
                   expected_counts: dict[str, dict[str, int]] | None = None,
                   require_no_technical: bool = True) -> pd.DataFrame:
    """Validate the exact origin-arm inventory and frozen survey design."""
    expected_counts = expected_counts or {p: counts.copy() for p, (_, counts) in PROFILES.items()}
    required = {
        *ARM_KEY, "stratum", *DESIGN_NUMERIC, "time_min", "status", "finite_unknown"
    }
    absent = sorted(required - set(data.columns))
    if absent:
        _fail(f"official times miss columns {absent}")
    out = data.copy()
    for col in ["city", "uid", "profile", "scope", "mode"]:
        if out[col].isna().any() or out[col].astype(str).str.strip().eq("").any():
            _fail(f"official times have missing {col}")
        out[col] = out[col].astype(str)
    out["stratum"] = out.stratum.fillna("unknown").astype(str)
    if out.duplicated(ARM_KEY).any():
        _fail("duplicate origin-arm rows")
    if set(out.city) != set(CITIES):
        _fail(f"city inventory differs: {sorted(set(out.city))}")
    if set(out.profile) != set(expected_counts):
        _fail(f"profile inventory differs: {sorted(set(out.profile))}")
    if out["status"].isna().any():
        _fail("missing route status")
    out["status"] = out.status.astype(str)
    unknown_status = set(out.status) - ROUTE_STATUSES
    if unknown_status:
        _fail(f"unknown route statuses: {sorted(unknown_status)}")
    out["finite_unknown"] = _strict_bool(out.finite_unknown, "finite_unknown")
    out["time_min"] = _numeric(out, "time_min", allow_missing=True)
    if (out.time_min.dropna() < 0).any():
        _fail("negative returned time")
    has_time = out.time_min.notna().to_numpy()
    expected_technical = out.status.isin(TECHNICAL_STATUSES).to_numpy()
    status_requires_time = out.status.isin({"ok", "partial_error"}).to_numpy()
    if not np.array_equal(has_time, status_requires_time):
        _fail("route status and returned-time availability disagree")
    if not np.array_equal(out.finite_unknown.to_numpy(bool), expected_technical):
        _fail("finite_unknown disagrees with route status")
    if require_no_technical and out.finite_unknown.any():
        _fail("technical route failures remain; supplementary summaries are not allowed")
    for col in DESIGN_NUMERIC:
        out[col] = _numeric(out, col)

    frames_checked: dict[tuple[str, str], pd.DataFrame] = {}
    if frozen_frames is not None:
        for key, frame in frozen_frames.items():
            frame_name, city = key
            profile_for_frame = next((p for p, (f, _) in PROFILES.items() if f == frame_name), None)
            expected_n = expected_counts.get(profile_for_frame, {}).get(city) if profile_for_frame else None
            frames_checked[key] = _validate_design_frame(frame, city, frame_name, expected_n)

    expected_arms = set()
    for profile, counts in expected_counts.items():
        if set(counts) != set(CITIES):
            _fail(f"{profile} expected-count city set differs")
        frame_name = PROFILES.get(profile, ("main", {}))[0]
        for city, expected_n in counts.items():
            z = out[(out.profile == profile) & (out.city == city)]
            if z.empty:
                _fail(f"missing {profile}/{city} rows")
            if frozen_frames is not None:
                frame = frames_checked[(frame_name, city)]
                expected_uids = set(frame.uid)
            else:
                expected_uids = set(z.uid)
            if len(expected_uids) != expected_n:
                _fail(f"{profile}/{city} UID count differs from expected {expected_n}")
            if set(z.uid) != expected_uids:
                _fail(f"{profile}/{city} UID inventory differs from frozen frame")
            expected_scopes = SCOPES[city]
            observed_scopes = set(z.scope)
            if observed_scopes != expected_scopes:
                _fail(f"{profile}/{city} scope inventory differs: {sorted(observed_scopes)}")
            for scope in expected_scopes:
                for mode in MODES:
                    arm = z[(z.scope == scope) & (z["mode"] == mode)]
                    if len(arm) != expected_n or set(arm.uid) != expected_uids:
                        _fail(f"incomplete {profile}/{city}/{scope}/{mode} UID denominator")
                    expected_arms.update((city, uid, profile, scope, mode) for uid in expected_uids)
                    if frozen_frames is not None:
                        frame = frames_checked[(frame_name, city)].set_index("uid")
                        indexed = arm.set_index("uid")
                        for col in ["stratum", *DESIGN_NUMERIC, "lat", "lon"]:
                            if col not in frame or col not in indexed:
                                if col in {"lat", "lon"} and col not in frame and col not in indexed:
                                    continue
                                if col in {"lat", "lon"}:
                                    continue
                                _fail(f"{profile}/{city}/{scope}/{mode} misses metadata {col}")
                            if col == "stratum":
                                equal = indexed[col].fillna("unknown").astype(str).reindex(frame.index).to_numpy() == frame[col].fillna("unknown").astype(str).to_numpy()
                            else:
                                actual = pd.to_numeric(indexed[col].reindex(frame.index), errors="coerce").to_numpy(float)
                                reference = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
                                equal = np.isclose(actual, reference, rtol=1e-9, atol=1e-8, equal_nan=False)
                            if not bool(np.all(equal)):
                                _fail(f"{profile}/{city}/{scope}/{mode} frozen metadata mismatch: {col}")
    observed_arms = set(map(tuple, out[ARM_KEY].itertuples(index=False, name=None)))
    if observed_arms != expected_arms:
        _fail("official origin-arm inventory is not the exact expected Cartesian inventory")
    # Validate the frozen design identities within the official long file too.
    if ((out[["pop", "design_weight", "N_h", "n_h", "inclusion_prob"]] <= 0).any().any()):
        _fail("official times have nonpositive design metadata")
    for _, group in out.groupby(["profile", "city", "scope", "mode", "stratum"], sort=False):
        if group.N_h.nunique() != 1 or group.n_h.nunique() != 1:
            _fail("official times have inconsistent stratum N_h/n_h")
        if not np.equal(group[["N_h", "n_h"]].to_numpy(float), np.floor(group[["N_h", "n_h"]].to_numpy(float))).all():
            _fail("official times have noninteger N_h/n_h")
        if len(group) != int(group.n_h.iloc[0]) or float(group.n_h.iloc[0]) > float(group.N_h.iloc[0]):
            _fail("official times do not match stratum sample sizes")
    pi = out.n_h.to_numpy(float) / out.N_h.to_numpy(float)
    if not np.allclose(out.inclusion_prob.to_numpy(float), pi, rtol=1e-9, atol=1e-12):
        _fail("official inclusion_prob != n_h/N_h")
    if not np.allclose(out.design_weight.to_numpy(float), out["pop"].to_numpy(float) / pi,
                       rtol=1e-9, atol=1e-8):
        _fail("official design_weight != pop/inclusion_prob")
    bogota_baseline = out[(out.city == "Bogota") &
                          (out.profile == "baseline_fixed_departure")]
    if bogota_baseline.empty:
        _fail("Bogota baseline census frame is absent")
    for _, arm in bogota_baseline.groupby(["scope", "mode"], sort=False):
        if not _is_census(arm):
            _fail("Bogota baseline must be explicitly census by stratum in every arm")
    _validate_nested_routes(out, require_no_technical=require_no_technical)
    return out


def validate_frozen_set_manifest(sets_dir: Path, filenames: list[str]) -> Path:
    manifest_path = sets_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("files", {})
    for filename in filenames:
        path = sets_dir / filename
        entry = entries.get(filename)
        if not path.is_file() or not isinstance(entry, dict):
            _fail(f"frozen-set manifest is missing {filename}")
        if entry.get("sha256") != sha(path):
            _fail(f"frozen-set manifest SHA-256 mismatch: {filename}")
        if "bytes" in entry and int(entry["bytes"]) != path.stat().st_size:
            _fail(f"frozen-set manifest byte-count mismatch: {filename}")
    return manifest_path


def load_official_times(root: Path, *, require_no_technical: bool = True,
                        expected_counts: dict[str, dict[str, int]] | None = None) -> tuple[pd.DataFrame, Path]:
    core = root / "core"
    expected_counts = expected_counts or {p: counts.copy() for p, (_, counts) in PROFILES.items()}
    frames = {}
    filenames = []
    for frame_name in {name for name, _ in PROFILES.values()}:
        for city, tag in (("Sao Paulo", "sp"), ("Bogota", "bo")):
            filename = f"{frame_name}_{tag}.csv"
            filenames.append(filename)
            frames[(frame_name, city)] = pd.read_csv(core / "sets" / filename, dtype={"uid": str})
    validate_frozen_set_manifest(core / "sets", filenames)
    source = core / "times_official.csv"
    data = pd.read_csv(source, dtype={"uid": str})
    return validate_times(data, frozen_frames=frames, expected_counts=expected_counts,
                          require_no_technical=require_no_technical), source


def _is_census(frame: pd.DataFrame) -> bool:
    return all(len(g) >= float(g.N_h.iloc[0]) for _, g in frame.groupby("stratum", observed=True))


def _validate_nested_routes(data: pd.DataFrame, *, require_no_technical: bool) -> None:
    tolerance = 1e-8
    for (city, profile), frame in data.groupby(["city", "profile"], sort=True):
        times = frame.pivot(index=["uid", "scope"], columns="mode", values="time_min")
        unknown = frame.pivot(index=["uid", "scope"], columns="mode", values="finite_unknown")
        for tight, loose in [("wt_cap1km", "wt_cap2km"),
                             ("wt_cap2km", "wt_cap3km"),
                             ("wt_cap3km", "wt"), ("foot", "wt")]:
            t = times[tight].to_numpy(float)
            l = times[loose].to_numpy(float)
            bad = np.isfinite(t) & (~np.isfinite(l) | (l > t + tolerance))
            if not require_no_technical:
                pair_unknown = (unknown[tight].to_numpy(bool) | unknown[loose].to_numpy(bool))
                bad &= ~pair_unknown
            if bad.any():
                _fail(f"{city}/{profile} nested-mode invariant failed: {tight} must include {loose}")

        times_by_scope = frame.pivot(index=["uid", "mode"], columns="scope", values="time_min")
        unknown_by_scope = frame.pivot(index=["uid", "mode"], columns="scope", values="finite_unknown")
        for scope in ("E", "A"):
            if scope not in times_by_scope.columns:
                continue
            r = times_by_scope["R"].to_numpy(float)
            sub = times_by_scope[scope].to_numpy(float)
            bad = np.isfinite(sub) & (~np.isfinite(r) | (r > sub + tolerance))
            if not require_no_technical:
                pair_unknown = (unknown_by_scope["R"].to_numpy(bool) |
                                unknown_by_scope[scope].to_numpy(bool))
                bad &= ~pair_unknown
            if bad.any():
                _fail(f"{city}/{profile} R-superset invariant failed for {scope}")


def expected_curve_rows(data: pd.DataFrame) -> list[dict]:
    """Recompute the analyzer's curve rows directly from validated official times."""
    rows = []
    for arm_key, frame in data.groupby(["city", "profile", "scope", "mode"], sort=True):
        city, profile, scope, mode = arm_key
        frame = frame.reset_index(drop=True)
        census = _is_census(frame)
        base = dict(city=city, profile=profile, scope=scope, mode=mode, n=len(frame),
                    sampling_design="census" if census else "stratified_SRS",
                    sampled_population=float(frame["pop"].sum()),
                    estimated_population=float(frame.design_weight.sum()))
        for threshold in THRESHOLDS:
            lower, upper = coverage_vector(frame, threshold)
            estimate = ratio_estimate(frame, lower)
            upper_estimate = ratio_estimate(frame, upper)["estimate"]
            rows.append(dict(
                **base, threshold=threshold, coverage_lower=estimate["estimate"],
                coverage_upper=upper_estimate, design_se=estimate["se"],
                design_ci_lower=max(0, estimate["lower"]) if np.isfinite(estimate["lower"]) else np.nan,
                design_ci_upper=min(100, estimate["upper"]) if np.isfinite(estimate["upper"]) else np.nan,
            ))
    return rows


def validate_curve_export(data: pd.DataFrame, curves: pd.DataFrame) -> None:
    required = {*CURVE_KEY, "n", "sampling_design", "sampled_population", "estimated_population",
                "coverage_lower", "coverage_upper", "design_se", "design_ci_lower", "design_ci_upper"}
    absent = sorted(required - set(curves.columns))
    if absent:
        _fail(f"coverage curves miss columns {absent}")
    actual = curves.copy()
    for col in CURVE_KEY:
        if actual[col].isna().any():
            _fail(f"coverage curves have missing key {col}")
    actual["threshold"] = _numeric(actual, "threshold")
    if not np.equal(actual.threshold, np.floor(actual.threshold)).all():
        _fail("coverage thresholds are noninteger")
    if actual.duplicated(CURVE_KEY).any():
        _fail("duplicate coverage curve keys")
    expected = pd.DataFrame(expected_curve_rows(data))
    expected_keys = set(map(tuple, expected[CURVE_KEY].itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, actual[CURVE_KEY].itertuples(index=False, name=None)))
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        extra = len(actual_keys - expected_keys)
        _fail(f"coverage curve arm/threshold inventory incomplete (missing={missing}, extra={extra})")
    expected = expected.set_index(CURVE_KEY).sort_index()
    actual = actual.set_index(CURVE_KEY).sort_index().reindex(expected.index)
    if not np.array_equal(actual.sampling_design.astype(str).to_numpy(), expected.sampling_design.astype(str).to_numpy()):
        _fail("coverage curve sampling_design differs from official times")
    for col in ["n", "sampled_population", "estimated_population", "coverage_lower", "coverage_upper",
                "design_se", "design_ci_lower", "design_ci_upper"]:
        got = pd.to_numeric(actual[col], errors="coerce").to_numpy(float)
        want = pd.to_numeric(expected[col], errors="coerce").to_numpy(float)
        if np.isinf(got).any() or not np.isclose(got, want, rtol=1e-9, atol=1e-8, equal_nan=True).all():
            _fail(f"coverage curve {col} is stale or inconsistent with official times")
    # Specifically guard the Bogota baseline census: suppressed CIs are required.
    bo_baseline = actual.reset_index()
    bo_baseline = bo_baseline[(bo_baseline.city == "Bogota") &
                              (bo_baseline.profile == "baseline_fixed_departure")]
    if not bo_baseline.empty and (bo_baseline.design_ci_lower.notna().any() or bo_baseline.design_ci_upper.notna().any()):
        _fail("Bogota baseline census curves must not report pseudo confidence intervals")


def validate_spatial_export(data: pd.DataFrame, spatial: pd.DataFrame) -> None:
    baseline = data[data.profile == "baseline_fixed_departure"].copy()
    spatial_metadata = ["lat", "lon"] if {"lat", "lon"}.issubset(baseline.columns) else []
    required = {*ARM_KEY, "stratum", *DESIGN_NUMERIC, "time_min", "status", "finite_unknown", "covered30",
                *spatial_metadata}
    absent = sorted(required - set(spatial.columns))
    if absent:
        _fail(f"spatial export misses columns {absent}")
    if spatial.duplicated(ARM_KEY).any():
        _fail("duplicate spatial origin-arm rows")
    expected_keys = set(map(tuple, baseline[ARM_KEY].itertuples(index=False, name=None)))
    observed_keys = set(map(tuple, spatial[ARM_KEY].itertuples(index=False, name=None)))
    if observed_keys != expected_keys:
        _fail("spatial export baseline origin-arm inventory differs from official times")
    expected = baseline.set_index(ARM_KEY).sort_index()
    actual = spatial.set_index(ARM_KEY).sort_index().reindex(expected.index)
    for col in ["stratum", *DESIGN_NUMERIC, *spatial_metadata, "time_min", "status"]:
        left = actual[col].fillna("unknown").astype(str).to_numpy() if col in {"stratum", "status"} else pd.to_numeric(actual[col], errors="coerce").to_numpy(float)
        right = expected[col].fillna("unknown").astype(str).to_numpy() if col in {"stratum", "status"} else pd.to_numeric(expected[col], errors="coerce").to_numpy(float)
        if col in {"stratum", "status"}:
            equal = left == right
        else:
            equal = np.isclose(left, right, rtol=1e-9, atol=1e-8, equal_nan=True)
        if not bool(np.all(equal)):
            _fail(f"spatial export is stale/inconsistent with official times: {col}")
    expected_unknown = _strict_bool(expected.finite_unknown, "official finite_unknown")
    actual_unknown = _strict_bool(actual.finite_unknown, "spatial finite_unknown")
    if not np.array_equal(actual_unknown, expected_unknown):
        _fail("spatial finite_unknown values disagree with official times")
    expected_coverage = np.isfinite(expected.time_min.to_numpy(float)) & (expected.time_min.to_numpy(float) <= 30)
    actual_coverage = _strict_bool(actual.covered30, "covered30")
    if not np.array_equal(actual_coverage, expected_coverage):
        _fail("spatial covered30 values disagree with official times")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    out = args.out
    core = root / "core"
    checks = {}
    evidence = {}

    def check(name, ok, detail=None):
        checks[name] = bool(ok)
        if detail is not None:
            evidence[name] = detail
        if not ok:
            print("FAIL", name, detail, flush=True)

    for name, path in [("primary", core / "pipeline.status.json"),
                       ("secondary", root / "secondary/status.json")]:
        status = json.loads(path.read_text())
        check(name + "_complete", status.get("status") == "done", status)
    if not all(checks.values()):
        raise RuntimeError("Production analyses incomplete")
    active = json.loads((root / "admin/ACTIVE_ROUTER_MANIFEST.json").read_text())
    check("complete_network_active", active.get("status") == "active_complete_networks")
    base = json.loads((core / "config_complete_baseline.json").read_text())
    for tag, key in [("sp", "SP_graph_sha256"), ("bo", "BO_graph_sha256")]:
        actual_hash = sha(root / f"graph_{tag}_complete/graph.obj")
        check("graph_" + tag + "_hash", actual_hash == base["router_profile"][key], actual_hash)

    data, times_path = load_official_times(root)
    check("official_times_exact_inventory_design_and_status", True, {"n_rows": len(data)})
    curves_path = root / "results/coverage_curves.csv"
    curves = pd.read_csv(curves_path)
    validate_curve_export(data, curves)
    check("coverage_curves_recomputed_complete", True, {"n_rows": len(curves)})
    spatial_path = root / "results/spatial_origin_arms.csv"
    spatial = pd.read_csv(spatial_path, dtype={"uid": str})
    validate_spatial_export(data, spatial)
    check("spatial_export_reconciled", True, {"n_rows": len(spatial)})

    output = {
        "status": "PASS" if all(checks.values()) else "FAIL", "pid": os.getpid(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks, "evidence": evidence,
        "official_times_sha256": sha(times_path), "coverage_curves_sha256": sha(curves_path),
        "frozen_set_manifest_sha256": sha(core / "sets/manifest.json"),
        "note": "Design and structural checks; no claim of observed route accuracy or universal destination optimality.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_scientific_qa.json").write_text(json.dumps(output, indent=2))
    print(output["status"], len(checks), "checks", flush=True)
    if output["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
