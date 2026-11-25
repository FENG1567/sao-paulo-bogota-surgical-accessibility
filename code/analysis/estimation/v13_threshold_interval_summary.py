"""Exploratory area summaries of the fixed 10--60 minute baseline curves."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = tuple(range(10, 61, 5))
PROFILE = "baseline_fixed_departure"
MODES = ("car", "foot", "wt", "wt_cap1km", "wt_cap2km", "wt_cap3km")
SCOPES = {"Sao Paulo": ("R", "E", "A"), "Bogota": ("R", "E")}
COUNTS = {"Sao Paulo": 1100, "Bogota": 1057}
DESIGN = ("stratum", "pop", "design_weight", "N_h", "n_h", "inclusion_prob")
ARM_KEY = ("city", "scope", "mode", "profile")
CURVE_KEY = ("city", "scope", "mode", "profile", "threshold")
EXPLORATORY_NOTE = (
    "Exploratory summary specified after baseline inspection, not preregistered. "
    "The fixed 10--60 minute grid was already chosen; no cutoff selection or per-cutoff "
    "discovery claim is made. Area averages can conceal threshold-curve crossings; "
    "the complete pointwise curves remain the source and are checked numerically."
)


def _area(values: np.ndarray, thresholds=THRESHOLDS) -> np.ndarray:
    return np.trapezoid(np.asarray(values, dtype=float), x=np.asarray(thresholds, dtype=float), axis=-1) / 50.0


def arm_area(frame, ratio_estimate, coverage_vector):
    frame = frame.sort_values("uid", kind="stable").reset_index(drop=True)
    low, high, point_curve, upper_curve = [], [], [], []
    for tau in THRESHOLDS:
        lo, hi = coverage_vector(frame, tau)
        low.append(lo); high.append(hi)
        point_curve.append(ratio_estimate(frame, lo)["estimate"])
        upper_curve.append(ratio_estimate(frame, hi)["estimate"])
    lo_y = _area(np.column_stack(low))
    hi_y = _area(np.column_stack(high))
    point = ratio_estimate(frame, lo_y)
    upper = ratio_estimate(frame, hi_y)
    return {"frame": frame, "lower_y": lo_y, "upper_y": hi_y, "estimate": point,
            "upper_estimate": upper["estimate"], "lower_curve": point_curve,
            "upper_curve": upper_curve}


def paired_area(arms, scope2, ratio_estimate, coverage_vector):
    """Area of paired D=R-WT - S-WT - R-car + S-car, including joint bounds."""
    parts = [(arms[("R", "wt")], 1), (arms[(scope2, "wt")], -1),
             (arms[("R", "car")], -1), (arms[(scope2, "car")], 1)]
    indexed = [(f.set_index("uid", drop=False), sign) for f, sign in parts]
    ids = indexed[0][0].index
    if ids.has_duplicates or any(not x.index.is_unique or set(x.index) != set(ids) for x, _ in indexed[1:]):
        raise ValueError("paired contrast arms must have identical unique UID sets")
    base = indexed[0][0].loc[sorted(ids)].reset_index(drop=True)
    for other, _ in indexed[1:]:
        other = other.loc[base.uid]
        for col in DESIGN:
            a, b = base[col], other[col]
            if col == "stratum":
                same = a.fillna("unknown").astype(str).to_numpy() == b.fillna("unknown").astype(str).to_numpy()
            else:
                same = np.isclose(pd.to_numeric(a, errors="raise"),
                                  pd.to_numeric(b, errors="raise"), rtol=1e-9, atol=1e-8)
            if not bool(np.all(same)):
                raise ValueError(f"paired origin design metadata differs: {col}")
    lows, highs, point_curve, lower_curve, upper_curve = [], [], [], [], []
    for tau in THRESHOLDS:
        y = np.zeros(len(base)); joint_lo = np.zeros(len(base)); joint_hi = np.zeros(len(base))
        for frame, sign in indexed:
            aligned = frame.loc[base.uid].reset_index(drop=True)
            lo, hi = coverage_vector(aligned, tau)
            y += sign * lo
            joint_lo += sign * (lo if sign > 0 else hi)
            joint_hi += sign * (hi if sign > 0 else lo)
        point_curve.append(ratio_estimate(base, y)["estimate"])
        lower_curve.append(ratio_estimate(base, joint_lo)["estimate"])
        upper_curve.append(ratio_estimate(base, joint_hi)["estimate"])
        lows.append(joint_lo); highs.append(joint_hi)
    point_y = _area(np.column_stack([sum(sign * coverage_vector(f.loc[base.uid].reset_index(drop=True), tau)[0]
                                                   for f, sign in indexed)
                                     for tau in THRESHOLDS]))
    lower_y, upper_y = _area(np.column_stack(lows)), _area(np.column_stack(highs))
    estimate = ratio_estimate(base, point_y)
    lower = ratio_estimate(base, lower_y)["estimate"]
    upper = ratio_estimate(base, upper_y)["estimate"]
    if lower > upper + 1e-9 or lower > estimate["estimate"] + 1e-9 or estimate["estimate"] > upper + 1e-9:
        raise ValueError("joint technical bounds do not contain paired point estimate")
    return {"estimate": estimate, "lower": lower, "upper": upper,
            "point_curve": point_curve, "lower_curve": lower_curve, "upper_curve": upper_curve,
            "sampling_design": "census" if _census(base) else "stratified_SRS"}


def _census(frame):
    return all(len(g) >= float(g.N_h.iloc[0]) for _, g in frame.groupby("stratum", observed=True))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _match(a, b, label):
    if not np.isclose(float(a), float(b), rtol=1e-10, atol=1e-8):
        raise ValueError(f"baseline point mismatch for {label}: {a} vs {b}")


def _thresholds(series):
    values = pd.to_numeric(series, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("curve thresholds must be finite integers on the fixed grid")
    result = values.astype(int)
    if set(result) != set(THRESHOLDS):
        raise ValueError("curve thresholds do not match the fixed 10--60 minute grid")
    return result


def summarize(times, curves, contrasts, ratio_estimate, coverage_vector):
    baseline = times[times.profile.astype(str).eq(PROFILE)].copy()
    expected_arms = {(c, s, m, PROFILE) for c, scopes in SCOPES.items()
                     for s, m in itertools.product(scopes, MODES)}
    groups = {key: g.copy() for key, g in baseline.groupby(list(ARM_KEY), sort=True)}
    if set(groups) != expected_arms or len(groups) != 30:
        raise ValueError("baseline city/scope/mode arm inventory is not the exact 30-arm set")
    curve = curves[curves.profile.astype(str).eq(PROFILE)].copy()
    curve["threshold"] = _thresholds(curve.threshold)
    expected_curve_keys = {(c, s, m, PROFILE, t) for c, s, m, _ in expected_arms for t in THRESHOLDS}
    if curve.duplicated(list(CURVE_KEY)).any() or set(map(tuple, curve[list(CURVE_KEY)].itertuples(index=False, name=None))) != expected_curve_keys or len(curve) != 330:
        raise ValueError("baseline coverage curves are not the exact 330-row threshold identity")
    out = []
    arm_results = {}
    curve_idx = curve.set_index(list(CURVE_KEY))
    for key, frame in groups.items():
        city, scope, mode, profile = key
        if len(frame) != COUNTS[city] or frame.uid.astype(str).nunique() != COUNTS[city]:
            raise ValueError(f"{city}/{scope}/{mode} does not retain its full main UID set")
        result = arm_area(frame, ratio_estimate, coverage_vector)
        arm_results[(city, scope, mode)] = result
        for i, tau in enumerate(THRESHOLDS):
            src = curve_idx.loc[(city, scope, mode, profile, tau)]
            _match(src.coverage_lower, result["lower_curve"][i], f"{key}/{tau}/lower curve")
            _match(src.coverage_upper, result["upper_curve"][i], f"{key}/{tau}/upper curve")
        source_lower = _area(np.array([curve_idx.loc[(city, scope, mode, profile, t)].coverage_lower for t in THRESHOLDS]))
        source_upper = _area(np.array([curve_idx.loc[(city, scope, mode, profile, t)].coverage_upper for t in THRESHOLDS]))
        _match(result["estimate"]["estimate"], source_lower, f"{key}/area")
        _match(result["upper_estimate"], source_upper, f"{key}/upper area")
        est = result["estimate"]
        out.append(dict(city=city, scope=scope, mode=mode, profile=profile, n=len(frame),
                        sampling_design="census" if _census(result["frame"]) else "stratified_SRS",
                        average_coverage_lower=est["estimate"], average_coverage_upper=result["upper_estimate"],
                        design_se=est["se"], design_ci_lower=est["lower"], design_ci_upper=est["upper"]))

    contrasts = contrasts[contrasts.profile.astype(str).eq(PROFILE)].copy()
    contrasts["threshold"] = _thresholds(contrasts.threshold)
    expected_d = {(city, PROFILE, "R", scope, t) for city, scopes in SCOPES.items()
                  for scope in scopes if scope != "R" for t in THRESHOLDS}
    dkey = ("city", "profile", "scope1", "scope2", "threshold")
    if contrasts.duplicated(list(dkey)).any() or set(map(tuple, contrasts[list(dkey)].itertuples(index=False, name=None))) != expected_d or len(contrasts) != 33:
        raise ValueError("baseline paired contrast curves are not the exact 33-row threshold identity")
    didx = contrasts.set_index(list(dkey))
    paired = []
    for city, scopes in SCOPES.items():
        for scope2 in scopes:
            if scope2 == "R":
                continue
            arms = {(scope, mode): arm_results[(city, scope, mode)]["frame"]
                    for scope, mode in (("R", "wt"), (scope2, "wt"), ("R", "car"), (scope2, "car"))}
            result = paired_area(arms, scope2, ratio_estimate, coverage_vector)
            for i, tau in enumerate(THRESHOLDS):
                src = didx.loc[(city, PROFILE, "R", scope2, tau)]
                _match(src.contrast, result["point_curve"][i], f"{city}/R-{scope2}/D curve/{tau}")
                _match(src.missing_bound_lower, result["lower_curve"][i],
                       f"{city}/R-{scope2}/D lower technical curve/{tau}")
                _match(src.missing_bound_upper, result["upper_curve"][i],
                       f"{city}/R-{scope2}/D upper technical curve/{tau}")
            source_area = _area(np.array([didx.loc[(city, PROFILE, "R", scope2, t)].contrast for t in THRESHOLDS]))
            _match(result["estimate"]["estimate"], source_area, f"{city}/R-{scope2}/D area")
            est = result["estimate"]
            paired.append(dict(city=city, scope1="R", scope2=scope2, profile=PROFILE,
                               sampling_design=result["sampling_design"],
                               average_contrast=est["estimate"], technical_bound_lower=result["lower"],
                               technical_bound_upper=result["upper"], design_se=est["se"],
                               design_ci_lower=est["lower"], design_ci_upper=est["upper"]))
    return pd.DataFrame(out).sort_values(["city", "scope", "mode"]).reset_index(drop=True), \
        pd.DataFrame(paired).sort_values(["city", "scope2"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--times", required=True, type=Path)
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    times_path, analysis_dir, out = args.times.resolve(), args.analysis_dir.resolve(), args.out.resolve()
    if not times_path.is_file() or not analysis_dir.is_dir() or not times_path.is_relative_to(root):
        raise ValueError("--times and --analysis-dir must identify existing inputs under --root")
    if (not analysis_dir.is_relative_to(root) or out.parent != root or out == analysis_dir
            or out.is_relative_to(root / "core") or analysis_dir.is_relative_to(out)
            or times_path.is_relative_to(out)):
        raise ValueError("--out must be a new independent child directory of --root outside core/input paths")
    if out.exists():
        raise FileExistsError("refusing to write into an existing threshold-summary directory")
    curve_path, d_path = analysis_dir / "coverage_curves.csv", analysis_dir / "mode_scope_contrasts.csv"
    core, work = root / "core", root / "work"
    helper_path = work / "v13_analyse.py"
    qa_path = Path(__file__).resolve().with_name("v13_final_scientific_qa.py")
    producer_helper_path = core / "v13_analyse.py"
    frame_paths = [core / "sets" / f"main_{tag}.csv" for tag in ("sp", "bo")]
    inputs = [times_path, curve_path, d_path, helper_path, producer_helper_path,
              qa_path, Path(__file__).resolve(), *frame_paths]
    if not all(p.is_file() for p in inputs):
        raise FileNotFoundError("one or more required times/curve/design/code inputs are absent")
    before = {str(p): _sha(p) for p in inputs}
    sys.path.insert(0, str(core))
    sys.path.insert(0, str(work))
    from v13_analyse import coverage_vector, ratio_estimate
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v13_final_scientific_qa import validate_times
    data = pd.read_csv(times_path, dtype={"uid": str})
    data = data[data.profile.astype(str).eq(PROFILE)].copy()
    frames = {("main", "Sao Paulo"): pd.read_csv(frame_paths[0], dtype={"uid": str}),
              ("main", "Bogota"): pd.read_csv(frame_paths[1], dtype={"uid": str})}
    data = validate_times(data, frozen_frames=frames, expected_counts={PROFILE: COUNTS},
                          require_no_technical=False)
    curves, contrasts = pd.read_csv(curve_path), pd.read_csv(d_path)
    arms, paired = summarize(data, curves, contrasts, ratio_estimate, coverage_vector)
    after = {str(p): _sha(p) for p in inputs}
    if before != after:
        raise RuntimeError("a read-only source changed while summaries were produced")
    outputs = [out / "threshold_interval_arm_summary.csv",
               out / "threshold_interval_paired_contrasts.csv", out / "THRESHOLD_INTERVAL_NOTES.json"]
    out.mkdir(parents=True, exist_ok=True)
    arms.to_csv(outputs[0], index=False)
    paired.to_csv(outputs[1], index=False)
    note = {"status": "exploratory_summary", "timing": EXPLORATORY_NOTE,
            "thresholds_min": list(THRESHOLDS), "area": "trapezoid over 10--60 minutes divided by 50 minutes",
            "arm_estimator": "ratio of design-weighted per-origin threshold-area outcomes; technical lower/upper propagated per origin",
            "paired_estimand": "R-WT - S-WT - R-car + S-car, integrated per origin with signs-correct joint technical bounds",
            "uncertainty": "same v13_analyse ratio_estimate finite-population-corrected stratified-SRS linearization; Bogota census has no sampling CI",
            "curve_policy": "all 330 baseline arm-threshold points and all 33 paired points were identity- and numeric-match checked; area averages do not replace curves",
            "source_sha256_before": before, "source_sha256_after": after}
    outputs[2].write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"DONE: {len(arms)} arm summaries, {len(paired)} paired summaries; source hashes stable", flush=True)


if __name__ == "__main__":
    main()
