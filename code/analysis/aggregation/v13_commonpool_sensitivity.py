#!/usr/bin/env python3
"""Build a strict, expanded physical-route-pool facility sensitivity analysis.

The candidate pool is the union of the frozen source scenarios.  Every scenario
is then re-aggregated against its own facility flags, so scope minima do not
depend on which other scopes happened to pull a destination into a source run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "routing"))
from v13_route_engine import choose_candidates, city_match_key, route_param_hash


MODES = ("car", "wt", "foot")
SCOPES = ("R", "E", "A")
CAPS = ("wt_cap1km", "wt_cap2km", "wt_cap3km")
ARM_MODES = ("car", "wt", "foot", *CAPS)
TECHNICAL = {"timeout", "http_error", "connection_error", "graphql_error", "pending", "config_error"}
TERMINAL = TECHNICAL | {"success", "no_pattern"}
CAR_FACTOR = 1.0 / 0.60
QUERY_IDENTITY_PROTOCOL = "route-query-identity-v1"
COORD_TOL = 1e-10


class CommonPoolError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_hash(value) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _number(value, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise CommonPoolError(f"{label} must be numeric") from exc
    if not math.isfinite(out):
        raise CommonPoolError(f"{label} must be finite")
    return out


def _coordinate(value, label: str, axis: str) -> float:
    if isinstance(value, bool):
        raise CommonPoolError(f"{label} must be numeric coordinates, not a boolean")
    out = _number(value, label)
    low, high = (-90.0, 90.0) if axis == "lat" else (-180.0, 180.0)
    if not low <= out <= high:
        raise CommonPoolError(f"{label} outside [{low},{high}]")
    return out


def _od_hex(origin: dict, destination: dict, label: str):
    if not isinstance(origin, dict) or not isinstance(destination, dict):
        raise CommonPoolError(f"{label}: origin/destination coordinates missing")
    o = (_coordinate(origin.get("lat"), f"{label} origin latitude", "lat"),
         _coordinate(origin.get("lon"), f"{label} origin longitude", "lon"))
    d = (_coordinate(destination.get("lat"), f"{label} destination latitude", "lat"),
         _coordinate(destination.get("lon"), f"{label} destination longitude", "lon"))
    return (o[0].hex(), o[1].hex(), d[0].hex(), d[1].hex()), o, d


def _strict_flag(value, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise CommonPoolError(f"{label} must be a boolean flag")
    text = str(value).strip().casefold()
    if text in {"", "nan", "none", "null", "0", "false", "f", "no", "n"}:
        return False
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    raise CommonPoolError(f"{label} has unrecognized flag value {value!r}")


def _site_id(value, city: str) -> str:
    if value is None:
        raise CommonPoolError("facility site_id is missing")
    site = str(value).strip()
    if not site or site.casefold() in {"nan", "none", "null"}:
        raise CommonPoolError("facility site_id is missing")
    if city_match_key(city) == city_match_key("Sao Paulo"):
        if re.fullmatch(r"\d+\.0+", site):
            site = site.split(".", 1)[0]
        if site.isdigit():
            if len(site) > 7:
                raise CommonPoolError(f"Sao Paulo CNES must be at most 7 digits: {site}")
            return site.zfill(7)
        raise CommonPoolError(f"Sao Paulo CNES must be numeric: {site}")
    if city_match_key(city) == city_match_key("Bogota"):
        if not site.isdigit() or len(site) != 12:
            raise CommonPoolError(f"Bogota facility id must retain its 12-digit representation: {site}")
    return site


def _city_key(value) -> str:
    key = city_match_key(value)
    if not key:
        raise CommonPoolError("city is missing")
    return key


def _config_path(root: Path, city: str) -> Path:
    core = root / "core"
    path = core / "config_complete_baseline.json"
    if not path.is_file():
        raise CommonPoolError(f"complete-network baseline routing config missing for {city}: {path}")
    return path


def _recommended_k(report: dict, city: str, label: str) -> int:
    values = report.get("recommended_k")
    value = values.get(city) if isinstance(values, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CommonPoolError(f"{label}: missing positive recommended_k for {city}")
    return value


def _executed_scope_k(report: dict, label: str, baseline_k: int) -> int:
    entry = report.get(label)
    if not isinstance(entry, dict):
        raise CommonPoolError(f"supplementary candidate report is missing {label}")
    recommended = entry.get("recommended_k")
    executed = entry.get("executed_k")
    if (isinstance(recommended, bool) or not isinstance(recommended, int) or recommended <= 0
            or isinstance(executed, bool) or not isinstance(executed, int) or executed <= 0):
        raise CommonPoolError(f"supplementary candidate report has invalid k values for {label}")
    if executed < max(recommended, baseline_k):
        raise CommonPoolError(
            f"supplementary candidate report for {label} executed k={executed}, "
            f"below recommended/baseline k={max(recommended, baseline_k)}"
        )
    return executed


def _facility_path(root: Path, scenario: str) -> Path:
    core = root / "core"
    if scenario in {"baseline_fixed_departure", "sensitivity_bo_historical_capacity"}:
        name = "facilities_main.csv" if scenario == "baseline_fixed_departure" else "facilities_bo_historical_sensitivity.csv"
        path = core / name
        if not path.is_file():
            raise CommonPoolError(f"required facility input missing: {path}")
        return path
    directory = root / "sp_activity" / "outputs"
    if not directory.is_dir():
        raise CommonPoolError(f"SP sensitivity facility directory missing: {directory}")
    token_sets = {
        "sensitivity_public_rd04": ({"public", "rd04"}, {"rd04", "public"}),
        "sensitivity_municipal_legacy": ({"municipal", "legacy"},),
        "sensitivity_rd04_ge100": ({"rd04", "ge100"},),
        "sensitivity_raw_activity": ({"raw", "activity"},),
    }
    matches = []
    for path in directory.glob("facilities_sp_*.csv"):
        name = path.stem.casefold().replace("-", "_")
        if any(all(token in name for token in tokens) for tokens in token_sets[scenario]):
            matches.append(path)
    if len(matches) != 1:
        raise CommonPoolError(f"expected one facility CSV for {scenario}, found {len(matches)}: {matches}")
    return matches[0]


def _route_path(root: Path, source_name: str) -> Path:
    core = root / "core"
    fixed = {
        "main_sp": core / "main_sp.jsonl",
        "main_bo": core / "main_bo.jsonl",
        "sensitivity_public_rd04": core / "sensitivity_public_rd04.jsonl",
        "sensitivity_municipal_legacy": core / "sensitivity_municipal_legacy.jsonl",
        "sensitivity_rd04_ge100": core / "sensitivity_rd04_ge100.jsonl",
        "sensitivity_raw_activity": core / "sensitivity_raw_activity.jsonl",
        "sensitivity_bo_historical_capacity": root / "secondary" / "bo_historical_capacity.jsonl",
    }
    path = fixed[source_name]
    if not path.is_file():
        raise CommonPoolError(f"route source missing: {path}")
    return path


def _origin_path(root: Path, city: str) -> Path:
    short = "sp" if city_match_key(city) == city_match_key("Sao Paulo") else "bo"
    path = root / "core" / "sets" / f"main_{short}.csv"
    if not path.is_file():
        raise CommonPoolError(f"origin frame missing: {path}")
    return path


@dataclass
class Scenario:
    name: str
    facility_path: Path
    facility_map: dict = field(default_factory=dict)
    scope_available: dict = field(default_factory=dict)
    origins: dict = field(default_factory=dict)
    origin_rows: list = field(default_factory=list)
    source_ids: list = field(default_factory=list)
    facility_ids: dict = field(default_factory=lambda: defaultdict(set))
    counts: dict = field(default_factory=dict)


@dataclass
class Source:
    name: str
    scenario: Scenario
    city: str
    path: Path
    config_path: Path
    config: dict
    route_hashes: dict
    origins_path: Path
    expected_k: int = 0
    expected: set = field(default_factory=set)
    actual: set = field(default_factory=set)
    expected_origin_keys: set = field(default_factory=set)
    actual_origin_keys: set = field(default_factory=set)
    sha256: str | None = None
    sidecar_status: str | None = None


def _read_facilities(path: Path, city_filter=None):
    table = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"city", "site_id", "lat", "lon", *SCOPES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise CommonPoolError(f"{path}: missing facility columns {sorted(missing)}")
        for rownum, row in enumerate(reader, start=2):
            city = str(row.get("city", "")).strip()
            ck = _city_key(city)
            if city_filter and ck != _city_key(city_filter):
                continue
            site = _site_id(row.get("site_id"), city)
            lat, lon = _coordinate(row.get("lat"), f"{path}:{rownum} lat", "lat"), _coordinate(row.get("lon"), f"{path}:{rownum} lon", "lon")
            flags = {scope: _strict_flag(row.get(scope), f"{path}:{rownum} {scope}") for scope in SCOPES}
            key = (ck, site)
            record = {"city": city, "city_key": ck, "site_id": site, "lat": lat, "lon": lon, **flags}
            if key in table:
                prior = table[key]
                if (prior["lat"], prior["lon"], *(prior[s] for s in SCOPES)) != (lat, lon, *(flags[s] for s in SCOPES)):
                    raise CommonPoolError(f"{path}: conflicting duplicate city/site row {city}/{site}")
                raise CommonPoolError(f"{path}: duplicate city/site row {city}/{site}")
            table[key] = record
    if not table:
        raise CommonPoolError(f"{path}: no facilities after city filtering")
    return table


def _read_origins(path: Path, city_filter=None):
    df = pd.read_csv(path, dtype={"uid": str, "city": str})
    required = {"uid", "city", "lat", "lon", "stratum", "pop", "design_weight", "N_h", "n_h", "inclusion_prob"}
    missing = required - set(df.columns)
    if missing:
        raise CommonPoolError(f"{path}: missing origin columns {sorted(missing)}")
    df["stratum"] = df["stratum"].fillna("unknown").astype(str)
    if city_filter:
        df = df[df.city.map(_city_key).eq(_city_key(city_filter))].copy()
    if df.empty or df.uid.isna().any() or df.uid.astype(str).str.strip().eq("").any():
        raise CommonPoolError(f"{path}: empty origin frame or missing uid")
    if df.uid.duplicated().any():
        raise CommonPoolError(f"{path}: duplicate uid")
    for col in ("lat", "lon", "pop", "design_weight", "N_h", "n_h", "inclusion_prob"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise CommonPoolError(f"{path}: non-finite {col}")
    if (df["pop"] <= 0).any() or (df.design_weight <= 0).any() or (df.N_h <= 0).any() or (df.n_h <= 0).any():
        raise CommonPoolError(f"{path}: nonpositive origin design fields")
    for rownum, row in enumerate(df.to_dict("records"), start=2):
        _coordinate(row.get("lat"), f"{path}:{rownum} latitude", "lat")
        _coordinate(row.get("lon"), f"{path}:{rownum} longitude", "lon")
    by_key = {}
    rows = []
    for rec in df.to_dict("records"):
        city = str(rec["city"]).strip()
        ck = _city_key(city)
        if city_filter and ck != _city_key(city_filter):
            raise CommonPoolError(f"{path}: city contradicts source city")
        rec["city"] = city
        rec["uid"] = str(rec["uid"])
        by_key[(ck, rec["uid"])] = rec
        rows.append(rec)
    return by_key, rows


def _plan(root: Path):
    core = root / "core"
    convergence_path = core / "convergence_complete" / "candidate_recommendation.json"
    supplementary_path = root / "secondary" / "supplementary_candidate_recommendations.json"
    if not convergence_path.is_file():
        raise CommonPoolError(f"complete-network candidate recommendation missing: {convergence_path}")
    if not supplementary_path.is_file():
        raise CommonPoolError(f"supplementary candidate recommendation missing: {supplementary_path}")
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    supplementary = json.loads(supplementary_path.read_text(encoding="utf-8"))
    profile_specs = [
        ("baseline_fixed_departure", "main", [("main_sp", "Sao Paulo"), ("main_bo", "Bogota")]),
        ("sensitivity_public_rd04", "sp", [("sensitivity_public_rd04", "Sao Paulo")]),
        ("sensitivity_municipal_legacy", "sp", [("sensitivity_municipal_legacy", "Sao Paulo")]),
        ("sensitivity_rd04_ge100", "sp", [("sensitivity_rd04_ge100", "Sao Paulo")]),
        ("sensitivity_raw_activity", "sp", [("sensitivity_raw_activity", "Sao Paulo")]),
        ("sensitivity_bo_historical_capacity", "bo", [("sensitivity_bo_historical_capacity", "Bogota")]),
    ]
    scenarios = {}
    sources = []
    all_input_paths = {convergence_path, supplementary_path}
    for profile, facility_kind, route_specs in profile_specs:
        if facility_kind == "main":
            facility_path = root / "core" / "facilities_main.csv"
        elif facility_kind == "bo":
            facility_path = root / "core" / "facilities_bo_historical_sensitivity.csv"
        else:
            facility_path = _facility_path(root, profile)
        if not facility_path.is_file():
            raise CommonPoolError(f"facility input missing: {facility_path}")
        scenario = Scenario(profile, facility_path)
        cities = {city for _, city in route_specs}
        # Read all rows once, retaining only the city or cities in this scenario.
        for city in cities:
            city_facilities = _read_facilities(facility_path, city)
            city_origins_path = _origin_path(root, city)
            origin_map, origin_rows = _read_origins(city_origins_path, city)
            config_path = _config_path(root, city)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            baseline_k = _recommended_k(convergence, city, str(convergence_path))
            route_hashes = {mode: route_param_hash(config, mode, city) for mode in MODES}
            scenario.facility_map.update(city_facilities)
            scenario.origins.update(origin_map)
            scenario.origin_rows.extend(origin_rows)
            scopes = {s for s in SCOPES if any(r[s] for r in city_facilities.values())}
            scenario.scope_available[_city_key(city)] = scopes
            for (ck, site), frec in city_facilities.items():
                scenario.facility_ids[(ck, site)] = {s for s in SCOPES if frec[s]}
            all_input_paths.update((facility_path, city_origins_path, config_path))
            source_name = next(n for n, c in route_specs if c == city)
            if source_name == "sensitivity_bo_historical_capacity":
                expected_k = _executed_scope_k(supplementary, "bo_historical_capacity", baseline_k)
            elif source_name.startswith("sensitivity_"):
                label = source_name.removeprefix("sensitivity_")
                expected_k = _executed_scope_k(supplementary, label, baseline_k)
            else:
                expected_k = baseline_k
            route_path = _route_path(root, source_name)
            src = Source(source_name, scenario, city, route_path, config_path, config, route_hashes, city_origins_path,
                         expected_k=expected_k)
            raw_facilities = [
                {"city": r["city"], "site_id": r["site_id"], "lat": str(r["lat"]), "lon": str(r["lon"]),
                 **{s: "1" if r[s] else "0" for s in SCOPES}}
                for r in city_facilities.values()
            ]
            for origin in origin_rows:
                for candidate in choose_candidates(origin, raw_facilities, expected_k):
                    site = _site_id(candidate["site_id"], city)
                    if ( _city_key(city), site) not in city_facilities:
                        raise CommonPoolError(f"candidate site missing from frozen facility table: {city}/{site}")
                    for mode in MODES:
                        rh = route_hashes[mode]
                        src.expected.add((_city_key(city), str(origin["uid"]), site, mode, rh))
                src.expected_origin_keys.add((_city_key(city), str(origin["uid"])))
            sources.append(src)
            scenario.source_ids.append(source_name)
            all_input_paths.update((route_path,))
        scenarios[profile] = scenario
        all_input_paths.add(facility_path)

    shared_hashes = {}
    for src in sources:
        for mode, rh in src.route_hashes.items():
            key = (_city_key(src.city), mode)
            if key in shared_hashes and shared_hashes[key] != rh:
                raise CommonPoolError(f"route request configuration differs across pool sources for {key}; mixed time/parameter pools are forbidden")
            shared_hashes[key] = rh

    # A site ID denotes one physical destination throughout the pool.
    physical_coords = {}
    for scenario in scenarios.values():
        for (ck, site), rec in scenario.facility_map.items():
            key = (ck, site)
            coords = (rec["lat"], rec["lon"])
            if key in physical_coords and any(abs(a - b) > 1e-10 for a, b in zip(physical_coords[key], coords)):
                raise CommonPoolError(f"facility coordinates vary across scenarios for {key}: {physical_coords[key]} vs {coords}")
            physical_coords.setdefault(key, coords)

    # The same fixed origin must retain identical design fields across its source frames.
    origin_truth = {}
    for scenario in scenarios.values():
        for key, rec in scenario.origins.items():
            fields = (rec["city"], rec["lat"], rec["lon"], rec["stratum"], rec["pop"], rec["design_weight"], rec["N_h"], rec["n_h"], rec["inclusion_prob"])
            if key in origin_truth:
                prior = origin_truth[key]
                if prior[0] != fields[0] or any(abs(float(a) - float(b)) > 1e-10 for a, b in zip(prior[1:3], fields[1:3])) or prior[3:] != fields[3:]:
                    raise CommonPoolError(f"origin metadata vary across scenarios for {key}")
            origin_truth[key] = fields
    for code_path in (root / "core" / "v13_route_engine.py", root / "core" / "v13_analyse.py", Path(__file__).resolve()):
        if code_path.is_file():
            all_input_paths.add(code_path)
    return scenarios, sources, physical_coords, origin_truth, sorted(all_input_paths)


def _check_route_record(rec, src: Source, physical_coords, global_signatures):
    if not isinstance(rec, dict) or rec.get("record_type") != "route":
        raise CommonPoolError(f"{src.path}: malformed route record")
    city = str(rec.get("city", "")).strip()
    ck = _city_key(city)
    uid = str(rec.get("uid", ""))
    mode = rec.get("mode")
    if ck != _city_key(src.city):
        raise CommonPoolError(f"{src.path}: route city contradicts its source")
    if mode not in MODES:
        raise CommonPoolError(f"{src.path}: unsupported route mode {mode!r}")
    site = _site_id(rec.get("site_id"), city)
    rh = rec.get("request_hash")
    logical_key = (ck, uid, site, mode, rh)
    expected_hash = src.route_hashes[mode]
    if rh != expected_hash:
        raise CommonPoolError(f"{src.path}: route request hash contradicts official config for {uid}/{site}/{mode}")
    if logical_key not in src.expected:
        raise CommonPoolError(f"{src.path}: unexpected candidate or hash for {uid}/{site}/{mode}")
    od_hex, origin, dest = _od_hex(rec.get("origin"), rec.get("destination"), f"{src.path} {uid}/{site}/{mode}")
    truth = src.scenario.origins.get((ck, uid))
    facility = src.scenario.facility_map.get((ck, site))
    if truth is None or facility is None:
        raise CommonPoolError(f"{src.path}: route UID or facility absent from frozen source inputs")
    for field_name in ("lat", "lon"):
        if abs(origin[0 if field_name == "lat" else 1] - float(truth[field_name])) > COORD_TOL:
            raise CommonPoolError(f"{src.path}: route origin coordinate mismatch for {uid}")
        if abs(dest[0 if field_name == "lat" else 1] - float(facility[field_name])) > COORD_TOL:
            raise CommonPoolError(f"{src.path}: route destination coordinate mismatch for {uid}/{site}")
    flags = rec.get("facility_flags")
    if not isinstance(flags, dict) or set(flags) != set(SCOPES):
        raise CommonPoolError(f"{src.path}: facility_flags missing, malformed, or contain unexpected scopes")
    normalized_flags = {}
    for scope in SCOPES:
        actual_flag = _strict_flag(flags.get(scope), f"route {uid}/{site} {scope}")
        if actual_flag != bool(facility[scope]):
            raise CommonPoolError(f"{src.path}: route flags contradict source mask for {uid}/{site}/{scope}")
        normalized_flags[scope] = actual_flag
    rec["facility_flags"] = normalized_flags
    status = rec.get("status")
    patterns = rec.get("patterns")
    if status not in TERMINAL or not isinstance(patterns, list):
        raise CommonPoolError(f"{src.path}: invalid status or patterns for {uid}/{site}/{mode}")
    if status == "success" and not patterns:
        raise CommonPoolError(f"{src.path}: success route has no patterns for {uid}/{site}/{mode}")
    if status != "success" and patterns:
        raise CommonPoolError(f"{src.path}: non-success route unexpectedly carries patterns for {uid}/{site}/{mode}")
    for pattern in patterns:
        if not isinstance(pattern, dict) or not isinstance(pattern.get("legs"), list):
            raise CommonPoolError(f"{src.path}: malformed pattern for {uid}/{site}/{mode}")
        t = _number(pattern.get("duration"), "pattern duration")
        if t < 0:
            raise CommonPoolError("negative route duration")
        journey = _number(pattern.get("journey_duration_min", t), "pattern journey duration")
        if journey < 0:
            raise CommonPoolError("negative route journey duration")
        walk = _number(pattern.get("foot_distance", 0), "pattern foot distance")
        initial_wait = _number(pattern.get("initial_departure_gap_min", 0), "pattern initial departure gap")
        if walk < 0 or initial_wait < 0:
            raise CommonPoolError("negative route walk distance or initial departure gap")
        for leg in pattern["legs"]:
            if not isinstance(leg, dict) or not isinstance(leg.get("mode"), str):
                raise CommonPoolError("malformed route leg")
            if _number(leg.get("distance"), "leg distance") < 0 or _number(leg.get("duration", 0), "leg duration") < 0:
                raise CommonPoolError("negative route leg distance or duration")
    content_hash = _json_hash({"status": status, "patterns": patterns})
    physical_key = (ck, *od_hex, mode, rh)
    prior = global_signatures.get(physical_key)
    if prior is not None and prior != content_hash:
        raise CommonPoolError(f"same physical route key has conflicting status/patterns: {uid}/{site}/{mode}")
    global_signatures[physical_key] = content_hash
    src.actual.add(logical_key)
    application_key = (logical_key, od_hex)
    return logical_key, physical_key, application_key, rec, status, patterns


def _new_acc():
    return {"best": math.inf, "best_site": None, "best_walk_m": np.nan,
            "best_modes": None, "best_journey": math.inf,
            "winning_journey": np.nan, "winning_initial_wait": np.nan,
            "technical": 0, "no_pattern": 0, "success": 0, "queried": 0,
            "_winner_tie": None}


def _route_targets(route_mode):
    if route_mode == "car":
        return ("car",)
    if route_mode == "wt":
        return ("wt", *CAPS)
    # Every independent pure-foot query is also a WT and walking-cap candidate.
    return ("foot", "wt", *CAPS)


def _add_route(acc, route_mode, status, patterns, site, cap_m=None, car_factor=CAR_FACTOR):
    acc["queried"] += 1
    if status in TECHNICAL:
        acc["technical"] += 1
        return
    if status == "no_pattern":
        acc["no_pattern"] += 1
        return
    if status != "success":
        raise CommonPoolError(f"unsupported route status {status}")
    acc["success"] += 1
    for pattern in patterns:
        elapsed = _number(pattern.get("duration"), "pattern duration")
        walk = _number(pattern.get("foot_distance", sum(
            _number(leg.get("distance", 0), "leg distance")
            for leg in pattern.get("legs", [])
            if str(leg.get("mode", "")).casefold() in {"walk", "foot"}
        )), "pattern foot distance")
        journey = elapsed if route_mode == "foot" else _number(pattern.get("journey_duration_min", elapsed), "pattern journey duration")
        if elapsed < 0 or walk < 0 or journey < 0:
            raise CommonPoolError("negative route duration or distance")
        if cap_m is not None and walk > cap_m + 1e-6:
            continue
        mode_factor = car_factor if route_mode == "car" else 1.0
        elapsed *= mode_factor
        journey *= mode_factor
        acc["best_journey"] = min(acc["best_journey"], journey)
        # A foot-only profile starts immediately by definition.
        initial_wait = 0.0 if route_mode == "foot" else _number(pattern.get("initial_departure_gap_min", 0), "initial wait")
        best_modes = "+".join(str(leg.get("mode", "")) for leg in pattern["legs"])
        tie_key = (str(site), json.dumps(pattern, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
                   str(route_mode), float(journey), float(initial_wait), best_modes, float(walk))
        if elapsed < acc["best"] or (elapsed == acc["best"] and (acc["_winner_tie"] is None or tie_key < acc["_winner_tie"])):
            acc.update(best=elapsed, best_site=site, best_walk_m=walk,
                       winning_journey=journey, winning_initial_wait=initial_wait,
                       best_modes=best_modes, _winner_tie=tie_key)


def _times_from_pool(scenarios, accum):
    rows = []
    for profile, scenario in scenarios.items():
        for origin in scenario.origin_rows:
            ck, uid = _city_key(origin["city"]), str(origin["uid"])
            scopes = sorted(scenario.scope_available.get(ck, set()))
            for scope in scopes:
                for mode in ARM_MODES:
                    z = accum.get((profile, ck, uid, scope, mode))
                    if z is None:
                        z = _new_acc()
                    t = z["best"]
                    unknown = z["technical"] > 0 or z["queried"] == 0
                    if z["queried"] == 0:
                        status = "not_queried"
                    elif unknown:
                        status = "partial_error" if math.isfinite(t) else "technical_error"
                    elif math.isfinite(t):
                        status = "ok"
                    else:
                        status = "no_returned_feasible_candidate"
                    meta = {k: origin[k] for k in ("city", "uid", "lat", "lon", "stratum", "pop", "design_weight", "N_h", "n_h", "inclusion_prob")}
                    rows.append(dict(**meta, scope=scope, mode=mode, profile=profile,
                        time_min=t if math.isfinite(t) else np.nan, status=status, finite_unknown=unknown,
                        best_journey_min=z["best_journey"] if math.isfinite(z["best_journey"]) else np.nan,
                        winning_journey_min=z["winning_journey"],
                        winning_initial_wait_min=z["winning_initial_wait"],
                        best_site=z["best_site"], best_modes=z["best_modes"], best_walk_m=z["best_walk_m"],
                        queried_destinations=z["queried"], successful_destinations=z["success"],
                        no_pattern_destinations=z["no_pattern"], technical_destinations=z["technical"]))
    out = pd.DataFrame(rows)
    if out.empty:
        raise CommonPoolError("no origin-arm rows materialized")
    return out


def _add_check(checks, name, n_checked, violations, detail=""):
    checks.append({"invariant": name, "n_checked": int(n_checked), "n_violations": int(violations),
                   "passed": int(violations) == 0, "detail": detail})


def _same_time_check(checks, df, name, profiles, scope, modes, city="Sao Paulo"):
    sub = df[(df.city.map(_city_key) == _city_key(city)) & (df.scope == scope) & df.profile.isin(profiles) & df["mode"].isin(modes)]
    violations = 0
    checked = 0
    for mode, g in sub.groupby("mode"):
        pivot = g.pivot(index="uid", columns="profile", values="time_min").reindex(columns=profiles)
        if len(pivot) == 0:
            violations += 1
            continue
        checked += len(pivot)
        first = pivot.iloc[:, 0]
        for col in pivot.columns[1:]:
            a, b = first.to_numpy(float), pivot[col].to_numpy(float)
            equal = (np.isnan(a) & np.isnan(b)) | (a == b)
            violations += int((~equal).sum())
    _add_check(checks, name, checked, violations, f"scope={scope}; modes={','.join(modes)}")


def _validate_invariants(df, scenarios, accum, require_domain=True):
    checks = []
    sp_profiles = ["baseline_fixed_departure", "sensitivity_public_rd04",
                   "sensitivity_rd04_ge100", "sensitivity_raw_activity"]
    _same_time_check(checks, df, "SP common-pool R339 equality", sp_profiles, "R", ARM_MODES)
    _same_time_check(checks, df, "SP legacy E89 baseline/ge100 equality",
                     ["baseline_fixed_departure", "sensitivity_rd04_ge100"], "E", ARM_MODES)
    _same_time_check(checks, df, "BO public E16 baseline/historical equality",
                     ["baseline_fixed_departure", "sensitivity_bo_historical_capacity"], "E", ARM_MODES, city="Bogota")

    # A looser walk cap must never have a slower returned-route minimum.
    arm_index = ["profile", "city", "scope", "uid"]
    arm_times = df.pivot(index=arm_index, columns="mode", values="time_min")
    cap_violations = cap_checked = 0
    for tight, loose in (("wt_cap1km", "wt_cap2km"), ("wt_cap2km", "wt_cap3km"), ("wt_cap3km", "wt")):
        a, b = arm_times[tight].to_numpy(float), arm_times[loose].to_numpy(float)
        has_tight = np.isfinite(a)
        cap_checked += int(has_tight.sum())
        cap_violations += int((has_tight & (~np.isfinite(b) | (a + 1e-10 < b))).sum())
    _add_check(checks, "common-pool walk-cap nesting", cap_checked, cap_violations)

    # WT is the union of transit-profile and standalone-foot queries.
    wt, foot = arm_times["wt"].to_numpy(float), arm_times["foot"].to_numpy(float)
    has_foot = np.isfinite(foot)
    wt_checked = int(has_foot.sum())
    wt_violations = int((has_foot & (~np.isfinite(wt) | (wt > foot + 1e-10))).sum())
    _add_check(checks, "WT includes standalone-foot candidate set", wt_checked, wt_violations)

    if require_domain:
        def facility_set(profile, scope, city):
            return {site for (ck, site), flags in scenarios[profile].facility_ids.items()
                    if ck == _city_key(city) and scope in flags}

        # Pin the facility masks as well as their expected route-time implications.
        main_r = facility_set("baseline_fixed_departure", "R", "Sao Paulo")
        _add_check(checks, "SP baseline R339 facility count", 1, abs(len(main_r) - 339), f"observed={len(main_r)} expected=339")
        for profile in ("sensitivity_public_rd04", "sensitivity_rd04_ge100", "sensitivity_raw_activity"):
            current = facility_set(profile, "R", "Sao Paulo")
            bad = len(current ^ main_r)
            _add_check(checks, f"{profile} SP R339 mask equality", len(main_r), bad)
            _add_check(checks, f"{profile} SP R339 facility count", 1, abs(len(current) - 339), f"observed={len(current)} expected=339")
        legacy_r = facility_set("sensitivity_municipal_legacy", "R", "Sao Paulo")
        _add_check(checks, "municipal SP R185 facility count", 1, abs(len(legacy_r) - 185), f"observed={len(legacy_r)} expected=185")
        _add_check(checks, "municipal R185 mask contained in R339", len(legacy_r), len(legacy_r - main_r))

        main_e = facility_set("baseline_fixed_departure", "E", "Sao Paulo")
        ge100_e = facility_set("sensitivity_rd04_ge100", "E", "Sao Paulo")
        _add_check(checks, "SP baseline E89 facility count", 1, abs(len(main_e) - 89), f"observed={len(main_e)} expected=89")
        _add_check(checks, "SP ge100 E89 facility count", 1, abs(len(ge100_e) - 89), f"observed={len(ge100_e)} expected=89")
        _add_check(checks, "SP legacy E89 mask equality", len(main_e), len(main_e ^ ge100_e))

        bo_main_r = facility_set("baseline_fixed_departure", "R", "Bogota")
        bo_hist_r = facility_set("sensitivity_bo_historical_capacity", "R", "Bogota")
        bo_main_e = facility_set("baseline_fixed_departure", "E", "Bogota")
        bo_hist_e = facility_set("sensitivity_bo_historical_capacity", "E", "Bogota")
        _add_check(checks, "BO main R142 facility count", 1, abs(len(bo_main_r) - 142), f"observed={len(bo_main_r)} expected=142")
        _add_check(checks, "BO historical R158 facility count", 1, abs(len(bo_hist_r) - 158), f"observed={len(bo_hist_r)} expected=158")
        _add_check(checks, "BO historical R158 contains main R142", len(bo_main_r), len(bo_main_r - bo_hist_r))
        _add_check(checks, "BO main E16 facility count", 1, abs(len(bo_main_e) - 16), f"observed={len(bo_main_e)} expected=16")
        _add_check(checks, "BO historical E16 facility count", 1, abs(len(bo_hist_e) - 16), f"observed={len(bo_hist_e)} expected=16")
        _add_check(checks, "BO public E16 mask equality", len(bo_main_e), len(bo_main_e ^ bo_hist_e))

        # The requested label and dataset counts are also pinned as source-integrity checks.
        for profile, expected in (("sensitivity_public_rd04", 135), ("sensitivity_rd04_ge100", 133), ("sensitivity_raw_activity", 156)):
            scenario = scenarios[profile]
            ids = {site for (ck, site), flags in scenario.facility_ids.items() if ck == _city_key("Sao Paulo") and "A" in flags}
            _add_check(checks, f"{profile} SP A facility count", 1, abs(len(ids) - expected), f"observed={len(ids)} expected={expected}")

        public = scenarios["sensitivity_public_rd04"].facility_ids
        ge100 = scenarios["sensitivity_rd04_ge100"].facility_ids
        public_a = {site for (ck, site), flags in public.items() if ck == _city_key("Sao Paulo") and "A" in flags}
        ge100_a = {site for (ck, site), flags in ge100.items() if ck == _city_key("Sao Paulo") and "A" in flags}
        subset_bad = len(ge100_a - public_a)
        _add_check(checks, "RD04 ge1 facility set contains ge100 set", len(ge100_a), subset_bad)
        sup_violations = 0
        lhs = df[(df.profile == "sensitivity_public_rd04") & (df.scope == "A") & df.city.map(_city_key).eq(_city_key("Sao Paulo"))]
        rhs = df[(df.profile == "sensitivity_rd04_ge100") & (df.scope == "A") & df.city.map(_city_key).eq(_city_key("Sao Paulo"))]
        joined = lhs.merge(rhs, on=["uid", "mode"], suffixes=("_ge1", "_ge100"), validate="one_to_one")
        for a, b in zip(joined.time_min_ge1, joined.time_min_ge100):
            if np.isfinite(b) and (not np.isfinite(a) or float(a) > float(b) + 1e-10):
                sup_violations += 1
        _add_check(checks, "RD04 ge1 A minimum no slower than ge100", len(joined), sup_violations)

        # In raw activity, the 156-site A mask contains its R∩A subset (expected 135).
        raw = scenarios["sensitivity_raw_activity"]
        raw_intersection = {site for (ck, site), flags in raw.facility_ids.items() if ck == _city_key("Sao Paulo") and {"R", "A"}.issubset(flags)}
        _add_check(checks, "raw activity R∩A facility count", 1, abs(len(raw_intersection) - 135), f"observed={len(raw_intersection)} expected=135")
        raw_violations = 0
        for origin in raw.origin_rows:
            uid = str(origin["uid"])
            for mode in ARM_MODES:
                a = accum.get((raw.name, _city_key("Sao Paulo"), uid, "A", mode), _new_acc())["best"]
                b = _new_acc()["best"]
                # The derived R∩A minimum is accumulated in the reserved pseudo-scope.
                b = accum.get((raw.name, _city_key("Sao Paulo"), uid, "R_intersection_A", mode), _new_acc())["best"]
                if math.isfinite(b) and (not math.isfinite(a) or a > b + 1e-10):
                    raw_violations += 1
        _add_check(checks, "raw A156 minimum no slower than R∩A135", len(raw.origin_rows) * len(ARM_MODES), raw_violations)

        municipal = df[(df.profile == "sensitivity_municipal_legacy") & (df.city.map(_city_key) == _city_key("Sao Paulo"))]
        reference = df[(df.profile == "baseline_fixed_departure") & (df.city.map(_city_key) == _city_key("Sao Paulo"))]
        r_m = municipal[municipal.scope == "R"]
        r_b = reference[reference.scope == "R"]
        j = r_m.merge(r_b, on=["uid", "mode"], suffixes=("_mun", "_base"), validate="one_to_one")
        municipal_violations = sum(1 for a, b in zip(j.time_min_mun, j.time_min_base)
                                   if np.isfinite(a) and np.isfinite(b) and float(a) + 1e-10 < float(b))
        _add_check(checks, "municipal R185 minimum no faster than R339", len(j), municipal_violations)

        hist = df[(df.profile == "sensitivity_bo_historical_capacity") & (df.city.map(_city_key) == _city_key("Bogota")) & (df.scope == "R")]
        main = df[(df.profile == "baseline_fixed_departure") & (df.city.map(_city_key) == _city_key("Bogota")) & (df.scope == "R")]
        j = hist.merge(main, on=["uid", "mode"], suffixes=("_hist", "_main"), validate="one_to_one")
        hist_violations = sum(1 for a, b in zip(j.time_min_hist, j.time_min_main)
                              if np.isfinite(b) and (not np.isfinite(a) or float(a) > float(b) + 1e-10))
        _add_check(checks, "BO historical R158 minimum no faster than R142", len(j), hist_violations)

    checks_df = pd.DataFrame(checks)
    if not checks_df.passed.all():
        failed = checks_df[~checks_df.passed][["invariant", "n_violations", "detail"]].to_dict("records")
        raise CommonPoolError("invariant validation failed: " + json.dumps(failed, ensure_ascii=False))
    return checks_df


def run(root: Path, out: Path, require_domain=True, run_analysis=True):
    root, out = Path(root).resolve(), Path(out).resolve()
    scenarios, sources, physical_coords, origin_truth, all_input_paths = _plan(root)
    accum = {}
    signatures = {}
    pool_seen = set()
    logical_pool = set()
    input_hashes = {str(p): _sha256_file(p) for p in all_input_paths if p.suffix != ".jsonl"}
    for src in sources:
        sha = hashlib.sha256()
        before = src.path.stat()
        with src.path.open("rb") as handle:
            for raw in handle:
                sha.update(raw)
                try:
                    rec = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise CommonPoolError(f"{src.path}: invalid JSONL record") from exc
                if not isinstance(rec, dict):
                    raise CommonPoolError(f"{src.path}: JSONL record is not an object")
                kind = rec.get("record_type")
                if kind == "origin":
                    ck, uid = _city_key(rec.get("city")), str(rec.get("uid", ""))
                    key = (ck, uid)
                    if key not in src.expected_origin_keys:
                        raise CommonPoolError(f"{src.path}: unexpected origin record {key}")
                    if key in src.actual_origin_keys:
                        raise CommonPoolError(f"{src.path}: duplicate origin record {key}")
                    truth = src.scenario.origins[key]
                    if abs(_number(rec.get("lat"), "origin latitude") - float(truth["lat"])) > 1e-10 or abs(_number(rec.get("lon"), "origin longitude") - float(truth["lon"])) > 1e-10:
                        raise CommonPoolError(f"{src.path}: origin record coordinate mismatch for {key}")
                    src.actual_origin_keys.add(key)
                    continue
                if kind != "route":
                    raise CommonPoolError(f"{src.path}: unknown record_type {kind!r}")
                key, physical_key, application_key, route, status, patterns = _check_route_record(rec, src, physical_coords, signatures)
                ck, uid, site, route_mode, _ = key
                # One physical route record is immediately accumulated against every
                # scenario's own masks. Duplicate source materializations are validated,
                # but counted/aggregated only once.
                first_for_key = application_key not in pool_seen
                pool_seen.add(application_key)
                logical_pool.add(key)
                if not first_for_key:
                    continue
                for profile, scenario in scenarios.items():
                    facility = scenario.facility_map.get((ck, site))
                    if facility is None:
                        continue
                    memberships = {s for s in SCOPES if facility[s]}
                    if profile == "sensitivity_raw_activity" and {"R", "A"}.issubset(memberships):
                        memberships.add("R_intersection_A")
                    for scope in memberships:
                        for arm in _route_targets(route_mode):
                            akey = (profile, ck, uid, scope, arm)
                            z = accum.setdefault(akey, _new_acc())
                            cap = {"wt_cap1km": 1000.0, "wt_cap2km": 2000.0, "wt_cap3km": 3000.0}.get(arm)
                            _add_route(z, route_mode, status, patterns, site, cap_m=cap)
        after = src.path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise CommonPoolError(f"{src.path}: changed while being read; rerun after source materialization is stable")
        src.sha256 = sha.hexdigest()
        input_hashes[str(src.path)] = src.sha256
        sidecar = Path(str(src.path) + ".status.json")
        if sidecar.is_file():
            sidecar_bytes = sidecar.read_bytes()
            input_hashes[str(sidecar)] = hashlib.sha256(sidecar_bytes).hexdigest()
            status_doc = json.loads(sidecar_bytes)
            src.sidecar_status = status_doc.get("status")
        missing = src.expected - src.actual
        if missing:
            raise CommonPoolError(f"{src.path}: incomplete source, expected={len(src.expected)} actual={len(src.actual)} missing={len(missing)}")
        if src.actual - src.expected:
            raise CommonPoolError(f"{src.path}: unexpected source candidates")
        if src.actual_origin_keys != src.expected_origin_keys:
            raise CommonPoolError(f"{src.path}: incomplete origin records")

    # The pseudo-scope used for the raw R∩A acceptance check is not a published arm.
    df = _times_from_pool(scenarios, accum)
    checks = _validate_invariants(df, scenarios, accum, require_domain=require_domain)
    out.mkdir(parents=True, exist_ok=True)
    times_path = out / "times_commonpool.csv"
    df.to_csv(times_path, index=False)
    checks.to_csv(out / "invariants.csv", index=False)
    manifest = {
        "pool_definition": "complete-network union of main_sp/main_bo and four SP plus BO historical sensitivity fixed-scenario physical route queries; candidate audit and patterns50 excluded",
        "query_identity": {
            "protocol_version": QUERY_IDENTITY_PROTOCOL,
            "logical_completeness_key": ["city", "uid", "site_id", "mode", "request_hash"],
            "physical_payload_key": ["city", "origin_lat_hex", "origin_lon_hex", "destination_lat_hex", "destination_lon_hex", "mode", "request_hash"],
            "application_dedup_key": ["logical_completeness_key", "OD_float_hex"],
            "logical_source_records_expected": sum(len(s.expected) for s in sources),
            "logical_source_records_actual": sum(len(s.actual) for s in sources),
            "unique_logical_keys_across_sources": len(logical_pool),
            "unique_physical_payload_keys": len(signatures),
            "unique_application_keys": len(pool_seen),
            "logical_keys_with_multiple_physical_variants": sum(n > 1 for n in Counter(k for k, _ in pool_seen).values()),
            "physical_variants_beyond_first": sum(max(0, n - 1) for n in Counter(k for k, _ in pool_seen).values()),
            "coordinate_tolerance_degrees": COORD_TOL,
        },
        "deduplication_key": ["logical_completeness_key", "OD_float_hex"],
        "duplicate_content_check": "exact physical OD/mode/request hash must have identical status and full patterns; facility flags are validated against each source before being excluded from the payload signature",
        "query_count_definition": "logical source completeness is checked per source; unique physical payloads and per-logical applications/coordinate variants are separately counted",
        "weighted_modes": "WT is the union of transit-query and standalone-foot-query returned candidates; standalone foot has immediate departure",
        "technical_failures": "retained as finite_unknown bounds for each source mode; a WT or walking-cap arm is unknown if either contributing query mode has a technical failure",
        "walk_caps": "same union pool, filtered by returned foot distance; bounds are conditional on fixed queried candidates",
        "profiles": {name: {"facility_csv": str(s.facility_path), "facility_sha256": _sha256_file(s.facility_path),
                             "scope_counts": {scope: sum(scope in flags for (ck, _), flags in s.facility_ids.items()) for scope in SCOPES}}
                     for name, s in scenarios.items()},
        "sources": [{"name": s.name, "path": str(s.path), "sha256": s.sha256,
                     "expected_k": s.expected_k,
                     "expected_unique_queries": len(s.expected), "actual_unique_queries": len(s.actual),
                     "expected_origins": len(s.expected_origin_keys), "actual_origins": len(s.actual_origin_keys),
                     "sidecar_status": s.sidecar_status} for s in sources],
        "input_sha256": input_hashes,
        "unique_physical_pool_queries": len(signatures),
        "unique_application_queries": len(pool_seen),
        "n_origins": sum(len(s.origin_rows) for s in scenarios.values()),
        "n_output_rows": len(df), "n_invariants": len(checks),
        "car_factor": CAR_FACTOR,
    }
    (out / "pool_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if run_analysis:
        analyse = root / "core" / "v13_analyse.py"
        if not analyse.is_file():
            raise CommonPoolError(f"analysis script missing: {analyse}")
        subprocess.run([sys.executable, str(analyse), "--times", str(times_path), "--out", str(out / "analysis")], check=True)
    print(f"DONE commonpool rows={len(df)} sources={len(sources)} invariants={len(checks)}", flush=True)
    return df, checks, manifest


def _fixture_self_test():
    """Small end-to-end route-pool fixture, with no network access or source writes."""
    with tempfile.TemporaryDirectory(prefix="v13_commonpool_selftest_") as tmp:
        root = Path(tmp)
        core = root / "core"
        (core / "sets").mkdir(parents=True)
        outputs = root / "sp_activity" / "outputs"
        outputs.mkdir(parents=True)
        (root / "secondary").mkdir()
        for filename in ("v13_route_engine.py", "v13_analyse.py"):
            source_code = Path(__file__).with_name(filename)
            if not source_code.is_file():
                spec = importlib.util.find_spec(Path(filename).stem)
                if spec is not None and spec.origin:
                    source_code = Path(spec.origin)
            if source_code.is_file():
                shutil.copy2(source_code, core / filename)
        origin_sp = [{"city": "Sao Paulo", "uid": "sp:u1", "lat": 0.0, "lon": 0.0, "stratum": "Q1", "pop": 1.0, "design_weight": 1.0, "N_h": 1, "n_h": 1, "inclusion_prob": 1.0}]
        origin_bo = [{"city": "Bogota", "uid": "bo:u1", "lat": 3.0, "lon": 3.0, "stratum": "low", "pop": 1.0, "design_weight": 1.0, "N_h": 1, "n_h": 1, "inclusion_prob": 1.0}]
        cols = ["city", "uid", "lat", "lon", "stratum", "pop", "design_weight", "N_h", "n_h", "inclusion_prob"]
        for name, rows in (("main_sp", origin_sp), ("main_bo", origin_bo)):
            pd.DataFrame(rows, columns=cols).to_csv(core / "sets" / f"{name}.csv", index=False)
        config = {
            "k": 1,
            "endpoint_by_city": {"Sao Paulo": "http://example.test/sp", "Bogota": "http://example.test/bo"},
            "date_time_by_city": {"Sao Paulo": "2026-11-11T07:30:00-03:00", "Bogota": "2026-11-11T07:30:00-05:00"},
            "profile": "baseline_fixed_departure",
            "router_profile": {"SP_graph_sha256": "fixture-sp", "BO_graph_sha256": "fixture-bo"},
            "modes": {
                "car": {"literal": "{directMode:car}", "args": "numTripPatterns:1"},
                "wt": {"literal": "{directMode:foot}", "args": "numTripPatterns:1"},
                "foot": {"literal": "{directMode:foot}", "args": "numTripPatterns:1"},
            },
        }
        (core / "config_complete_baseline.json").write_text(json.dumps(config), encoding="utf-8")
        (core / "convergence_complete").mkdir()
        (core / "convergence_complete" / "candidate_recommendation.json").write_text(
            json.dumps({"recommended_k": {"Sao Paulo": 1, "Bogota": 1}}), encoding="utf-8"
        )
        supplementary = {name: {"recommended_k": 1, "executed_k": 1} for name in
                         ("public_rd04", "municipal_legacy", "rd04_ge100", "raw_activity")}
        supplementary["bo_historical_capacity"] = {"recommended_k": 2, "executed_k": 2}
        (root / "secondary" / "supplementary_candidate_recommendations.json").write_text(
            json.dumps(supplementary), encoding="utf-8")
        sp_facilities = [
            {"city": "Sao Paulo", "site_id": "123", "lat": 0.01, "lon": 0.0, "R": 1, "E": 1, "A": 1},
            {"city": "Sao Paulo", "site_id": "456", "lat": 0.02, "lon": 0.0, "R": 0, "E": 0, "A": 1},
        ]
        bo_facilities = [{"city": "Bogota", "site_id": "110010413101", "lat": 3.01, "lon": 3.0, "R": 1, "E": 1, "A": 1}]
        fcols = ["city", "site_id", "lat", "lon", *SCOPES]
        pd.DataFrame(sp_facilities + bo_facilities, columns=fcols).to_csv(core / "facilities_main.csv", index=False)
        pd.DataFrame(bo_facilities, columns=fcols).to_csv(core / "facilities_bo_historical_sensitivity.csv", index=False)
        scenario_facilities = {
            "public_rd04": [{**r, "A": 0 if r["site_id"] == "123" else 1} for r in sp_facilities],
            "municipal_legacy": sp_facilities,
            "rd04_ge100": [{**r, "A": 0 if r["site_id"] == "123" else 1} for r in sp_facilities],
            "raw_activity": [{**r, "A": 1} for r in sp_facilities],
        }
        for suffix, rows in scenario_facilities.items():
            pd.DataFrame(rows, columns=fcols).to_csv(outputs / f"facilities_sp_{suffix}.csv", index=False)
        # Resolve normalized fixture facility rows and write complete source JSONLs.
        source_specs = [
            ("main_sp", "baseline_fixed_departure", "Sao Paulo", core / "facilities_main.csv", core / "sets" / "main_sp.csv", core / "config_complete_baseline.json", core / "main_sp.jsonl"),
            ("main_bo", "baseline_fixed_departure", "Bogota", core / "facilities_main.csv", core / "sets" / "main_bo.csv", core / "config_complete_baseline.json", core / "main_bo.jsonl"),
            ("sensitivity_public_rd04", "sensitivity_public_rd04", "Sao Paulo", outputs / "facilities_sp_public_rd04.csv", core / "sets" / "main_sp.csv", core / "config_complete_baseline.json", core / "sensitivity_public_rd04.jsonl"),
            ("sensitivity_municipal_legacy", "sensitivity_municipal_legacy", "Sao Paulo", outputs / "facilities_sp_municipal_legacy.csv", core / "sets" / "main_sp.csv", core / "config_complete_baseline.json", core / "sensitivity_municipal_legacy.jsonl"),
            ("sensitivity_rd04_ge100", "sensitivity_rd04_ge100", "Sao Paulo", outputs / "facilities_sp_rd04_ge100.csv", core / "sets" / "main_sp.csv", core / "config_complete_baseline.json", core / "sensitivity_rd04_ge100.jsonl"),
            ("sensitivity_raw_activity", "sensitivity_raw_activity", "Sao Paulo", outputs / "facilities_sp_raw_activity.csv", core / "sets" / "main_sp.csv", core / "config_complete_baseline.json", core / "sensitivity_raw_activity.jsonl"),
            ("sensitivity_bo_historical_capacity", "sensitivity_bo_historical_capacity", "Bogota", core / "facilities_bo_historical_sensitivity.csv", core / "sets" / "main_bo.csv", core / "config_complete_baseline.json", root / "secondary" / "bo_historical_capacity.jsonl"),
        ]
        for source_name, _, city, fac_path, origins_path, config_path, route_path in source_specs:
            facility_table = _read_facilities(fac_path, city)
            origins, origin_rows = _read_origins(origins_path, city)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            raw_facilities = [{"city": r["city"], "site_id": r["site_id"], "lat": r["lat"], "lon": r["lon"], **{s: "1" if r[s] else "0" for s in SCOPES}} for r in facility_table.values()]
            with route_path.open("w", encoding="utf-8") as handle:
                for origin in origin_rows:
                    handle.write(json.dumps({"record_type": "origin", "city": city, "uid": origin["uid"],
                                             "lat": origin["lat"], "lon": origin["lon"], "status": "ready"},
                                            separators=(",", ":")) + "\n")
                    for candidate in choose_candidates(origin, raw_facilities, 1):
                        site = _site_id(candidate["site_id"], city)
                        target = facility_table[(_city_key(city), site)]
                        flags = {s: bool(target[s]) for s in SCOPES}
                        for mode in MODES:
                            if city == "Sao Paulo" and site == "0000456" and mode == "foot":
                                status, patterns = "timeout", []
                            else:
                                status = "success"
                                duration = {"car": 9.0, "wt": 25.0, "foot": 20.0 if site == "0000123" else 28.0}[mode]
                                if site == "0000456" and mode in {"wt", "foot"}:
                                    duration = 15.0 if mode == "wt" else 16.0
                                leg_mode = "walk" if mode == "foot" else ("bus" if mode == "wt" else "car")
                                walk_dist = 500.0 if site == "0000123" else 1500.0
                                patterns = [{"duration": duration, "journey_duration_min": duration - 1.0 if mode == "wt" else duration,
                                             "initial_departure_gap_min": 1.0 if mode == "wt" else 0.0,
                                             "foot_distance": walk_dist, "legs": [{"mode": leg_mode, "duration": duration, "distance": walk_dist}]}]
                            rec = {"record_type": "route", "city": city, "uid": origin["uid"], "site_id": site,
                                   "mode": mode, "request_hash": route_param_hash(config, mode, city),
                                   "origin": {"lat": origin["lat"], "lon": origin["lon"]},
                                   "destination": {"lat": target["lat"], "lon": target["lon"]},
                                   "facility_flags": flags, "status": status, "patterns": patterns}
                            handle.write(json.dumps(rec, separators=(",", ":")) + "\n")

        # Run the production CLI path against the synthetic sources, with only the
        # fixed real-data facility counts disabled for this deliberately tiny frame.
        output = root / "commonpool_selftest_out"
        df, checks, manifest = run(root, output, require_domain=False, run_analysis=True)
        if not (output / "analysis" / "coverage_curves.csv").is_file():
            raise CommonPoolError("self-test failed: v13_analyse did not create coverage_curves.csv")
        _, sources, coords, _, _ = _plan(root)
        hist_source = next(s for s in sources if s.name == "sensitivity_bo_historical_capacity")
        if hist_source.expected_k != 2:
            raise CommonPoolError("self-test did not apply the BO historical-capacity executed_k")
        sp = df[(df.profile == "baseline_fixed_departure") & (df.city == "Sao Paulo") & (df.uid == "sp:u1") & (df.scope == "R")].set_index("mode")
        if not math.isclose(float(sp.loc["wt", "time_min"]), 20.0):
            raise CommonPoolError("self-test failed: WT did not choose the 20-minute standalone-foot candidate over the 25-minute transit candidate")
        asp = df[(df.profile == "baseline_fixed_departure") & (df.city == "Sao Paulo") & (df.uid == "sp:u1") & (df.scope == "A")].set_index("mode")
        if not (float(asp.loc["wt_cap1km", "time_min"]) >= float(asp.loc["wt_cap2km", "time_min"]) >= float(asp.loc["wt_cap3km", "time_min"]) >= float(asp.loc["wt", "time_min"])):
            raise CommonPoolError("self-test failed: common-pool walk caps are not nested")
        mun = df[(df.profile == "sensitivity_municipal_legacy") & (df.city == "Sao Paulo") & (df.uid == "sp:u1") & (df.scope == "A")].set_index("mode")
        if not bool(mun.loc["wt", "finite_unknown"]):
            raise CommonPoolError("self-test failed: technical WT failure did not propagate to WT bounds")
        if not (float(mun.loc["wt_cap1km", "time_min"]) >= float(mun.loc["wt_cap2km", "time_min"]) >= float(mun.loc["wt_cap3km", "time_min"]) >= float(mun.loc["wt", "time_min"])):
            raise CommonPoolError("self-test failed: walking caps are not nested")
        # Source flags are permitted to differ for the same physical key; the public scenario changes A on CNES 0000123.
        public_source = next(s for s in sources if s.name == "sensitivity_public_rd04")
        main_source = next(s for s in sources if s.name == "main_sp")
        def first_site_flags(src):
            for line in src.path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("record_type") == "route" and r.get("site_id") == "0000123":
                    return r["facility_flags"]
            return None
        if first_site_flags(main_source) == first_site_flags(public_source):
            raise CommonPoolError("self-test fixture did not vary source flags as intended")
        # Mutating one frozen request hash must be rejected by the exact-hash validator.
        bad = {"record_type": "route", "city": "Sao Paulo", "uid": "sp:u1", "site_id": "0000123", "mode": "car",
               "request_hash": "different-time-profile", "origin": {"lat": 0.0, "lon": 0.0},
               "destination": {"lat": 0.01, "lon": 0.0}, "facility_flags": {"R": True, "E": True, "A": False},
               "status": "no_pattern", "patterns": []}
        rejected = False
        try:
            _check_route_record(bad, main_source, coords, {})
        except CommonPoolError:
            rejected = True
        if not rejected:
            raise CommonPoolError("self-test failed: cross-time request hash was accepted")
        print(f"SELF-TEST PASS rows={len(df)} structural_invariants={len(checks)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="frozen v13 project root")
    ap.add_argument("--out", help="new common-pool output directory")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _fixture_self_test()
        if not args.root:
            return
    if not args.root or not args.out:
        ap.error("--root and --out are required unless using --self-test alone")
    run(Path(args.root), Path(args.out))


if __name__ == "__main__":
    main()
