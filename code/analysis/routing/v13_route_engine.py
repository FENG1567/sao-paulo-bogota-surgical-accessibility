#!/usr/bin/env python3
"""Recoverable OTP route batching engine. All SQLite writes stay on main thread."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import signal
import socket
import sqlite3
import sys
import tempfile
import threading
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DEFAULT_TEMPLATE = (
    "{{alias}}: trip(from:{{from}},to:{{to}},dateTime:{{date_time}},"
    "modes:{{modes}}{{trip_args}}) "
    "{ tripPatterns { duration legs { mode duration distance } } }"
)
DEFAULT_QUERY_TEMPLATE = "query { {{aliases}} }"
DEFAULT_MODES = {
    "car": {"literal": "{directMode:car}", "args": "numTripPatterns:15"},
    "wt": {
        "literal": (
            "{accessMode:foot,egressMode:foot,directMode:foot,"
            "transportModes:[{transportMode:bus},{transportMode:rail}]}"
        ),
        "args": (
            'numTripPatterns:15,walkSpeed:1.33,'
            'maxDirectDurationForMode:[{streetMode:foot,duration:"90m"}],'
            'maxAccessEgressDurationForMode:[{streetMode:foot,duration:"90m"}]'
        ),
    },
    "foot": {
        "literal": "{directMode:foot}",
        "args": (
            'numTripPatterns:15,'
            'maxDirectDurationForMode:[{streetMode:foot,duration:"90m"}]'
        ),
    },
}
MODES = ("car", "wt", "foot")
COMPLETED = {"success", "no_pattern"}
TECHNICAL = {"timeout", "http_error", "connection_error", "graphql_error"}
EARTH_RADIUS_KM = 6371.0088

def configured_workers(config):
    ceiling=max(1,min(16,int(config.get('worker_ceiling',8))))
    return max(1,min(ceiling,int(config.get('workers',8))))


class RouteEngineError(Exception):
    pass


class RouteJob:
    def __init__(self, city, origin_lat, origin_lon, dest_lat, dest_lon, mode,
                 param_hash, endpoint, date_time, profile):
        self.city = city
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.dest_lat = dest_lat
        self.dest_lon = dest_lon
        self.mode = mode
        self.param_hash = param_hash
        self.endpoint = endpoint
        self.date_time = date_time
        self.profile = profile

    @property
    def cache_key(self):
        return (
            self.city, coordinate_key(self.origin_lat), coordinate_key(self.origin_lon),
            coordinate_key(self.dest_lat), coordinate_key(self.dest_lon),
            self.param_hash, self.mode,
        )


def clean_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if not value or value.lower() in {"nan", "none", "null"} else value


def finite_float(value):
    value = clean_text(value)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_flag(value):
    value = clean_text(value)
    return value is not None and value.lower() in {"1", "true", "t", "yes", "y"}


def city_match_key(value):
    text = clean_text(value)
    if text is None:
        return None
    normalized = unicodedata.normalize("NFKD", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def coordinate_key(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("coordinate must be finite")
    return value.hex()  # Exact parsed coordinate, never rounded.


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def public_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    host = parsed.hostname or ""
    if parsed.port:
        host += ":" + str(parsed.port)
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if re.search(r"token|secret|password|api.?key|credential", key, re.I):
            value = "***"
        query.append((key, value))
    return urlunsplit((parsed.scheme, host, parsed.path, urlencode(query), parsed.fragment))


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RouteEngineError("CSV has no header: " + Path(path).name)
        return [{str(key): clean_text(value) for key, value in row.items() if key is not None} for row in reader]


def validate_columns(rows, required, label):
    if not rows:
        raise RouteEngineError(label + " CSV has no data rows")
    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise RouteEngineError(label + " CSV missing columns: " + ", ".join(missing))


def valid_coords(lat, lon):
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def normalize_modes(value=None):
    if value is None or value == "all":
        return list(MODES)
    values = [value] if isinstance(value, str) else list(value)
    output = []
    for item in values:
        mode = str(item).strip().lower()
        if mode not in MODES:
            raise RouteEngineError("unsupported mode: " + str(item))
        if mode not in output:
            output.append(mode)
    if not output:
        raise RouteEngineError("at least one mode is required")
    return output


def _lookup_casefold(mapping, key):
    if not isinstance(mapping, Mapping):
        return None
    for candidate, value in mapping.items():
        if str(candidate).lower() == key.lower():
            return value
    return None


def profile_for(config, mode, city):
    modes = config.get("modes", DEFAULT_MODES)
    base = _lookup_casefold(modes, mode)
    if base is None and mode == "wt":
        base = _lookup_casefold(modes, "ourwt")
    if base is None:
        raise RouteEngineError("config.modes missing profile for " + mode)
    profile = {"literal": base} if isinstance(base, str) else dict(base) if isinstance(base, Mapping) else None
    if profile is None:
        raise RouteEngineError("mode profile must be a string or object: " + mode)
    city_profiles = config.get("mode_profiles_by_city", {})
    city_map = city_profiles.get(city, {}) if isinstance(city_profiles, Mapping) else {}
    override = _lookup_casefold(city_map, mode)
    if isinstance(override, str):
        profile["literal"] = override
    elif isinstance(override, Mapping):
        profile.update(override)
    literal = profile.get("literal", profile.get("modes"))
    if not isinstance(literal, str) or not literal.strip():
        raise RouteEngineError("mode profile missing literal/modes: " + mode)
    profile["literal"] = literal
    return profile


def endpoint_for(config, city):
    endpoints = config.get("endpoint_by_city", {})
    endpoint = endpoints.get(city) if isinstance(endpoints, Mapping) else None
    endpoint = endpoint or config.get("endpoint")
    if not endpoint:
        raise RouteEngineError("no GraphQL endpoint for city " + repr(city))
    return str(endpoint)


def datetime_for(config, city):
    date_times = config.get("date_time_by_city", {})
    value = date_times.get(city) if isinstance(date_times, Mapping) else None
    value = value or config.get("date_time")
    if not value:
        raise RouteEngineError("no date_time for city " + repr(city))
    value = str(value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RouteEngineError("invalid ISO date_time for city " + repr(city)) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RouteEngineError("date_time must include timezone offset for " + repr(city))
    return value


def trip_args_for(config, profile):
    count = int(config.get("num_trip_patterns", 15))
    args = profile.get("args", profile.get("trip_args"))
    args = str(args) if args is not None else "numTripPatterns:" + str(count)
    if not re.search(r"\bnumTripPatterns\s*:", args):
        args += ",numTripPatterns:" + str(count)
    return args


def template_for_profile(config, profile):
    """Return the profile query template, adding expected times only by opt-in."""
    template = str(profile.get("template", config.get("template", DEFAULT_TEMPLATE)))
    if not profile.get("measure_elapsed_from_request", False):
        return template
    opening = re.search(r"tripPatterns\s*\{", template)
    if opening is None:
        raise RouteEngineError("elapsed-from-request profile template must select tripPatterns")
    depth, block_end = 1, None
    for index in range(opening.end(), len(template)):
        if template[index] == "{":
            depth += 1
        elif template[index] == "}":
            depth -= 1
            if depth == 0:
                block_end = index
                break
    if block_end is None:
        raise RouteEngineError("elapsed-from-request profile has an unclosed tripPatterns selection")
    selection = template[opening.end():block_end]
    fields = []
    for field in ("expectedStartTime", "expectedEndTime"):
        if not re.search(r"\b" + field + r"\b", selection):
            fields.append(field)
    if fields:
        template = template[:opening.end()] + " " + " ".join(fields) + template[opening.end():]
    return template


def route_param_hash(config, mode, city):
    profile = profile_for(config, mode, city)
    args = trip_args_for(config, profile)
    request_fields = "tripPatterns.duration,legs.mode,duration,distance"
    if profile.get("measure_elapsed_from_request", False):
        request_fields += ",tripPatterns.expectedStartTime,expectedEndTime;duration=end-requested_datetime"
    return sha256_json({
        "endpoint": public_endpoint(endpoint_for(config, city)),
        "date_time": datetime_for(config, city),
        "mode": mode,
        "mode_literal": profile["literal"],
        "trip_args": args,
        "template": template_for_profile(config, profile),
        "query_template": config.get("query_template", DEFAULT_QUERY_TEMPLATE),
        "request_fields": request_fields,
        "profile": profile.get("profile", profile.get("cap_profile", config.get("cap_profile"))),
        "router_profile": config.get("router_profile"),
    })


def choose_candidates(origin, facilities, k):
    """Union the nearest k eligible destinations for each independent R/E/A scope."""
    if k <= 0:
        raise RouteEngineError("k must be positive")
    city = clean_text(origin.get("city"))
    lat, lon = finite_float(origin.get("lat")), finite_float(origin.get("lon"))
    if not valid_coords(lat, lon):
        return []
    by_site = {}
    for row in facilities:
        if city_match_key(row.get("city")) != city_match_key(city):
            continue
        site_id = clean_text(row.get("site_id"))
        flat, flon = finite_float(row.get("lat")), finite_float(row.get("lon"))
        if site_id is None or not valid_coords(flat, flon):
            continue
        if site_id not in by_site:
            by_site[site_id] = {
                "city": city, "site_id": site_id, "lat": flat, "lon": flon,
                "flags": {scope: parse_flag(row.get(scope)) for scope in ("R", "E", "A")},
                "scopes": set(), "distance": haversine_km(lat, lon, flat, flon),
            }
    selected = set()
    for scope in ("R", "E", "A"):
        eligible = [item for item in by_site.values() if item["flags"][scope]]
        eligible.sort(key=lambda item: (item["distance"], item["site_id"]))
        for item in eligible[:k]:
            item["scopes"].add(scope)
            selected.add(item["site_id"])
    return [
        {**by_site[site_id], "scopes": sorted(by_site[site_id]["scopes"]),
         "straight_distance_km": by_site[site_id]["distance"]}
        for site_id in sorted(selected, key=lambda sid: (by_site[sid]["distance"], sid))
    ]


def _format_template(template, values):
    result = template
    for key, value in values.items():
        result = result.replace("{{" + key + "}}", value)
    unresolved = re.findall(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}", result)
    if unresolved:
        raise RouteEngineError("unresolved template placeholders: " + ", ".join(sorted(set(unresolved))))
    return result


def build_graphql_query(jobs, config):
    if not jobs:
        raise RouteEngineError("cannot build empty GraphQL batch")
    limit = min(8, int(config.get("batch_alias", 8)))
    if limit <= 0 or len(jobs) > limit:
        raise RouteEngineError("GraphQL alias count exceeds configured maximum")
    aliases, alias_jobs = [], {}
    for index, job in enumerate(jobs):
        template = template_for_profile(config, job.profile)
        alias = "route_" + str(index)
        args = trip_args_for(config, job.profile)
        values = {
            "alias": alias,
            "from": "{coordinates:{latitude:" + repr(job.origin_lat) + ",longitude:" + repr(job.origin_lon) + "}}",
            "to": "{coordinates:{latitude:" + repr(job.dest_lat) + ",longitude:" + repr(job.dest_lon) + "}}",
            "date_time": json.dumps(job.date_time, ensure_ascii=False),
            "modes": job.profile["literal"],
            "trip_args": "," + args if args else "",
            "args": args,
            "mode": job.mode,
        }
        aliases.append(_format_template(template, values))
        alias_jobs[alias] = job
    query_template = str(config.get("query_template", DEFAULT_QUERY_TEMPLATE))
    return _format_template(query_template, {"aliases": " ".join(aliases)}), alias_jobs


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_aware_datetime(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " is missing")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(label + " is not a valid ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(label + " must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def parse_trip_patterns(trip, job=None):
    if not isinstance(trip, Mapping) or "tripPatterns" not in trip:
        return "graphql_error", None, "missing tripPatterns field"
    patterns = trip["tripPatterns"]
    if not isinstance(patterns, list):
        return "graphql_error", None, "tripPatterns is not a list"
    if not patterns:
        return "no_pattern", [], None
    output = []
    for p, pattern in enumerate(patterns):
        if not isinstance(pattern, Mapping):
            return "graphql_error", None, "malformed pattern " + str(p)
        duration = _number(pattern.get("duration"))
        legs = pattern.get("legs")
        if duration is None or not isinstance(legs, list):
            return "graphql_error", None, "pattern missing duration or legs"
        parsed_legs = []
        for i, leg in enumerate(legs):
            if not isinstance(leg, Mapping):
                return "graphql_error", None, "malformed leg " + str(i)
            mode, leg_duration, distance = clean_text(leg.get("mode")), _number(leg.get("duration")), _number(leg.get("distance"))
            if mode is None or leg_duration is None or distance is None:
                return "graphql_error", None, "leg missing mode/duration/distance"
            parsed_legs.append({"mode": mode, "duration": leg_duration / 60.0, "distance": distance})
        normalized = {"duration": duration / 60.0, "legs": parsed_legs}
        if job is not None and job.profile.get("measure_elapsed_from_request", False):
            expected_start = pattern.get("expectedStartTime")
            expected_end = pattern.get("expectedEndTime")
            try:
                requested_utc = _parse_aware_datetime(job.date_time, "requested datetime")
                start_utc = _parse_aware_datetime(expected_start, "expectedStartTime")
                end_utc = _parse_aware_datetime(expected_end, "expectedEndTime")
            except ValueError as exc:
                return "graphql_error", None, str(exc)
            elapsed = (end_utc - requested_utc).total_seconds() / 60.0
            if elapsed < 0:
                return "graphql_error", None, "elapsed duration from requested datetime is negative"
            normalized.update({
                "journey_duration_min": duration / 60.0,
                "duration": elapsed,
                "initial_departure_gap_min": (start_utc - requested_utc).total_seconds() / 60.0,
                "expected_start_time": expected_start,
                "expected_end_time": expected_end,
                "requested_time": job.date_time,
            })
        output.append(normalized)
    return "success", output, None


def classify_payload(payload, alias_jobs):
    if not isinstance(payload, Mapping):
        return {alias: ("graphql_error", None, "response is not a JSON object") for alias in alias_jobs}
    errors = payload.get("errors")
    errors = errors if isinstance(errors, list) else []
    errors_by_alias = defaultdict(list)
    global_errors = []
    for item in errors:
        if not isinstance(item, Mapping) or not item.get("message"):
            continue
        message = str(item["message"])
        path = item.get("path")
        alias = path[0] if isinstance(path, list) and path else None
        if alias in alias_jobs:
            errors_by_alias[alias].append(message)
        else:
            global_errors.append(message)
    data = payload.get("data")
    if not isinstance(data, Mapping):
        messages = global_errors + [message for group in errors_by_alias.values() for message in group]
        error = "; ".join(messages)[:500] if messages else "missing data object"
        return {alias: ("graphql_error", None, error) for alias in alias_jobs}
    output = {}
    for alias in alias_jobs:
        alias_errors = global_errors + errors_by_alias.get(alias, [])
        if alias_errors:
            output[alias] = ("graphql_error", None, "; ".join(alias_errors)[:500])
        elif alias not in data:
            output[alias] = ("graphql_error", None, "missing response alias " + alias)
        else:
            output[alias] = parse_trip_patterns(data[alias], job=alias_jobs[alias])
    return output


def _is_timeout(exc):
    return isinstance(exc, (TimeoutError, socket.timeout)) or (
        isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout))
    )


def _scrub_error(text, config):
    text = str(text)
    headers = config.get("headers", {})
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if value and re.search(r"authorization|token|secret|password|api.?key|credential", str(key), re.I):
                text = text.replace(str(value), "[REDACTED]")
    endpoints = []
    raw_by_city = config.get("endpoint_by_city", {})
    if isinstance(raw_by_city, Mapping):
        endpoints.extend(str(value) for value in raw_by_city.values())
    if config.get("endpoint"):
        endpoints.append(str(config["endpoint"]))
    for endpoint in endpoints:
        parsed = urlsplit(endpoint)
        for secret in (parsed.username, parsed.password):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if value and re.search(r"token|secret|password|api.?key|credential", key, re.I):
                text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)(token|secret|password|api.?key|credential)(\s*[=:]\s*)[^\s,;]+", r"\1\2[REDACTED]", text)
    return text[:500]


def post_graphql(endpoint, query, config):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    custom = config.get("headers", {})
    if isinstance(custom, Mapping):
        headers.update({str(k): str(v) for k, v in custom.items()})
    request = urllib.request.Request(
        endpoint, data=json.dumps({"query": query}, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(config.get("request_timeout_seconds", 30))) as response:
            raw = response.read()
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        if _is_timeout(exc):
            return "timeout", None, "request timed out"
        if isinstance(exc, urllib.error.HTTPError):
            return "http_error", None, "HTTP status " + str(exc.code)
        return "connection_error", None, _scrub_error(type(exc).__name__ + ": " + str(exc), config)
    try:
        return "response", json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "graphql_error", None, "invalid JSON response: " + type(exc).__name__


def request_batch(jobs, config):
    query, alias_jobs = build_graphql_query(jobs, config)
    endpoints = {job.endpoint for job in jobs}
    if len(endpoints) != 1:
        raise RouteEngineError("batch contains more than one endpoint")
    kind, payload, error = post_graphql(next(iter(endpoints)), query, config)
    if kind != "response":
        return [(job, kind, None, error) for job in jobs]
    classified = classify_payload(payload, alias_jobs)
    return [
        (job, status, patterns, _scrub_error(message, config) if message else None)
        for alias, job in alias_jobs.items()
        for status, patterns, message in [classified[alias]]
    ]


def connect_cache(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS route_cache(
      city TEXT NOT NULL, origin_lat_key TEXT NOT NULL, origin_lon_key TEXT NOT NULL,
      dest_lat_key TEXT NOT NULL, dest_lon_key TEXT NOT NULL, param_hash TEXT NOT NULL,
      mode TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
      error_summary TEXT, patterns_json TEXT, updated_at TEXT NOT NULL,
      PRIMARY KEY(city,origin_lat_key,origin_lon_key,dest_lat_key,dest_lon_key,param_hash,mode));
    CREATE TABLE IF NOT EXISTS run_origins(
      run_id TEXT NOT NULL,row_index INTEGER NOT NULL,city TEXT,uid TEXT,lat REAL,lon REAL,
      stratum TEXT,pop REAL,design_weight REAL,status TEXT NOT NULL,candidate_count INTEGER NOT NULL,
      error_summary TEXT,PRIMARY KEY(run_id,row_index));
    CREATE TABLE IF NOT EXISTS run_candidates(
      run_id TEXT NOT NULL,row_index INTEGER NOT NULL,city TEXT NOT NULL,uid TEXT NOT NULL,
      site_id TEXT NOT NULL,mode TEXT NOT NULL,origin_lat REAL NOT NULL,origin_lon REAL NOT NULL,
      dest_lat REAL NOT NULL,dest_lon REAL NOT NULL,flags_json TEXT NOT NULL,scopes_json TEXT NOT NULL,
      straight_distance_km REAL NOT NULL,param_hash TEXT NOT NULL,
      PRIMARY KEY(run_id,row_index,site_id,mode));
    CREATE INDEX IF NOT EXISTS idx_run_candidates_route
      ON run_candidates(run_id,city,origin_lat,origin_lon,dest_lat,dest_lon,param_hash,mode);
    """)
    return conn


def _cache_row(conn, job):
    return conn.execute(
        "SELECT * FROM route_cache WHERE city=? AND origin_lat_key=? AND origin_lon_key=? AND dest_lat_key=? AND dest_lon_key=? AND param_hash=? AND mode=?",
        job.cache_key,
    ).fetchone()


def ensure_cache_jobs(conn, jobs):
    now = datetime.now(timezone.utc).isoformat()
    for job in jobs:
        conn.execute(
            "INSERT OR IGNORE INTO route_cache(city,origin_lat_key,origin_lon_key,dest_lat_key,dest_lon_key,param_hash,mode,status,attempts,updated_at) VALUES(?,?,?,?,?,?,?,'pending',0,?)",
            (*job.cache_key, now),
        )
    conn.commit()


def store_batch_results(conn, results):
    now = datetime.now(timezone.utc).isoformat()
    for job, status, patterns, error in results:
        current = _cache_row(conn, job)
        attempts = (int(current["attempts"]) if current else 0) + 1
        payload = canonical_json(patterns) if status in COMPLETED and patterns is not None else None
        conn.execute("""
          INSERT INTO route_cache(city,origin_lat_key,origin_lon_key,dest_lat_key,dest_lon_key,param_hash,mode,status,attempts,error_summary,patterns_json,updated_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
          ON CONFLICT(city,origin_lat_key,origin_lon_key,dest_lat_key,dest_lon_key,param_hash,mode)
          DO UPDATE SET status=excluded.status,attempts=excluded.attempts,error_summary=excluded.error_summary,
                        patterns_json=excluded.patterns_json,updated_at=excluded.updated_at
        """, (*job.cache_key, status, attempts, error, payload, now))
    conn.commit()


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_id(origins_path, facilities_path, config, modes, k, selected_uids):
    endpoints = config.get("endpoint_by_city", {})
    safe_endpoints = {str(city): public_endpoint(str(url)) for city, url in endpoints.items()} if isinstance(endpoints, Mapping) else {}
    public_config = {
        "endpoint_by_city": safe_endpoints,
        "endpoint": public_endpoint(str(config["endpoint"])) if config.get("endpoint") else None,
        "date_time_by_city": config.get("date_time_by_city", {}),
        "date_time": config.get("date_time"),
        "template": config.get("template", DEFAULT_TEMPLATE),
        "query_template": config.get("query_template", DEFAULT_QUERY_TEMPLATE),
        "modes": config.get("modes", DEFAULT_MODES),
        "mode_profiles_by_city": config.get("mode_profiles_by_city", {}),
        "num_trip_patterns": config.get("num_trip_patterns", 15),
        "cap_profile": config.get("cap_profile"),
        "router_profile": config.get("router_profile"),
    }
    return sha256_json({
        "origins_sha256": _file_hash(origins_path),
        "facilities_sha256": _file_hash(facilities_path),
        "config": public_config, "modes": list(modes), "k": k,
        "selected_uids": sorted(selected_uids) if selected_uids is not None else None,
    })[:24]


def _selected_uids(config, uids_file):
    selected = None
    if "selected_uids" in config:
        values = config["selected_uids"]
        if not isinstance(values, list):
            raise RouteEngineError("selected_uids must be a list")
        selected = {str(value) for value in values}
    if uids_file:
        with open(uids_file, "r", encoding="utf-8-sig") as handle:
            from_file = {line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")}
        selected = from_file if selected is None else selected & from_file
    return selected


def _prepare_run(conn, run_id, origins, facilities, k, modes, config):
    conn.execute("DELETE FROM run_origins WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM run_candidates WHERE run_id=?", (run_id,))
    jobs = {}
    job_origins = defaultdict(set)
    for row_index, origin in enumerate(origins):
        city, uid = clean_text(origin.get("city")), clean_text(origin.get("uid"))
        lat, lon = finite_float(origin.get("lat")), finite_float(origin.get("lon"))
        invalid = city is None or uid is None or not valid_coords(lat, lon)
        matching_facilities = [
            row for row in facilities
            if city is not None and city_match_key(row.get("city")) == city_match_key(city)
        ]
        candidates = [] if invalid or not matching_facilities else choose_candidates(origin, facilities, k)
        if invalid:
            status, error = "invalid_input", "missing/invalid city, uid, or origin coordinates"
        elif not matching_facilities:
            status, error = "facility_city_mismatch", "no facility rows match this origin city"
        elif not candidates:
            status, error = "no_candidate", "same-city facility rows exist but no valid R/E/A candidate was eligible"
        else:
            status, error = "ready", None
        conn.execute(
            "INSERT INTO run_origins VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id,row_index,city,uid,lat,lon,clean_text(origin.get("stratum")),finite_float(origin.get("pop")),
             finite_float(origin.get("design_weight")),status,len(candidates),error),
        )
        if status != "ready":
            continue
        for candidate in candidates:
            for mode in modes:
                try:
                    profile = profile_for(config, mode, city)
                    endpoint, date_time = endpoint_for(config, city), datetime_for(config, city)
                    param_hash = route_param_hash(config, mode, city)
                except RouteEngineError as exc:
                    conn.execute(
                        "INSERT INTO run_candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id,row_index,city,uid,candidate["site_id"],mode,lat,lon,candidate["lat"],candidate["lon"],
                         canonical_json(candidate["flags"]),canonical_json(candidate["scopes"]),candidate["straight_distance_km"],
                         "CONFIG_ERROR:" + str(exc)),
                    )
                    continue
                job = RouteJob(city,lat,lon,candidate["lat"],candidate["lon"],mode,param_hash,endpoint,date_time,profile)
                jobs.setdefault(job.cache_key, job)
                job_origins[job.cache_key].add(row_index)
                conn.execute(
                    "INSERT INTO run_candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id,row_index,city,uid,candidate["site_id"],mode,lat,lon,candidate["lat"],candidate["lon"],
                     canonical_json(candidate["flags"]),canonical_json(candidate["scopes"]),candidate["straight_distance_km"],param_hash),
                )
    conn.commit()
    ensure_cache_jobs(conn, jobs.values())
    return list(jobs.values()), job_origins


def _pending(conn, jobs, max_attempts):
    output = []
    for job in jobs:
        row = _cache_row(conn, job)
        if row is None or row["status"] not in COMPLETED and not (row["status"] in TECHNICAL and int(row["attempts"]) >= max_attempts):
            output.append(job)
    return output


def _counts(conn, jobs, max_attempts):
    complete = failed = pending = 0
    for job in jobs:
        row = _cache_row(conn, job)
        status = str(row["status"]) if row else "pending"
        if status in COMPLETED:
            complete += 1
        elif status in TECHNICAL and int(row["attempts"]) >= max_attempts:
            failed += 1
        else:
            pending += 1
    return complete, failed, pending


def _load_json(value, default):
    try:
        return json.loads(value) if value is not None else default
    except json.JSONDecodeError:
        return default


def _foot_distance(pattern):
    return sum(
        float(leg.get("distance", 0))
        for leg in pattern.get("legs", [])
        if str(leg.get("mode", "")).lower() in {"walk", "foot"}
    )


def materialize_jsonl(conn, run_id, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(temp_name)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            for row in conn.execute("SELECT * FROM run_origins WHERE run_id=? ORDER BY row_index", (run_id,)):
                record = {
                    "record_type": "origin", "run_id": run_id, "row_index": row["row_index"],
                    "city": row["city"], "uid": row["uid"], "lat": row["lat"], "lon": row["lon"],
                    "stratum": row["stratum"], "pop": row["pop"], "design_weight": row["design_weight"],
                    "status": row["status"], "candidate_count": row["candidate_count"], "error": row["error_summary"],
                }
                handle.write(canonical_json(record) + "\n")
            candidates = conn.execute(
                "SELECT * FROM run_candidates WHERE run_id=? ORDER BY row_index,site_id,mode", (run_id,)
            )
            for row in candidates:
                param_hash = row["param_hash"]
                config_error = str(param_hash).startswith("CONFIG_ERROR:")
                route = None
                if not config_error:
                    route = conn.execute(
                        "SELECT status,attempts,error_summary,patterns_json FROM route_cache WHERE city=? AND origin_lat_key=? AND origin_lon_key=? AND dest_lat_key=? AND dest_lon_key=? AND param_hash=? AND mode=?",
                        (row["city"],coordinate_key(row["origin_lat"]),coordinate_key(row["origin_lon"]),
                         coordinate_key(row["dest_lat"]),coordinate_key(row["dest_lon"]),param_hash,row["mode"]),
                    ).fetchone()
                patterns = _load_json(route["patterns_json"], []) if route else []
                normalized_patterns = [
                    {**item, "pattern_index": i, "foot_distance": _foot_distance(item)}
                    for i, item in enumerate(patterns or [])
                ]
                record = {
                    "record_type": "route", "run_id": run_id, "row_index": row["row_index"],
                    "city": row["city"], "uid": row["uid"], "site_id": row["site_id"], "mode": row["mode"],
                    "origin": {"lat": row["origin_lat"], "lon": row["origin_lon"]},
                    "destination": {"lat": row["dest_lat"], "lon": row["dest_lon"]},
                    "facility_flags": _load_json(row["flags_json"], {}),
                    "selected_scopes": _load_json(row["scopes_json"], []),
                    "straight_distance_km": row["straight_distance_km"],
                    "status": "config_error" if config_error else (route["status"] if route else "pending"),
                    "attempts": int(route["attempts"]) if route else 0,
                    "error": str(param_hash)[len("CONFIG_ERROR:"):] if config_error else (route["error_summary"] if route else None),
                    "request_hash": None if config_error else param_hash,
                    "patterns": normalized_patterns,
                    "units": {"duration": "min", "distance": "m", "straight_distance": "km"},
                }
                handle.write(canonical_json(record) + "\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _write_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(state) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _batches(jobs, alias_limit):
    grouped = defaultdict(list)
    for job in jobs:
        grouped[(job.endpoint, job.date_time)].append(job)
    output = []
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda j: (j.city,j.mode,j.origin_lat,j.origin_lon,j.dest_lat,j.dest_lon,j.param_hash))
        output.extend([values[i:i + alias_limit] for i in range(0, len(values), alias_limit)])
    return output


def _route_state_class(status, attempts, max_attempts):
    if status in COMPLETED:
        return "success_or_no_pattern"
    if status in TECHNICAL and attempts >= max_attempts:
        return "technical_failed_after_retries"
    return "pending"


def _initial_route_counts(conn, jobs, max_attempts, job_origins):
    states = {}
    counts = {"success_or_no_pattern": 0, "technical_failed_after_retries": 0, "pending": 0}
    failed_origins = set()
    for job in jobs:
        row = _cache_row(conn, job)
        status, attempts = (str(row["status"]), int(row["attempts"])) if row else ("pending", 0)
        states[job.cache_key] = (status, attempts)
        label = _route_state_class(status, attempts, max_attempts)
        counts[label] += 1
        if label == "technical_failed_after_retries":
            failed_origins.update(job_origins.get(job.cache_key, ()))
    return states, counts, failed_origins


def _origin_status_counts(conn, run_id):
    return {
        str(row["status"]): int(row["n"])
        for row in conn.execute(
            "SELECT status,COUNT(*) AS n FROM run_origins WHERE run_id=? GROUP BY status", (run_id,)
        )
    }


def _state(run_id, status, origins, jobs, batches, workers, alias_limit, counts, failed_origins, origin_counts, detail=None):
    origin_errors = sum(origin_counts.get(key, 0) for key in ("invalid_input", "facility_city_mismatch", "no_candidate"))
    return {
        "run_id": run_id, "status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
        "progress": {"origins": origins, "route_profiles": len(jobs),
                     "success_or_no_pattern": counts["success_or_no_pattern"],
                     "technical_failed_after_retries": counts["technical_failed_after_retries"],
                     "origins_with_terminal_technical_failures": len(failed_origins),
                     "origins_invalid_input": origin_counts.get("invalid_input", 0),
                     "origins_facility_city_mismatch": origin_counts.get("facility_city_mismatch", 0),
                     "origins_no_candidate": origin_counts.get("no_candidate", 0),
                     "origins_with_input_errors": origin_errors,
                     "pending": counts["pending"], "request_batches": batches,
                     "workers": workers, "batch_alias_limit": alias_limit},
        "detail": detail,
    }


def run_engine(config_path, origins_path, facilities_path, cache_path, out_path, *,
               limit_origins=None, mode=None, k=None, uids_file=None, state_path=None):
    with open(config_path, "r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, Mapping):
        raise RouteEngineError("config JSON root must be an object")
    modes = normalize_modes(mode if mode is not None else config.get("mode", "all"))
    origins = read_csv_rows(origins_path)
    facilities = read_csv_rows(facilities_path)
    validate_columns(origins, ("city","uid","lat","lon","stratum","pop","design_weight"), "origins")
    validate_columns(facilities, ("city","site_id","lat","lon","R","E","A"), "facilities")
    selected = _selected_uids(config, uids_file)
    if selected is not None:
        origins = [row for row in origins if clean_text(row.get("uid")) in selected]
        if not origins:
            raise RouteEngineError("fixed UID subset matched no origin rows")
    if limit_origins is not None:
        if limit_origins <= 0:
            raise RouteEngineError("--limit-origins must be positive")
        origins = origins[:limit_origins]
    k = int(k if k is not None else config.get("k", 8))
    if k <= 0:
        raise RouteEngineError("k must be positive")
    workers = configured_workers(config)
    alias_limit = max(1, min(8, int(config.get("batch_alias", 8))))
    max_attempts = 1 + max(0, int(config.get("max_retries", 2)))
    run_id = _run_id(origins_path, facilities_path, config, modes, k, selected)
    # A fixed pilot prefix and a later full cohort must have distinct materialized runs.
    run_id = sha256_json({
        "base_run_id": run_id,
        "selected_origin_rows": [
            [clean_text(row.get("city")), clean_text(row.get("uid")),
             finite_float(row.get("lat")), finite_float(row.get("lon"))]
            for row in origins
        ],
    })[:24]
    conn = connect_cache(cache_path)
    state_file = Path(state_path) if state_path else Path(str(out_path) + ".status.json")
    stopping = threading.Event()
    old_handlers = {}
    def stop_handler(_signum, _frame):
        stopping.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[sig] = signal.signal(sig, stop_handler)
        except (ValueError, OSError):
            pass
    batch_count, jobs = 0, []
    job_origins = {}
    route_states, route_counts, failed_origins = {}, {
        "success_or_no_pattern": 0, "technical_failed_after_retries": 0, "pending": 0
    }, set()
    origin_counts = {}
    submitted = {}
    processed = set()
    pool = None
    materialize_every = max(0, int(config.get("materialize_every_batches", 0)))

    def persist_future(future, batch):
        nonlocal batch_count
        try:
            results = future.result()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            error = _scrub_error(type(exc).__name__ + ": " + str(exc), config)
            results = [(job, "connection_error", None, error) for job in batch]
        store_batch_results(conn, results)
        for job, status, _patterns, _error in results:
            old_status, old_attempts = route_states.get(job.cache_key, ("pending", 0))
            old_label = _route_state_class(old_status, old_attempts, max_attempts)
            new_attempts = old_attempts + 1
            new_label = _route_state_class(status, new_attempts, max_attempts)
            if old_label != new_label:
                route_counts[old_label] -= 1
                route_counts[new_label] += 1
            route_states[job.cache_key] = (status, new_attempts)
            if new_label == "technical_failed_after_retries":
                failed_origins.update(job_origins.get(job.cache_key, ()))
        processed.add(future)
        batch_count += 1

    try:
        jobs, job_origins = _prepare_run(conn, run_id, origins, facilities, k, modes, config)
        origin_counts = _origin_status_counts(conn, run_id)
        route_states, route_counts, failed_origins = _initial_route_counts(conn,jobs,max_attempts,job_origins)
        _write_state(state_file, _state(run_id,"running",len(origins),jobs,batch_count,workers,alias_limit,route_counts,failed_origins,origin_counts))
        print(f"run_id={run_id} origins={len(origins)} route_profiles={len(jobs)} workers={workers} batch_alias={alias_limit}", flush=True)
        config_errors = conn.execute(
            "SELECT COUNT(*) FROM run_candidates WHERE run_id=? AND param_hash LIKE 'CONFIG_ERROR:%'",
            (run_id,),
        ).fetchone()[0]
        origin_errors = sum(origin_counts.get(key, 0) for key in ("invalid_input", "facility_city_mismatch", "no_candidate"))
        if config_errors or origin_errors or not jobs:
            reasons = []
            if config_errors:
                reasons.append(str(config_errors) + " candidate-mode rows have configuration errors")
            if origin_counts.get("facility_city_mismatch"):
                reasons.append(str(origin_counts["facility_city_mismatch"]) + " origins have no matching facility city")
            if origin_counts.get("no_candidate"):
                reasons.append(str(origin_counts["no_candidate"]) + " origins have no eligible scope candidate")
            if origin_counts.get("invalid_input"):
                reasons.append(str(origin_counts["invalid_input"]) + " origins have invalid required fields")
            if not jobs:
                reasons.append("no route jobs were prepared")
            materialize_jsonl(conn,run_id,out_path)
            failed_state = _state(
                run_id,"failed",len(origins),jobs,batch_count,workers,alias_limit,route_counts,failed_origins,
                origin_counts,"; ".join(reasons),
            )
            _write_state(state_file,failed_state)
            print(f"status=failed batches=0 out={Path(out_path).resolve()} state={state_file.resolve()}", flush=True)
            return failed_state
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="otp-route")
        try:
            while not stopping.is_set():
                pending = _pending(conn, jobs, max_attempts)
                if not pending:
                    break
                futures = {pool.submit(request_batch, batch, config): batch for batch in _batches(pending, alias_limit)}
                submitted = futures
                if not futures:
                    break
                for future in concurrent.futures.as_completed(futures):
                    if stopping.is_set():
                        for queued in futures:
                            if not queued.done():
                                queued.cancel()
                        break
                    batch = futures[future]
                    persist_future(future, batch)
                    if batch_count % 10 == 0:
                        print(
                            f"progress batches={batch_count} "
                            f"success_or_no_pattern={route_counts['success_or_no_pattern']} "
                            f"failed={route_counts['technical_failed_after_retries']} "
                            f"pending={route_counts['pending']}", flush=True
                        )
                        _write_state(state_file,_state(run_id,"running",len(origins),jobs,batch_count,workers,alias_limit,route_counts,failed_origins,origin_counts))
                    if materialize_every and batch_count % materialize_every == 0:
                        materialize_jsonl(conn,run_id,out_path)
                if stopping.is_set():
                    break
            pool.shutdown(wait=True, cancel_futures=True)
            # In-flight requests finish within the configured HTTP timeout. Save their
            # results before exposing an interrupted checkpoint; queued requests cancel.
            for future, batch in submitted.items():
                if future not in processed and not future.cancelled():
                    if future.done():
                        persist_future(future, batch)
            materialize_jsonl(conn,run_id,out_path)
            config_errors = conn.execute(
                "SELECT COUNT(*) FROM run_candidates WHERE run_id=? AND param_hash LIKE 'CONFIG_ERROR:%'",
                (run_id,),
            ).fetchone()[0]
            terminal_failures = route_counts["technical_failed_after_retries"]
            origin_errors = sum(origin_counts.get(key, 0) for key in ("invalid_input", "facility_city_mismatch", "no_candidate"))
            no_jobs = len(jobs) == 0
            status = "interrupted" if stopping.is_set() else ("failed" if config_errors or terminal_failures or origin_errors or no_jobs else "done")
            details = []
            if config_errors:
                details.append("configuration errors in " + str(config_errors) + " candidate-mode rows")
            if terminal_failures:
                details.append(
                    str(terminal_failures) + " route profiles exhausted technical retries across "
                    + str(len(failed_origins)) + " origin rows"
                )
            if origin_counts.get("facility_city_mismatch"):
                details.append(str(origin_counts["facility_city_mismatch"]) + " origins have no matching facility city")
            if origin_counts.get("no_candidate"):
                details.append(str(origin_counts["no_candidate"]) + " origins have no eligible scope candidate")
            if origin_counts.get("invalid_input"):
                details.append(str(origin_counts["invalid_input"]) + " origins have invalid required fields")
            if no_jobs:
                details.append("no route jobs were prepared")
            final = _state(
                run_id,status,len(origins),jobs,batch_count,workers,alias_limit,route_counts,failed_origins,origin_counts,
                "; ".join(details) or None,
            )
            _write_state(state_file,final)
            print(f"status={status} batches={batch_count} out={Path(out_path).resolve()} state={state_file.resolve()}", flush=True)
            return final
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
    except BaseException as exc:
        try:
            materialize_jsonl(conn,run_id,out_path)
            _write_state(
                state_file,
                _state(
                    run_id,"failed",len(origins),jobs,batch_count,workers,alias_limit,
                    route_counts,failed_origins,origin_counts,_scrub_error(type(exc).__name__ + ": " + str(exc),config),
                ),
            )
        except BaseException:
            pass
        raise
    finally:
        conn.close()
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig,handler)
            except (ValueError,OSError):
                pass


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Recoverable OTP route batch runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--origins", required=True)
    parser.add_argument("--facilities", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-origins", type=int)
    parser.add_argument("--mode", default=None, help="all, car, wt/WT, or foot")
    parser.add_argument("--k", type=int)
    parser.add_argument("--uids-file", help="fixed UID list; no random selection")
    parser.add_argument("--state", help="status JSON; defaults to OUT.status.json")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        state = run_engine(args.config,args.origins,args.facilities,args.cache,args.out,
                           limit_origins=args.limit_origins,mode=args.mode,k=args.k,
                           uids_file=args.uids_file,state_path=args.state)
        return 130 if state["status"] == "interrupted" else (1 if state["status"] == "failed" else 0)
    except (RouteEngineError,OSError,sqlite3.Error,json.JSONDecodeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr, flush=True)
        return 2
    except BaseException as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
