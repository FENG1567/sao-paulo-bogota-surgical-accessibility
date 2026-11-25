#!/usr/bin/env python3
"""Copy publication table CSVs and render readable Markdown views without recalculation."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "data" / "tables"
CROSSWALK = ROOT / "metadata" / "table_crosswalk.json"
EXPORT_DIR = ROOT / "artifacts" / "tables"


def safe_source_path(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe table source path: {name}")
    path = (TABLES_DIR / Path(*relative.parts)).resolve()
    if not path.is_relative_to(TABLES_DIR.resolve()):
        raise ValueError(f"Table source escapes data/tables: {name}")
    return path


def output_names(value: str) -> tuple[str, str]:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"Unsafe table output name: {value}")
    if path.suffix:
        return path.name, path.with_suffix(".md").name
    return path.name + ".csv", path.name + ".md"


def markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def markdown_table(title: str, rows: list[list[str]]) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        return "\n".join(lines + ["(Empty CSV.)", ""])
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{title}: inconsistent CSV row widths; no cell was dropped.")
    header = [markdown_cell(value) for value in rows[0]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tables"),
        help="Output directory (relative paths are resolved from the release root).",
    )
    args = parser.parse_args()

    if not TABLES_DIR.is_dir():
        raise SystemExit(f"Missing repository input directory: data/tables ({TABLES_DIR})")
    if not CROSSWALK.is_file():
        raise SystemExit("Missing metadata/table_crosswalk.json.")
    crosswalk = json.loads(CROSSWALK.read_text(encoding="utf-8"))
    entries = crosswalk.get("tables") if isinstance(crosswalk, dict) else None
    if not isinstance(entries, list) or not entries:
        raise SystemExit("table_crosswalk.json must contain a non-empty 'tables' list.")

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_outputs: set[str] = set()
    report_tables = []
    for entry in entries:
        source_name = entry["source"]
        table_name = entry["table"]
        output_stem = entry["output"]
        role = entry.get("role", "publication")
        csv_name, markdown_name = output_names(output_stem)
        if csv_name in seen_outputs:
            raise ValueError(f"Duplicate table_crosswalk output name: {csv_name}")
        seen_outputs.add(csv_name)

        source_path = safe_source_path(source_name)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing table source CSV: {source_name}")
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        target_csv = output_dir / csv_name
        shutil.copyfile(source_path, target_csv)
        markdown = markdown_table(str(table_name), rows)
        (output_dir / markdown_name).write_text(markdown, encoding="utf-8")

        row_count = max(len(rows) - 1, 0)
        column_count = len(rows[0]) if rows else 0
        report_tables.append(
            {
                "source": source_name,
                "table": table_name,
                "role": role,
                "csv": csv_name,
                "markdown": markdown_name,
                "rows": row_count,
                "columns": column_count,
                "csv_copy": "byte-for-byte",
            }
        )

    report = {
        "operation": "Source CSVs copied and rendered as Markdown.",
        "statistical_recalculation": False,
        "value_transformations": False,
        "tables": report_tables,
    }
    (output_dir / "reproduction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
