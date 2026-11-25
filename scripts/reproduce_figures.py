#!/usr/bin/env python3
"""Re-render the eight publication figures from repository source_data CSVs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "source_data"
REFERENCE_FIGURES = ROOT / "reference" / "figures"
WRAPPER_SOURCE = ROOT / "code" / "figures" / "reader_layout_20261125"
MAIN_SCRIPT = "relayout_main_final.py"
SUPPLEMENT_SCRIPT = "relayout_supplement.py"
MAIN_STEMS = {
    1: "Figure1_population_facilities",
    2: "Figure2_mode_scope_threshold",
    3: "Figure3_walk_caps",
    4: "Figure4_income_geography",
    5: "Figure5_validation_sensitivity",
}
SUPPLEMENT_STEMS = {
    "S1": "FigureS1_car_time_sensitivity",
    "S2": "FigureS2_paired_sensitivity",
    "S3": "FigureS3_bogota_classification",
}
EXPORT_SUFFIXES = (".pdf", ".svg", ".png", ".tiff", ".style.json")
FONT_REGULAR = ("arial.ttf",)
FONT_BOLD = ("arialbd.ttf",)


def font_pair(directory: Path | None) -> tuple[Path, Path] | None:
    if directory is None or not directory.is_dir():
        return None
    files = {item.name.casefold(): item for item in directory.iterdir() if item.is_file()}
    regular = next((files[name.casefold()] for name in FONT_REGULAR if name.casefold() in files), None)
    bold = next((files[name.casefold()] for name in FONT_BOLD if name.casefold() in files), None)
    return (regular, bold) if regular is not None and bold is not None else None


def resolve_fonts(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.font_dir is not None:
        candidates = [args.font_dir.expanduser()]
    elif os.name == "nt":
        candidates = [Path("C:/Windows/Fonts")]
    else:
        candidates = []

    for candidate in candidates:
        pair = font_pair(candidate)
        if pair is not None:
            return pair

    location = (
        "the configured font directory"
        if args.font_dir is not None
        else "C:/Windows/Fonts on Windows or an installed --font-dir elsewhere"
    )
    raise SystemExit(
        "Arial regular (arial.ttf) and bold (arialbd.ttf) are required; "
        f"not found in {location}. Install/configure Arial with --font-dir PATH."
    )


def normalize_main(values: list[str] | None, defaults: list[int]) -> list[int]:
    if values is None:
        return defaults
    result: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip().upper()
            if token == "ALL":
                return list(MAIN_STEMS)
            if token.startswith("FIGURE"):
                token = token.removeprefix("FIGURE").split("_", 1)[0]
            try:
                number = int(token)
            except ValueError as error:
                raise ValueError(f"Unknown main figure selector: {token}") from error
            if number not in MAIN_STEMS:
                raise ValueError(f"Unknown main figure selector: {token}")
            result.add(number)
    return sorted(result)


def normalize_supplement(values: list[str] | None, defaults: list[str]) -> list[str]:
    if values is None:
        return defaults
    result: set[str] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip().upper()
            if token == "ALL":
                return list(SUPPLEMENT_STEMS)
            if token.startswith("FIGURE"):
                token = token.removeprefix("FIGURE")
            if token.isdigit():
                token = "S" + token
            if token not in SUPPLEMENT_STEMS:
                raise ValueError(f"Unknown supplement selector: {token}")
            result.add(token)
    return sorted(result, key=lambda value: int(value[1:]))


def copy_exports(source_dir: Path, stems: list[str], output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for stem in stems:
        for suffix in EXPORT_SUFFIXES:
            source = source_dir / f"{stem}{suffix}"
            if not source.is_file():
                raise RuntimeError(f"Figure wrapper did not create {stem}{suffix}.")
            target = output_dir / source.name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def count_panel_tags(style_files: list[Path]) -> int:
    count = 0
    for style_file in style_files:
        record = json.loads(style_file.read_text(encoding="utf-8"))
        count += sum(
            str(text.get("gid", "")).startswith("panel_tag_")
            for text in record.get("text", [])
        )
    return count


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/figures"),
        help="Output directory (relative paths are resolved from the release root).",
    )
    result.add_argument(
        "--font-dir",
        type=Path,
        help="Directory containing arial.ttf and arialbd.ttf; never downloaded by this script.",
    )
    result.add_argument(
        "--figures",
        nargs="+",
        metavar="FIGURE",
        help="Main figures: 1-5, Figure1-Figure5, comma lists, or all.",
    )
    result.add_argument(
        "--supplement",
        nargs="+",
        metavar="FIGURE",
        help="Supplementary figures: S1-S3, 1-3, comma lists, or all.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if not SOURCE_DATA.is_dir():
        raise SystemExit(f"Missing repository input directory: data/source_data ({SOURCE_DATA})")
    required_wrappers = (
        "figure_style.py",
        "relayout_main.py",
        MAIN_SCRIPT,
        SUPPLEMENT_SCRIPT,
    )
    missing = [name for name in required_wrappers if not (WRAPPER_SOURCE / name).is_file()]
    if missing:
        raise SystemExit(f"Missing figure wrapper source file(s): {', '.join(missing)}")

    no_selection = args.figures is None and args.supplement is None
    try:
        main_ids = normalize_main(args.figures, list(MAIN_STEMS) if no_selection else [])
        supplement_ids = normalize_supplement(
            args.supplement, list(SUPPLEMENT_STEMS) if no_selection else []
        )
    except ValueError as error:
        parser().error(str(error))

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fonts = resolve_fonts(args)

    copied: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="portable-figure-reproduction-") as temporary:
        work = Path(temporary)
        wrapper_dir = work / "wrapper"
        wrapper_dir.mkdir()
        for name in required_wrappers:
            shutil.copy2(WRAPPER_SOURCE / name, wrapper_dir / name)

        run_env = os.environ.copy()
        run_env.update(
            {
                "FROZEN_SOURCE_DATA": str(SOURCE_DATA),
                "FROZEN_REFERENCE_PDF": str(REFERENCE_FIGURES),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "MPLCONFIGDIR": str(work / "matplotlib-cache"),
            }
        )
        staged_fonts = wrapper_dir / "fonts"
        staged_fonts.mkdir()
        shutil.copy2(fonts[0], staged_fonts / "arial.ttf")
        shutil.copy2(fonts[1], staged_fonts / "arialbd.ttf")
        run_env["ARIAL_FONT_DIR"] = str(staged_fonts)

        if main_ids:
            command = [
                sys.executable,
                str(wrapper_dir / MAIN_SCRIPT),
                "--figures",
                *[str(number) for number in main_ids],
                "--run",
                "run01",
            ]
            subprocess.run(command, check=True, cwd=wrapper_dir, env=run_env)
            main_output = wrapper_dir / "figures_main" / "run01"
            stems = [MAIN_STEMS[number] for number in main_ids]
            copied.extend(copy_exports(main_output, stems, output_dir))

        if supplement_ids:
            command = [
                sys.executable,
                str(wrapper_dir / SUPPLEMENT_SCRIPT),
                "--run-id",
                "run01",
                "--figures",
                *supplement_ids,
            ]
            subprocess.run(command, check=True, cwd=wrapper_dir, env=run_env)
            supplement_output = wrapper_dir / "figures_supp" / "run01"
            stems = [SUPPLEMENT_STEMS[number] for number in supplement_ids]
            copied.extend(copy_exports(supplement_output, stems, output_dir))

        style_files = [path for path in copied if path.name.endswith(".style.json")]
        panel_tags = count_panel_tags(style_files)
        complete_release = main_ids == list(MAIN_STEMS) and supplement_ids == list(SUPPLEMENT_STEMS)
        if complete_release and panel_tags != 32:
            raise RuntimeError(f"Expected 32 panel tags across eight figures; found {panel_tags}.")

    report = {
        "operation": "Figures re-rendered from data/source_data using the supplied wrappers.",
        "reference_figures_used_for_rendering": False,
        "rendered_main_figures": [MAIN_STEMS[number] for number in main_ids],
        "rendered_supplementary_figures": [SUPPLEMENT_STEMS[number] for number in supplement_ids],
        "panel_tag_count": panel_tags,
        "expected_full_release_panel_tags": 32,
        "font_mode": "Arial",
        "outputs": [path.name for path in copied],
        "statistical_recalculation": False,
    }
    (output_dir / "reproduction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
