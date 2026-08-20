#!/usr/bin/env python3
"""Summarize intervention results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SETTING_COLUMNS = [
    ("full", "full"),
    ("box", "box"),
    ("crop", "crop"),
    ("context_crop", "context"),
    ("dim_non_target", "dim"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        level = row.get("level")
        if not level:
            continue
        groups["ALL"].append(row)
        groups[level].append(row)

    out: dict[str, dict[str, float]] = {}
    for level, rs in groups.items():
        total = len(rs)
        correct = sum(row.get("correct") == "True" for row in rs)
        wrong = total - correct
        misbinding = sum("misbinding" in row.get("error_type", "") for row in rs)
        adjacent = sum(row.get("error_type") == "adjacent_misbinding" for row in rs)
        out_of_set = sum(row.get("error_type") == "out_of_set_hallucination" for row in rs)
        invalid = sum(row.get("error_type") == "invalid_answer" for row in rs)
        out[level] = {
            "total": float(total),
            "accuracy": correct / total if total else 0.0,
            "overall_mbr": misbinding / total if total else 0.0,
            "error_conditioned_mbr": misbinding / wrong if wrong else 0.0,
            "a_mbr": adjacent / misbinding if misbinding else 0.0,
            "out_of_set_rate": out_of_set / total if total else 0.0,
            "invalid_rate": invalid / total if total else 0.0,
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_setting_metrics(row: dict[str, Any], setting_name: str, metrics: dict[str, float]) -> None:
    row[f"{setting_name}_accuracy"] = metrics["accuracy"]
    row[f"{setting_name}_mbr"] = metrics["overall_mbr"]
    row[f"{setting_name}_error_conditioned_mbr"] = metrics["error_conditioned_mbr"]
    row[f"{setting_name}_a_mbr"] = metrics["a_mbr"]
    row[f"{setting_name}_out_of_set_rate"] = metrics["out_of_set_rate"]
    row[f"{setting_name}_invalid_rate"] = metrics["invalid_rate"]


def add_gap_metrics(row: dict[str, Any], setting_name: str, metrics: dict[str, float], full_metrics: dict[str, float]) -> None:
    if setting_name == "crop":
        gap_name = "crop_binding_gap"
    else:
        gap_name = f"{setting_name}_gap"
    row[gap_name] = metrics["accuracy"] - full_metrics["accuracy"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-detailed", type=Path, required=True)
    parser.add_argument("--box-detailed", type=Path, default=None)
    parser.add_argument("--crop-detailed", type=Path, default=None)
    parser.add_argument("--context-detailed", type=Path, default=None)
    parser.add_argument("--dim-detailed", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("evaluation/intervention_gap_summary.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("evaluation/intervention_gap_summary.json"))
    args = parser.parse_args()

    detailed_paths = {
        "full": args.full_detailed,
        "box": args.box_detailed,
        "crop": args.crop_detailed,
        "context_crop": args.context_detailed,
        "dim_non_target": args.dim_detailed,
    }
    metrics_by_setting = {
        setting_name: metric_rows(read_csv(path))
        for setting_name, path in detailed_paths.items()
        if path is not None
    }

    levels = ["ALL", "L1", "L3"]
    rows: list[dict[str, Any]] = []
    for level in levels:
        if level not in metrics_by_setting["full"]:
            continue
        full_metrics = metrics_by_setting["full"][level]
        row: dict[str, Any] = {
            "level": level,
            "total": int(full_metrics["total"]),
        }
        for setting_name, _short_name in SETTING_COLUMNS:
            if setting_name in metrics_by_setting and level in metrics_by_setting[setting_name]:
                add_setting_metrics(row, setting_name, metrics_by_setting[setting_name][level])
        for setting_name, _short_name in SETTING_COLUMNS:
            if setting_name == "full":
                continue
            if setting_name in metrics_by_setting and level in metrics_by_setting[setting_name]:
                add_gap_metrics(row, setting_name, metrics_by_setting[setting_name][level], full_metrics)
        rows.append(row)

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
