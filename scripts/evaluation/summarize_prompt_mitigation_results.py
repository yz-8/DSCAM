#!/usr/bin/env python3
"""Summarize full-image vs Instance-First Prompt results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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
        invalid = sum(row.get("error_type") == "invalid_or_unparseable" for row in rs)
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


def add_metrics(row: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    row[f"{prefix}_accuracy"] = metrics["accuracy"]
    row[f"{prefix}_mbr"] = metrics["overall_mbr"]
    row[f"{prefix}_error_conditioned_mbr"] = metrics["error_conditioned_mbr"]
    row[f"{prefix}_a_mbr"] = metrics["a_mbr"]
    row[f"{prefix}_out_of_set_rate"] = metrics["out_of_set_rate"]
    row[f"{prefix}_invalid_rate"] = metrics["invalid_rate"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-detailed", type=Path, required=True)
    parser.add_argument("--instance-first-detailed", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    full = metric_rows(read_csv(args.full_detailed))
    inst = metric_rows(read_csv(args.instance_first_detailed))

    levels = ["ALL", "L1", "L2", "L3", "L4"]
    rows: list[dict[str, Any]] = []
    for level in levels:
        if level not in full or level not in inst:
            continue
        row: dict[str, Any] = {"level": level, "total": int(full[level]["total"])}
        add_metrics(row, "full", full[level])
        add_metrics(row, "instance_first", inst[level])
        row["accuracy_gap"] = inst[level]["accuracy"] - full[level]["accuracy"]
        row["mbr_gap"] = inst[level]["overall_mbr"] - full[level]["overall_mbr"]
        row["a_mbr_gap"] = inst[level]["a_mbr"] - full[level]["a_mbr"]
        row["out_of_set_gap"] = inst[level]["out_of_set_rate"] - full[level]["out_of_set_rate"]
        rows.append(row)

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
