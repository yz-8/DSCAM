#!/usr/bin/env python3
"""Image-cluster bootstrap confidence intervals for InstaBind-Lite outputs."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from evaluate_outputs import evaluate, read_jsonl


Metric = Callable[[list[dict[str, Any]]], float]


def accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0


def mbr(rows: list[dict[str, Any]]) -> float:
    return (
        sum("misbinding" in str(row["error_type"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def a_mbr(rows: list[dict[str, Any]]) -> float:
    misbindings = [row for row in rows if "misbinding" in str(row["error_type"])]
    return (
        sum(row["error_type"] == "adjacent_misbinding" for row in misbindings)
        / len(misbindings)
        if misbindings
        else 0.0
    )


def percentile_interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    low = ordered[round(0.025 * (len(ordered) - 1))]
    high = ordered[round(0.975 * (len(ordered) - 1))]
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("model_outputs", type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.iterations < 2:
        raise ValueError("--iterations must be at least 2")

    questions = read_jsonl(args.questions)
    outputs = read_jsonl(args.model_outputs)
    detailed, _, _ = evaluate(questions, outputs)
    image_by_question = {row["question_id"]: row["image_id"] for row in questions}

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        clusters[image_by_question[row["question_id"]]].append(row)
    image_ids = sorted(clusters)
    if not image_ids:
        raise ValueError("No evaluated image clusters were found")

    metrics: dict[str, Metric] = {"accuracy": accuracy, "mbr": mbr, "a_mbr": a_mbr}
    observed_rows = [row for image_id in image_ids for row in clusters[image_id]]
    sampled: dict[str, list[float]] = {name: [] for name in metrics}
    rng = random.Random(args.seed)

    for _ in range(args.iterations):
        sample_rows: list[dict[str, Any]] = []
        for _ in image_ids:
            sample_rows.extend(clusters[rng.choice(image_ids)])
        for name, metric in metrics.items():
            sampled[name].append(metric(sample_rows))

    report: dict[str, Any] = {
        "cluster_unit": "image",
        "image_clusters": len(image_ids),
        "questions": len(observed_rows),
        "iterations": args.iterations,
        "seed": args.seed,
        "metrics": {},
    }
    for name, metric in metrics.items():
        low, high = percentile_interval(sampled[name])
        report["metrics"][name] = {
            "estimate": metric(observed_rows),
            "ci_95": [low, high],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

