#!/usr/bin/env python3
"""Generate Instance-First Prompt questions for InstaBind-Lite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plural_class_name(class_name: str) -> str:
    if class_name == "person":
        return "people"
    if class_name.endswith("s"):
        return class_name
    return class_name + "s"


def attribute_phrase(row: dict[str, Any]) -> str:
    class_name = row.get("class_name")
    attr_key = row.get("attribute_key")
    if class_name == "person":
        if attr_key == "lower_clothing_color":
            return "lower clothing color"
        return "upper clothing color"
    if attr_key == "material":
        return "material"
    return "color"


def build_instance_first_prompt(row: dict[str, Any]) -> str:
    cls = row["class_name"]
    cls_plural = plural_class_name(cls)
    attr = attribute_phrase(row)
    original_question = row["question"].strip()
    return (
        "You are answering a visual binding question about multiple same-class instances.\n"
        f"Step 1: Identify all {cls_plural} from left to right and list each one's {attr}.\n"
        "Step 2: Use that ordered list to answer the final question.\n"
        f"Final question: {original_question}\n"
        "At the end, write exactly one line in this format:\n"
        "Final answer: <short answer>"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/questions/instabind_lite_v0.4_questions.jsonl",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/questions.instance_first.jsonl")
    parser.add_argument("--levels", nargs="*", default=["L1", "L2", "L3", "L4"])
    args = parser.parse_args()

    keep_levels = set(args.levels)
    rows = []
    for row in read_jsonl(args.questions):
        if row.get("level") not in keep_levels:
            continue
        out = dict(row)
        out["question"] = build_instance_first_prompt(row)
        out["intervention"] = "instance_first_prompt"
        rows.append(out)

    write_jsonl(args.output, rows)
    print(json.dumps({"instance_first_questions": len(rows), "levels": args.levels}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
