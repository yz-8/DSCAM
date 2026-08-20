#!/usr/bin/env python3
"""Prepare an AutoDL-ready export for InstaBind-Lite.

The export rewrites absolute local image paths to relative paths such as
`images/COCO_val2014_000000454661.jpg` and creates a manifest that can be used
to copy the required images before uploading to AutoDL.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def relative_image_path(source_path: str) -> str:
    return str(Path("images") / Path(source_path).name).replace("\\", "/")


def collect_image_paths(annotations: dict[str, Any]) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for image in annotations.get("images", []):
        source_path = image["image_path"]
        target_path = relative_image_path(source_path)
        manifest[image["image_id"]] = {
            "image_id": image["image_id"],
            "source_path": source_path,
            "target_path": target_path,
            "filename": Path(source_path).name,
        }
    return manifest


def rewrite_annotations(annotations: dict[str, Any], manifest: dict[str, dict[str, str]]) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(annotations, ensure_ascii=False))
    for image in rewritten.get("images", []):
        image["image_path"] = manifest[image["image_id"]]["target_path"]
    return rewritten


def rewrite_questions(questions: list[dict[str, Any]], manifest: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for row in questions:
        item = dict(row)
        item["image_path"] = manifest[item["image_id"]]["target_path"]
        rewritten.append(item)
    return rewritten


def write_manifest_csv(path: Path, manifest: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "source_path", "target_path", "filename"])
        writer.writeheader()
        for row in sorted(manifest.values(), key=lambda x: x["image_id"]):
            writer.writerow(row)


def write_manifest_jsonl(path: Path, manifest: dict[str, dict[str, str]]) -> None:
    rows = [manifest[key] for key in sorted(manifest)]
    write_jsonl(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    questions = read_jsonl(args.questions)
    manifest = collect_image_paths(annotations)

    output_data = args.output_dir / "data"
    write_json(output_data / "instabind_lite_annotations.current.autodl.json", rewrite_annotations(annotations, manifest))
    write_jsonl(output_data / "questions.current.autodl.jsonl", rewrite_questions(questions, manifest))
    write_manifest_csv(output_data / "image_manifest.csv", manifest)
    write_manifest_jsonl(output_data / "image_manifest.jsonl", manifest)
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_outputs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    summary = {
        "images": len(manifest),
        "questions": len(questions),
        "output_dir": str(args.output_dir),
        "next_step": "Run scripts/collect_images_local.py from the export directory before uploading to AutoDL.",
    }
    write_json(args.output_dir / "export_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

