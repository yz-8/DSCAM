#!/usr/bin/env python3
"""Check the public InstaBind-Lite release without third-party dependencies."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data/annotations/instabind_lite_v0.4.json"
QUESTIONS = ROOT / "data/questions/instabind_lite_v0.4_questions.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    with ANNOTATIONS.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    with QUESTIONS.open("r", encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle if line.strip()]

    images = annotations["images"]
    groups = [group for image in images for group in image["same_class_groups"]]
    instances = [instance for group in groups for instance in group["instances"]]

    assert len(images) == 524
    assert len(groups) == 529
    assert len(instances) == 1773
    assert len(questions) == 9580
    assert len({row["question_id"] for row in questions}) == 9580
    assert Counter(row["level"] for row in questions) == {
        "L1": 1773,
        "L2": 1773,
        "L3": 2488,
        "L4": 3546,
    }
    assert all(not Path(image["image_path"]).is_absolute() for image in images)
    assert all(not Path(row["image_path"]).is_absolute() for row in questions)

    print("Release integrity checks passed.")
    print(f"annotations sha256: {sha256(ANNOTATIONS)}")
    print(f"questions sha256:   {sha256(QUESTIONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

