#!/usr/bin/env python3
"""Evaluate model outputs on InstaBind-Lite questions.

Expected model-output JSONL rows:
  {"question_id": "...", "prediction": "...", "model": "model-name", "setting": "full_image"}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PUNCT_RE = re.compile(r"^[\s\"'`.,:;!?()\[\]{}]+|[\s\"'`.,:;!?()\[\]{}]+$")
FINAL_ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*[:：]\s*([^\r\n]+)", re.IGNORECASE)
COLOR_ALIASES = {
    "grey": "gray",
    "dark grey": "gray",
    "light grey": "gray",
    "dark gray": "gray",
    "light gray": "gray",
    "navy": "blue",
    "dark blue": "blue",
    "light blue": "blue",
    "dark green": "green",
    "light green": "green",
    "maroon": "red",
    "burgundy": "red",
    "yello": "yellow",
    "tan": "beige",
    "cream": "beige",
    "clear": "transparent",
    "see through": "transparent",
    "see-through": "transparent",
    "colorless": "transparent",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
    return rows


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_final_answer(prediction: Any) -> Any:
    if prediction is None:
        return prediction
    text = str(prediction).strip()
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return prediction
    return matches[-1].group(1).strip()


def canonical_color(text: str) -> str:
    return COLOR_ALIASES.get(text, text)


def phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def extract_unique_candidate(pred: str, question: dict[str, Any]) -> str | None:
    if question.get("answer_type") == "attribute":
        raw_values = [normalize_text(inst.get("attribute_value")) for inst in context_instances(question)]
        candidates = sorted({canonical_color(value) for value in raw_values if value}, key=len, reverse=True)
    elif question.get("answer_type") == "position":
        raw_values = [normalize_text(inst.get("order_label")) for inst in context_instances(question)]
        candidates = sorted({value for value in raw_values if value}, key=len, reverse=True)
    else:
        return None

    hits = [candidate for candidate in candidates if phrase_in_text(candidate, pred)]
    unique_hits = sorted(set(hits))
    if len(unique_hits) == 1:
        return unique_hits[0]
    return None


def normalize_prediction(prediction: Any, question: dict[str, Any]) -> str:
    pred = normalize_text(extract_final_answer(prediction))
    if not pred:
        return ""

    options = question.get("answer_options", [])
    if options:
        for opt in options:
            label = normalize_text(opt["label"])
            text = normalize_text(opt["text"])
            if pred == label or pred.startswith(label + ".") or pred.startswith(label + " "):
                return text
            if text in pred:
                return text

    yes = {"yes", "true", "correct"}
    no = {"no", "false", "incorrect"}
    if question.get("answer_type") == "yes_no":
        if pred in yes or pred.startswith("yes"):
            return "yes"
        if pred in no or pred.startswith("no"):
            return "no"
        if phrase_in_text("yes", pred) and not phrase_in_text("no", pred):
            return "yes"
        if phrase_in_text("no", pred) and not phrase_in_text("yes", pred):
            return "no"

    if " and " in pred or "&" in pred or "+" in pred:
        return "multicolor"
    pred = canonical_color(pred)
    unique_candidate = extract_unique_candidate(pred, question)
    if unique_candidate is not None:
        return unique_candidate
    return pred


def context_instances(question: dict[str, Any]) -> list[dict[str, Any]]:
    return question.get("binding_context", {}).get("instances", [])


def instance_by_id(question: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {inst["instance_id"]: inst for inst in context_instances(question)}


def distance(question: dict[str, Any], source_id: str) -> int | None:
    id_to_inst = instance_by_id(question)
    target_id = question.get("target_instance_id")
    if target_id not in id_to_inst or source_id not in id_to_inst:
        return None
    return abs(id_to_inst[target_id]["order_index"] - id_to_inst[source_id]["order_index"])


def adjacent(question: dict[str, Any], source_id: str) -> bool:
    return distance(question, source_id) == 1


def classify_error(question: dict[str, Any], pred: str) -> tuple[str, str | None, int | None]:
    if not pred or pred in {"unknown", "unclear", "not sure", "cannot determine"}:
        return "invalid_or_unparseable", None, None

    target_id = question.get("target_instance_id")
    instances = context_instances(question)

    if question.get("answer_type") == "attribute":
        for inst in instances:
            if inst["instance_id"] == target_id:
                continue
            if normalize_text(inst.get("attribute_value")) == pred:
                source_id = inst["instance_id"]
                dist = distance(question, source_id)
                return ("adjacent_misbinding" if dist == 1 else "non_adjacent_misbinding"), source_id, dist
        values = {normalize_text(inst.get("attribute_value")) for inst in instances}
        if pred not in values:
            return "out_of_set_hallucination", None, None
        return "attribute_recognition_failure", None, None

    if question.get("answer_type") == "position":
        for inst in instances:
            if inst["instance_id"] == target_id:
                continue
            if normalize_text(inst.get("order_label")) == pred:
                source_id = inst["instance_id"]
                dist = distance(question, source_id)
                return ("adjacent_misbinding" if dist == 1 else "non_adjacent_misbinding"), source_id, dist
        positions = {normalize_text(inst.get("order_label")) for inst in instances}
        if pred not in positions:
            return "out_of_set_hallucination", None, None
        return "attribute_recognition_failure", None, None

    if question.get("answer_type") == "yes_no":
        answer = normalize_text(question.get("canonical_answer"))
        source_id = question.get("proposed_attribute_source_instance_id")
        if answer == "no" and pred == "yes" and source_id and source_id != target_id:
            dist = distance(question, source_id)
            return ("adjacent_misbinding" if dist == 1 else "non_adjacent_misbinding"), source_id, dist
        if pred not in {"yes", "no"}:
            return "invalid_or_unparseable", None, None
        return "attribute_recognition_failure", None, None

    return "invalid_or_unparseable", None, None


def evaluate(questions: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    q_by_id = {q["question_id"]: q for q in questions}
    detailed: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    distance_counts: Counter = Counter()

    for out in outputs:
        qid = out.get("question_id")
        if qid not in q_by_id:
            # Evaluation question files may be subsets, e.g. L1/L3 intervention
            # questions evaluated against a full-image output file containing
            # L2/L4 rows as well. Ignore outputs outside the active question set.
            continue

        q = q_by_id[qid]
        model = out.get("model", "unknown")
        setting = out.get("setting", "full_image")
        pred = normalize_prediction(out.get("prediction"), q)
        gold = normalize_prediction(q.get("canonical_answer"), q)
        correct = pred == gold
        error_type = "correct"
        source_id = None
        dist = None
        if not correct:
            error_type, source_id, dist = classify_error(q, pred)
            if "misbinding" in error_type and dist is not None:
                distance_counts[str(dist)] += 1

        key = (model, setting, q["level"])
        buckets[key]["total"] += 1
        buckets[key]["correct_count"] += int(correct)
        buckets[key][error_type] += 1
        if error_type in {"adjacent_misbinding", "non_adjacent_misbinding"}:
            buckets[key]["misbinding"] += 1

        target_order = q.get("binding_context", {}).get("target_order_index")
        if target_order and source_id:
            source_order = instance_by_id(q).get(source_id, {}).get("order_index")
            if source_order:
                cm_key = f"{model}|{setting}|{q['level']}|n={q.get('binding_context', {}).get('group_size')}"
                confusion[cm_key][f"{target_order}->{source_order}"] += 1

        detailed.append(
            {
                "question_id": qid,
                "model": model,
                "setting": setting,
                "level": q["level"],
                "class_name": q["class_name"],
                "answer_type": q["answer_type"],
                "gold": gold,
                "prediction_normalized": pred,
                "correct": correct,
                "error_type": error_type,
                "misbinding_source_instance_id": source_id,
                "misbinding_distance": dist,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for (model, setting, level), c in sorted(buckets.items()):
        total = c["total"]
        wrong = total - c["correct_count"]
        misbinding = c["misbinding"]
        adjacent_misbinding = c["adjacent_misbinding"]
        summary_rows.append(
            {
                "model": model,
                "setting": setting,
                "level": level,
                "total": total,
                "accuracy": c["correct_count"] / total if total else 0.0,
                "overall_mbr": misbinding / total if total else 0.0,
                "error_conditioned_mbr": misbinding / wrong if wrong else 0.0,
                "a_mbr": adjacent_misbinding / misbinding if misbinding else 0.0,
                "out_of_set_rate": c["out_of_set_hallucination"] / total if total else 0.0,
                "invalid_rate": c["invalid_or_unparseable"] / total if total else 0.0,
                "wrong": wrong,
            }
        )

    analysis = {
        "distance_mbr_counts": dict(distance_counts),
        "misbinding_confusion_counts": {key: dict(value) for key, value in confusion.items()},
    }
    return detailed, summary_rows, analysis


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
    parser.add_argument("questions", type=Path)
    parser.add_argument("model_outputs", type=Path)
    parser.add_argument("--summary-csv", type=Path, default=Path("data/evaluation/evaluation_summary.csv"))
    parser.add_argument("--detailed-csv", type=Path, default=Path("data/evaluation/evaluation_detailed.csv"))
    parser.add_argument("--analysis-json", type=Path, default=Path("data/evaluation/evaluation_analysis.json"))
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    outputs = read_jsonl(args.model_outputs)
    detailed, summary_rows, analysis = evaluate(questions, outputs)

    write_csv(args.summary_csv, summary_rows)
    write_csv(args.detailed_csv, detailed)
    args.analysis_json.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"evaluated_outputs": len(outputs), "summary_rows": len(summary_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
