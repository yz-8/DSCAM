#!/usr/bin/env python3
"""Generate L1-L4 InstaBind-Lite questions from cleaned annotations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PERSON_CLASS = "person"
OBJECT_CLASSES = {"car", "chair", "cup", "bottle", "bag", "umbrella", "animal"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def sorted_instances(group: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(group["instances"], key=lambda inst: inst["order_index"])


def attribute_key_for(group: dict[str, Any]) -> str:
    return group["attribute_focus"]


def class_phrase(class_name: str) -> str:
    if class_name == "person":
        return "person"
    return class_name


def person_clothing_phrase(attr_key: str) -> str:
    if attr_key == "lower_clothing_color":
        return "lower clothing"
    return "upper clothing"


def get_attr(inst: dict[str, Any], attr_key: str) -> str:
    return inst["attributes"][attr_key]


def context_for(group: dict[str, Any], attr_key: str, target: dict[str, Any]) -> dict[str, Any]:
    instances = sorted_instances(group)
    return {
        "group_id": group["group_id"],
        "class_name": group["class_name"],
        "spatial_axis": group.get("spatial_axis", "left_to_right"),
        "group_size": len(instances),
        "target_instance_id": target["instance_id"],
        "target_order_index": target["order_index"],
        "target_order_label": target["order_label"],
        "queried_attribute_key": attr_key,
        "instances": [
            {
                "instance_id": inst["instance_id"],
                "order_index": inst["order_index"],
                "order_label": inst["order_label"],
                "attribute_value": get_attr(inst, attr_key),
                "neighbors": inst.get("neighbors", {}),
            }
            for inst in instances
        ],
    }


def options_for(instances: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [
        {"label": letters[i], "text": inst["order_label"], "is_correct": inst["order_label"] == answer}
        for i, inst in enumerate(instances)
    ]


def format_options(options: list[dict[str, Any]]) -> str:
    return "\n".join(f"{opt['label']}. {opt['text']}" for opt in options)


def base_record(
    *,
    question_id: str,
    image: dict[str, Any],
    group: dict[str, Any],
    level: str,
    question_type: str,
    question: str,
    answer_type: str,
    answer: str,
    attr_key: str,
    target: dict[str, Any],
    options: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "question_id": question_id,
        "image_id": image["image_id"],
        "image_path": image["image_path"],
        "group_id": group["group_id"],
        "class_name": group["class_name"],
        "level": level,
        "question_type": question_type,
        "question": question,
        "answer_type": answer_type,
        "canonical_answer": answer,
        "attribute_key": attr_key,
        "target_instance_id": target["instance_id"],
        "binding_context": context_for(group, attr_key, target),
    }
    if options:
        record["answer_options"] = options
    if extra:
        record.update(extra)
    return record


def generate_l1(image: dict[str, Any], group: dict[str, Any]) -> list[dict[str, Any]]:
    attr_key = attribute_key_for(group)
    class_name = group["class_name"]
    rows = []
    for inst in sorted_instances(group):
        if class_name == PERSON_CLASS and attr_key == "upper_clothing_color":
            question = f"What color is the upper clothing of the {inst['order_label']} person?"
        elif class_name == PERSON_CLASS and attr_key == "lower_clothing_color":
            question = f"What color is the lower clothing of the {inst['order_label']} person?"
        else:
            question = f"What color is the {inst['order_label']} {class_phrase(class_name)}?"
        qid = slug(f"{image['image_id']}_{group['group_id']}_l1_{inst['instance_id']}_{attr_key}")
        rows.append(
            base_record(
                question_id=qid,
                image=image,
                group=group,
                level="L1",
                question_type="position_to_attribute",
                question=question,
                answer_type="attribute",
                answer=get_attr(inst, attr_key),
                attr_key=attr_key,
                target=inst,
            )
        )
    return rows


def generate_l2(image: dict[str, Any], group: dict[str, Any]) -> list[dict[str, Any]]:
    attr_key = attribute_key_for(group)
    class_name = group["class_name"]
    instances = sorted_instances(group)
    counts = Counter(get_attr(inst, attr_key) for inst in instances)
    rows = []
    for inst in instances:
        value = get_attr(inst, attr_key)
        if counts[value] != 1:
            continue
        if class_name == PERSON_CLASS:
            phrase = person_clothing_phrase(attr_key)
            question = f"Where is the person wearing {value} {phrase}?"
        else:
            question = f"Where is the {value} {class_phrase(class_name)}?"
        answer = inst["order_label"]
        options = options_for(instances, answer)
        full_question = question + "\n" + format_options(options)
        qid = slug(f"{image['image_id']}_{group['group_id']}_l2_{inst['instance_id']}_{attr_key}_{value}")
        rows.append(
            base_record(
                question_id=qid,
                image=image,
                group=group,
                level="L2",
                question_type="attribute_to_position",
                question=full_question,
                answer_type="position",
                answer=answer,
                attr_key=attr_key,
                target=inst,
                options=options,
            )
        )
    return rows


def generate_l3(image: dict[str, Any], group: dict[str, Any]) -> list[dict[str, Any]]:
    attr_key = attribute_key_for(group)
    class_name = group["class_name"]
    instances = sorted_instances(group)
    id_to_inst = {inst["instance_id"]: inst for inst in instances}
    counts = Counter(get_attr(inst, attr_key) for inst in instances)
    rows = []
    for anchor in instances:
        anchor_value = get_attr(anchor, attr_key)
        if counts[anchor_value] != 1:
            continue
        for relation in ("left", "right"):
            target_id = anchor.get("neighbors", {}).get(relation)
            if not target_id:
                continue
            target = id_to_inst[target_id]
            if class_name == PERSON_CLASS:
                phrase = person_clothing_phrase(attr_key)
                question = (
                    f"What color is the {phrase} of the person immediately to the {relation} "
                    f"of the person wearing {anchor_value} {phrase}?"
                )
            else:
                question = (
                    f"What color is the {class_phrase(class_name)} immediately to the {relation} "
                    f"of the {anchor_value} {class_phrase(class_name)}?"
                )
            qid = slug(
                f"{image['image_id']}_{group['group_id']}_l3_{anchor['instance_id']}_{relation}_{target['instance_id']}"
            )
            rows.append(
                base_record(
                    question_id=qid,
                    image=image,
                    group=group,
                    level="L3",
                    question_type="relation_interference",
                    question=question,
                    answer_type="attribute",
                    answer=get_attr(target, attr_key),
                    attr_key=attr_key,
                    target=target,
                    extra={
                        "anchor_instance_id": anchor["instance_id"],
                        "anchor_attribute_value": anchor_value,
                        "relation": relation,
                    },
                )
            )
    return rows


def choose_false_attribute(inst: dict[str, Any], id_to_inst: dict[str, dict[str, Any]], attr_key: str) -> tuple[str, str] | None:
    true_value = get_attr(inst, attr_key)
    neighbor_ids = [inst.get("neighbors", {}).get("left"), inst.get("neighbors", {}).get("right")]
    for neighbor_id in neighbor_ids:
        if neighbor_id and neighbor_id in id_to_inst:
            neighbor_value = get_attr(id_to_inst[neighbor_id], attr_key)
            if neighbor_value != true_value:
                return neighbor_value, neighbor_id
    for other_id, other in id_to_inst.items():
        if other_id == inst["instance_id"]:
            continue
        other_value = get_attr(other, attr_key)
        if other_value != true_value:
            return other_value, other_id
    return None


def generate_l4(image: dict[str, Any], group: dict[str, Any]) -> list[dict[str, Any]]:
    attr_key = attribute_key_for(group)
    class_name = group["class_name"]
    instances = sorted_instances(group)
    id_to_inst = {inst["instance_id"]: inst for inst in instances}
    rows = []
    for inst in instances:
        true_value = get_attr(inst, attr_key)
        prompts = [(true_value, "yes", inst["instance_id"])]
        false_choice = choose_false_attribute(inst, id_to_inst, attr_key)
        if false_choice:
            false_value, source_id = false_choice
            prompts.append((false_value, "no", source_id))

        for value, answer, source_id in prompts:
            if class_name == PERSON_CLASS:
                phrase = person_clothing_phrase(attr_key)
                question = f"Is the {phrase} of the {inst['order_label']} person {value}?"
            else:
                question = f"Is the {inst['order_label']} {class_phrase(class_name)} {value}?"
            qid = slug(
                f"{image['image_id']}_{group['group_id']}_l4_{inst['instance_id']}_{attr_key}_{value}_{answer}"
            )
            rows.append(
                base_record(
                    question_id=qid,
                    image=image,
                    group=group,
                    level="L4",
                    question_type="instance_verification",
                    question=question,
                    answer_type="yes_no",
                    answer=answer,
                    attr_key=attr_key,
                    target=inst,
                    extra={
                        "proposed_attribute_value": value,
                        "proposed_attribute_source_instance_id": source_id,
                    },
                )
            )
    return rows


def generate(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in data.get("images", []):
        if image.get("quality", {}).get("approved") is not True:
            continue
        for group in image.get("same_class_groups", []):
            if any(inst.get("ambiguity_flag") for inst in group.get("instances", [])):
                continue
            rows.extend(generate_l1(image, group))
            rows.extend(generate_l2(image, group))
            rows.extend(generate_l3(image, group))
            rows.extend(generate_l4(image, group))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    data = load_json(args.annotations)
    rows = generate(data)
    write_jsonl(args.output, rows)

    summary = {
        "question_count": len(rows),
        "by_level": dict(Counter(row["level"] for row in rows)),
        "by_class": dict(Counter(row["class_name"] for row in rows)),
        "by_answer_type": dict(Counter(row["answer_type"] for row in rows)),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
