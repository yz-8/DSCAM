# Reproducibility guide

## 1. Validate annotations

```bash
python scripts/dataset/validate_annotations.py \
  data/annotations/instabind_lite_v0.4.json
```

Expected result: 524 images, 529 groups, 1,773 instances, zero errors, and
zero warnings.

## 2. Regenerate questions

```bash
python scripts/dataset/generate_questions.py \
  data/annotations/instabind_lite_v0.4.json \
  regenerated_questions.jsonl
```

Expected result: 9,580 questions with L1/L2/L3/L4 counts of
1,773/1,773/2,488/3,546. The regenerated file is byte-identical to the
released question file under Python 3.10+.

## 3. Restore source images

Place images under `images/` using the relative `image_path` stored in each
annotation record. Source images are not covered by the repository licenses.

## 4. Generate intervention assets

```bash
pip install -r requirements/base.txt
python scripts/evaluation/generate_intervention_assets.py \
  --annotations data/annotations/instabind_lite_v0.4.json \
  --questions data/questions/instabind_lite_v0.4_questions.jsonl
```

Default settings reproduce the paper protocol: 15% crop padding, 50% context
padding, maximum source-image side 1,800 pixels, and L1/L3 selection.

## 5. Run or import predictions

Inference adapters are provided for Qwen2.5-VL, InternVL3, LLaVA-1.5,
LLaVA-OneVision, MiniCPM-V 2.6, Gemini-style APIs, and OpenAI-compatible
vision APIs. Model dependencies should be installed in separate environments
using the corresponding upstream model instructions; their dependency ranges
are not mutually compatible.

All adapters write resumable JSONL files with `question_id`, `prediction`,
`model`, and `setting` fields. API keys must be supplied through environment
variables and must never be committed.

## 6. Evaluate

```bash
python scripts/evaluation/evaluate_outputs.py \
  data/questions/instabind_lite_v0.4_questions.jsonl \
  model_outputs/qwen25vl/full_image_outputs.jsonl \
  --summary-csv /tmp/summary.csv \
  --detailed-csv /tmp/detailed.csv \
  --analysis-json /tmp/analysis.json
```

The checked-in `results/evaluation/` summaries were generated from the raw
JSONL outputs in `model_outputs/`. Statistical and parser-audit artifacts are
under `results/audits/`.

## Decoding protocol

Open-weight adapters use deterministic decoding (`do_sample=False`) and short
answers. API adapters request concise responses and use low-temperature or
deterministic settings when supported. Exact prompt construction is retained
inside each inference adapter.

