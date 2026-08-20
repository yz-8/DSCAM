# DSCAM / InstaBind-Lite

Official code and benchmark metadata for **InstaBind-Lite: Diagnosing Dense
Same-Class Attribute Misbinding in Large Vision-Language Models**.

InstaBind-Lite is a controlled diagnostic benchmark for testing whether a
large vision-language model assigns a visible attribute to the correct
same-class instance. Version 0.4 contains:

| Item | Count |
|---|---:|
| Images | 524 |
| Same-class groups | 529 |
| Boxed instances | 1,773 |
| Questions | 9,580 |
| L1 / L2 / L3 / L4 | 1,773 / 1,773 / 2,488 / 3,546 |

The release includes the exact annotations, generated questions, model
outputs, and evaluation code used for the reported results. Original source
images are not included in this Git repository; see
[`docs/DATA_CARD.md`](docs/DATA_CARD.md) for access and licensing details.

## Main full-image results

All values are percentages. MBR is measured over all questions; A-MBR is the
adjacent share among identifiable misbindings.

| Model | Accuracy | MBR | Err-MBR | A-MBR | Out-of-set |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 75.89 | 13.65 | 56.62 | 82.65 | 7.07 |
| InternVL3-8B | 76.69 | 14.01 | 60.10 | 84.65 | 7.36 |
| LLaVA-1.5-7B | 54.37 | 34.01 | 74.54 | 78.70 | 7.89 |
| LLaVA-OneVision-7B | 72.30 | 18.58 | 67.07 | 76.97 | 6.88 |
| MiniCPM-V-2.6 | 71.18 | 18.96 | 65.77 | 80.51 | 7.13 |
| Gemini-3.5-Flash | 80.13 | 5.79 | 29.15 | 78.38 | 9.66 |
| Qwen3-VL-Plus | 81.52 | 9.31 | 50.40 | 84.64 | 7.53 |

## What is measured

- **MBR**: fraction of all questions answered with information from another
  visible same-class instance.
- **Err-MBR**: fraction of wrong answers that are identifiable misbindings.
- **A-MBR**: fraction of misbindings originating from an ordinally adjacent
  instance.
- **Distance-MBR**: distribution of source-target ordinal distance.
- **Out-of-set hallucination**: an answer unsupported by any instance in the
  annotated same-class group.

## Repository layout

```text
data/annotations/     InstaBind-Lite v0.4 instance annotations
data/questions/       Exact L1-L4 evaluation questions
model_outputs/        Raw JSONL predictions used in the paper
results/              Summaries, robustness checks, and parser audit
scripts/dataset/      Validation and deterministic question generation
scripts/evaluation/   Metrics, bootstrap, and intervention summaries
scripts/inference/    Model-specific inference adapters
docs/                 Data card, protocol, and licensing guidance
tests/                 Release integrity checks
```

## Quick verification

Only Python 3.10+ is needed to validate the metadata and reproduce the
question file.

```bash
python scripts/dataset/validate_annotations.py \
  data/annotations/instabind_lite_v0.4.json

python scripts/dataset/generate_questions.py \
  data/annotations/instabind_lite_v0.4.json \
  regenerated_questions.jsonl

python tests/check_release.py
```

The expected SHA-256 digests are recorded in `data/CHECKSUMS.sha256`.

## Re-evaluate a model

Model outputs use one JSON object per line:

```json
{"question_id": "...", "prediction": "red", "model": "...", "setting": "full_image"}
```

Example:

```bash
python scripts/evaluation/evaluate_outputs.py \
  data/questions/instabind_lite_v0.4_questions.jsonl \
  model_outputs/qwen25vl/full_image_outputs.jsonl \
  --summary-csv /tmp/qwen_summary.csv \
  --detailed-csv /tmp/qwen_detailed.csv \
  --analysis-json /tmp/qwen_analysis.json
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for interventions,
model adapters, and the exact result directories.

## Data and image access

The annotations contain relative image paths and source identifiers. Images
derived from COCO, GQA, and VAW remain governed by their original datasets.
Web-sourced images remain governed by their original licenses and are not
redistributed here when recoverable provenance is unavailable. Self-shot
images are also kept outside this source-code repository. The annotation and
question files are sufficient to inspect the benchmark schema and reproduce
all non-visual evaluation logic.

## Citation

```bibtex
@misc{xu2026instabindlite,
  title        = {InstaBind-Lite: Diagnosing Dense Same-Class Attribute
                  Misbinding in Large Vision-Language Models},
  author       = {Xu, Yuanzhi and Gao, Qian and Fan, Jun and Ding, Guohui and
                  Yang, Zhenyu and Xiao, Yuteng and Lin, Sixue},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/yz-8/DSCAM}
}
```

## Licenses

- Code: MIT License, see [`LICENSE`](LICENSE).
- Benchmark annotations and generated questions: CC BY 4.0, subject to the
  exclusions in [`DATA_LICENSE.md`](DATA_LICENSE.md).
- Source images and model weights: not covered by either repository license.
