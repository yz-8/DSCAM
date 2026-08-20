# InstaBind-Lite v0.4 data card

## Intended use

InstaBind-Lite is a diagnostic evaluation set for studying attribute
misbinding among multiple visible instances of the same class. It is intended
for model evaluation, error analysis, and mitigation research. It is not a
training corpus for identity recognition or demographic inference.

## Composition

- 524 natural images
- 529 same-class groups
- 1,773 boxed instances
- 9,580 questions
- group size: 3 to 6 instances
- attributes: object color and person upper-clothing color
- spatial order: left to right

Image sources in the annotation metadata:

| Source | Images |
|---|---:|
| ManualWeb | 277 |
| COCO | 205 |
| GQA | 21 |
| SelfShot | 17 |
| VAW | 4 |

## Annotation schema

Each image record contains an image identifier, source metadata, dimensions,
and one or more same-class groups. Each group contains:

- class name and queried attribute;
- left-to-right instance order;
- `bbox_xywh` in source-image pixels;
- canonical attribute values;
- left and right neighbor identifiers;
- visibility and quality-control fields.

Questions retain a compact `binding_context` so that a wrong answer can be
traced to a competing source instance without an LLM judge.

## Quality control

The final release passes `validate_annotations.py` with zero errors and zero
warnings. All 524 images are approved, all boxes are in bounds, neighbor links
are consistent, and all 529 groups have unique queried attributes for L2.
Question generation is deterministic: regenerating from the released
annotations produces the exact 9,580-line question file recorded in
`data/CHECKSUMS.sha256`.

## Image availability

Original images are not bundled in this Git repository. COCO, GQA, and VAW
images should be obtained from their official distributions and matched using
the source identifiers in the annotation file. ManualWeb records originated
primarily from Pexels, but files without recoverable page-level provenance are
not redistributed here. Self-shot images may be distributed separately by
the authors in a future archival data release.

## People, privacy, and sensitive use

Person questions concern visible upper-clothing color only. The benchmark
does not contain identity, demographic, biometric, or sensitive-attribute
questions. It should not be used for surveillance, identity inference, or
demographic profiling.

## Known limitations

The benchmark is intentionally compact and high-purity. It is class
imbalanced, focuses on color-like attributes, uses still images, and models
ordinal left-to-right structure rather than depth or temporal identity.

