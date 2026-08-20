# Bootstrap Significance Summary

Bootstrap seed: 20260712. Confidence interval: percentile 95%.

## Full Image Metric CIs

| model | setting | metric | observed_pct | ci95_low_pct | ci95_high_pct | n | bootstrap_iterations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-7B | full_image | accuracy | 70.34 | 68.98 | 71.68 | 4852 | 1000 |
| Qwen2.5-VL-7B | full_image | overall_mbr | 17.35 | 16.30 | 18.43 | 4852 | 1000 |
| Qwen2.5-VL-7B | full_image | a_mbr | 83.97 | 81.61 | 86.49 | 4852 | 1000 |
| InternVL3-8B | full_image | accuracy | 69.00 | 67.62 | 70.36 | 4852 | 1000 |
| InternVL3-8B | full_image | overall_mbr | 20.01 | 18.90 | 21.08 | 4852 | 1000 |
| InternVL3-8B | full_image | a_mbr | 82.90 | 80.55 | 85.14 | 4852 | 1000 |
| LLaVA-1.5-7B | full_image | accuracy | 51.59 | 50.16 | 53.01 | 4852 | 1000 |
| LLaVA-1.5-7B | full_image | overall_mbr | 33.22 | 31.82 | 34.56 | 4852 | 1000 |
| LLaVA-1.5-7B | full_image | a_mbr | 79.03 | 77.13 | 81.14 | 4852 | 1000 |

## Intervention Gap CIs

| model | comparison | metric | observed_gap_pct | ci95_low_pct | ci95_high_pct | n_pairs | bootstrap_iterations | sign_consistent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-7B | box_guided - full_l1_l3 | accuracy_gap | -0.09 | -1.55 | 1.22 | 2134 | 1000 | False |
| Qwen2.5-VL-7B | box_guided - full_l1_l3 | mbr_gap | -1.08 | -2.30 | 0.14 | 2134 | 1000 | False |
| Qwen2.5-VL-7B | crop_oracle - full_l1_l3 | accuracy_gap | 4.78 | 2.86 | 6.89 | 2134 | 1000 | True |
| Qwen2.5-VL-7B | crop_oracle - full_l1_l3 | mbr_gap | -6.84 | -8.58 | -5.15 | 2134 | 1000 | True |
| Qwen2.5-VL-7B | context_crop - full_l1_l3 | accuracy_gap | 5.95 | 3.98 | 7.78 | 2134 | 1000 | True |
| Qwen2.5-VL-7B | context_crop - full_l1_l3 | mbr_gap | -6.79 | -8.67 | -4.92 | 2134 | 1000 | True |
| Qwen2.5-VL-7B | dim_non_target - full_l1_l3 | accuracy_gap | -5.34 | -7.08 | -3.61 | 2134 | 1000 | True |
| Qwen2.5-VL-7B | dim_non_target - full_l1_l3 | mbr_gap | 2.11 | 0.56 | 3.66 | 2134 | 1000 | True |
| InternVL3-8B | box_guided - full_l1_l3 | accuracy_gap | 7.31 | 5.62 | 8.90 | 2134 | 1000 | True |
| InternVL3-8B | box_guided - full_l1_l3 | mbr_gap | -7.22 | -8.72 | -5.76 | 2134 | 1000 | True |
| InternVL3-8B | crop_oracle - full_l1_l3 | accuracy_gap | 6.04 | 4.03 | 7.92 | 2134 | 1000 | True |
| InternVL3-8B | crop_oracle - full_l1_l3 | mbr_gap | -8.95 | -10.73 | -7.12 | 2134 | 1000 | True |
| InternVL3-8B | context_crop - full_l1_l3 | accuracy_gap | 5.76 | 3.89 | 7.83 | 2134 | 1000 | True |
| InternVL3-8B | context_crop - full_l1_l3 | mbr_gap | -6.94 | -8.86 | -5.11 | 2134 | 1000 | True |
| InternVL3-8B | dim_non_target - full_l1_l3 | accuracy_gap | 3.66 | 2.01 | 5.30 | 2134 | 1000 | True |
| InternVL3-8B | dim_non_target - full_l1_l3 | mbr_gap | -5.48 | -7.08 | -3.98 | 2134 | 1000 | True |
| LLaVA-1.5-7B | box_guided - full_l1_l3 | accuracy_gap | 0.33 | -1.27 | 1.78 | 2134 | 1000 | False |
| LLaVA-1.5-7B | box_guided - full_l1_l3 | mbr_gap | 1.41 | -0.28 | 2.86 | 2134 | 1000 | False |
| LLaVA-1.5-7B | crop_oracle - full_l1_l3 | accuracy_gap | 21.98 | 19.63 | 24.09 | 2134 | 1000 | True |
| LLaVA-1.5-7B | crop_oracle - full_l1_l3 | mbr_gap | -21.27 | -23.43 | -19.12 | 2134 | 1000 | True |
| LLaVA-1.5-7B | context_crop - full_l1_l3 | accuracy_gap | 20.71 | 18.37 | 22.91 | 2134 | 1000 | True |
| LLaVA-1.5-7B | context_crop - full_l1_l3 | mbr_gap | -19.40 | -21.60 | -17.29 | 2134 | 1000 | True |
| LLaVA-1.5-7B | dim_non_target - full_l1_l3 | accuracy_gap | -0.94 | -2.58 | 0.84 | 2134 | 1000 | False |
| LLaVA-1.5-7B | dim_non_target - full_l1_l3 | mbr_gap | 1.55 | -0.19 | 3.37 | 2134 | 1000 | False |

## Instance-First Gap CIs

| model | comparison | metric | observed_gap_pct | ci95_low_pct | ci95_high_pct | n_pairs | bootstrap_iterations | sign_consistent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-7B | instance_first - full_image | accuracy_gap | -2.74 | -3.79 | -1.73 | 4852 | 1000 | True |
| Qwen2.5-VL-7B | instance_first - full_image | mbr_gap | -1.30 | -2.18 | -0.37 | 4852 | 1000 | True |
| InternVL3-8B | instance_first - full_image | accuracy_gap | 2.25 | 1.09 | 3.54 | 4852 | 1000 | True |
| InternVL3-8B | instance_first - full_image | mbr_gap | -5.54 | -6.60 | -4.49 | 4852 | 1000 | True |
| LLaVA-1.5-7B | instance_first - full_image | accuracy_gap | -22.09 | -23.66 | -20.38 | 4852 | 1000 | True |
| LLaVA-1.5-7B | instance_first - full_image | mbr_gap | 1.53 | -0.04 | 3.07 | 4852 | 1000 | False |

Interpretation: `sign_consistent=True` means the 95% bootstrap CI does not cross 0.