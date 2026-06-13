# Ablation Results Table

Main paper should use method names. Implementation identifiers and exact commands stay here or in the supplement.

RainMix is not treated as an independent paper method. It supplies the auxiliary negative domain for the triplet objective, so the main ablation table uses a coupled RainMix-triplet row.

| ID | Run name | Method | Epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Evidence path | Status | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| A1 | `sanity_pr1_baseline` | Source-only YOLO-G detector | 50 | 0.37841 | 0.24214 | 0.74702 | 0.32365 | `/home/kacchan/VuHung/project/nhandang_project_2/runs/train/sanity_pr1_baseline/results.csv` | found | name says sanity but run is 50 epochs per `results.csv` |
| A2 | `city_foggycity_advgrl_faithful` | YOLO-G image-level domain adaptation | 50 | 0.41195 | 0.25884 | 0.77726 | 0.35936 | `/home/kacchan/VuHung/project/nhandang_project_2/runs/train/city_foggycity_advgrl_faithful/results.csv` | found | original YOLO-G image-level DA baseline |
| A3 | `city_foggycity_advgrl_faithful_alpha080` | Adaptive gradient reversal | 50 | 0.39839 | 0.25088 | 0.67457 | 0.35689 | `/home/kacchan/VuHung/project/nhandang_project_2/runs/train/city_foggycity_advgrl_faithful_alpha080/results.csv` | found | SPPF faithful AdvGRL, alpha 0.80 |
| A4 | `city_foggycity_advgrl_faithful_triplet_50ep` | RainMix-coupled triplet adaptation | 50 | 0.31183 | 0.18733 | 0.73212 | 0.27307 | `/home/kacchan/VuHung/project/nhandang_project_2/runs/train/city_foggycity_advgrl_faithful_triplet_50ep/results.csv` | found | fixed faithful adaptation + RainMix auxiliary domain + triplet objective, without adaptive reversal |
| A5 | `city_foggycity_advgrl_faithful_full_12` | Adaptive gradient reversal with RainMix-coupled triplet adaptation | 50 | 0.34015 | 0.21109 | 0.74506 | 0.28123 | `/home/kacchan/VuHung/project/nhandang_project_2/runs/train/city_foggycity_advgrl_faithful_full_12/results.csv` | found | adaptive reversal + RainMix auxiliary domain + triplet objective, alpha 0.75 |
| A6 | `city_foggycity_advgrl_faithful_neck_all_alpha075_50ep` | Adaptive gradient reversal with multi-scale neck adaptation | 50 | 0.46665 | 0.28955 | 0.72261 | 0.41646 | `/home/kacchan/VuHung/project/nhandang_project_2/runs/train/city_foggycity_advgrl_faithful_neck_all_alpha075_50ep/results.csv` | found | best verified mAP@0.5 and mAP@0.5:0.95 among found runs |
| A7 | `city_foggycity_advgrl_faithful_neck_all_alpha075_objgate_50ep` | Adaptive gradient reversal with foreground-gated multi-scale neck adaptation | 50 | 0.44293 | 0.27584 | 0.74389 | 0.38448 | `/home/kacchan/VuHung/project/nhandang_project_2/runs/train/city_foggycity_advgrl_faithful_neck_all_alpha075_objgate_50ep/results.csv` | found | optional foreground-aware extension |

## Additional found runs

| Run name | Method variant | Epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `city_foggycity_advgrl_faithful_neck_all_alpha065_50ep` | neck-all adaptive reversal, alpha 0.65 | 50 | 0.45005 | 0.27442 | 0.76292 | 0.37925 | alpha sweep |
| `city_foggycity_advgrl_faithful_neck_all_alpha070_50ep` | neck-all adaptive reversal, alpha 0.70 | 50 | 0.44393 | 0.27207 | 0.76334 | 0.37474 | alpha sweep |
| `city_foggycity_advgrl_faithful_neck_all_alpha070_grl015_50ep` | neck-all adaptive reversal, alpha 0.70, higher GRL base | 50 | 0.45068 | 0.27661 | 0.76088 | 0.39070 | GRL-weight sweep |
| `city_foggycity_advgrl_faithful_neck_all_alpha075_grl005_50ep` | neck-all adaptive reversal, alpha 0.75, lower GRL base | 50 | 0.43972 | 0.27214 | 0.74219 | 0.38413 | GRL-weight sweep |
| `city_foggycity_advgrl_faithful_neck_all_alpha075_grl015_50ep` | neck-all adaptive reversal, alpha 0.75, higher GRL base | 50 | 0.44433 | 0.27154 | 0.77278 | 0.37461 | GRL-weight sweep |
| `city_foggycity_advgrl_faithful_neck_all_alpha080_50ep` | neck-all adaptive reversal, alpha 0.80 | 50 | 0.43509 | 0.26726 | 0.74484 | 0.37125 | alpha sweep |
| `city_foggycity_advgrl_faithful_neck_all_alpha085_50ep` | neck-all adaptive reversal, alpha 0.85 | 50 | 0.44404 | 0.27052 | 0.74711 | 0.38796 | alpha sweep |
| `city_foggycity_advgrl_faithful_neck_p4_alpha075_50ep` | neck-P4 adaptive reversal, alpha 0.75 | 50 | 0.44931 | 0.26906 | 0.73591 | 0.40150 | single-neck comparison |

## Missing experiments

Required for the coupled RainMix/triplet ablation story:

None currently identified from the checked run set. The isolated RainMix-coupled triplet run has been found and entered above.

Optional depending on final report scope:

1. Canonical adaptive gradient reversal at alpha 0.75 on SPPF features, if alpha 0.80 should not be treated as the main A3 run.
2. Fixed neck-all adaptation without adaptive reversal, if comparing neck-all adaptive reversal fairly against fixed neck-all adaptation.
3. Foreground-gated neck-all fixed adaptation without adaptive reversal, if isolating objectness gating from adaptive reversal.

Not required as a main paper row:

- RainMix-only adaptation without triplet learning. RainMix has methodological meaning here as the triplet objective's auxiliary domain, not as a standalone loss component.

## Evidence rules

- Do not enter a metric unless the source file exists.
- Prefer final validation metrics over memory/manual notes.
- Keep failure runs in the table with crash/NaN notes instead of hiding them.
- For adaptation runs, also archive the diagnostic CSV and summarize adaptive strength, classifier loss, image-level adaptation loss, and triplet loss in a separate diagnostic table.
