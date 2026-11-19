# A6P3 code - Predictive process mining with a TCN on robotic work-order logs

Run everything (preprocessing, prefix extraction, proposed TCN, baselines, ablation, sensitivity,
temporal hold-out, significance tests, bottleneck mining, figures):

    pip install -r requirements.txt      # plus Graphviz `dot` on the PATH
    python3 run_all.py                   # ~35 min on 1 CPU thread; QUICK=1 for a 4-min smoke test
    FIGS_ONLY=1 python3 run_all.py       # redraw all figures in ~15 s from results/raw_checkpoint.pkl

`run_all.py` reads `../data/A6P3_workorder_cases.csv` and `../data/A6P3_workorder_event_log.csv` and writes
to `../results/` (CSV tables, metrics.json, JPG figures) and `../figures/` (300-dpi PNG figures).
All seeds are fixed (base seed 2026); torch and LightGBM run with one thread.

## Protocol (no label leakage)
* A running work order is observed at checkpoint tau = rho * due_in_hours (rho in 0.25/0.5/0.75 for evaluation;
  0.1...0.9 for training). Only events *completed* by tau and the activity *in progress* at tau are visible.
  Orders already finished at tau are skipped. `n_events`, `n_rework_loops`, `any_robot_fault`,
  `cycle_time_hours`, `late_completion` are never used as features (targets: remaining time = cycle - tau, late flag).
* Run-time context known at tau: plant / cell WIP (orders released and not yet finished), time of week, slack.
* 5-fold CV grouped by work order; 15 % of training orders are a validation set (early stopping, F1 threshold).

## Outputs
| file | content |
|---|---|
| results/table_dataset_summary.csv | dataset summary (Table I) |
| results/table_main_comparison.csv, main_per_fold.csv | MAE / PR-AUC / F1 / ROC-AUC per checkpoint, mean+std over folds |
| results/table_ablation.csv, table_ablation_avg.csv, ablation_per_fold.csv | TCN ablation (+ LightGBM without run-time context) |
| results/table_sensitivity.csv, sensitivity_per_fold.csv | kernel size x residual levels (folds 0-2) |
| results/table_temporal_holdout.csv, temporal_holdout_per_seed.csv | train on first 80 % of releases, test last 20 %, 5 seeds |
| results/significance.csv | Wilcoxon signed-rank on absolute errors, bootstrap CI of PR-AUC difference |
| results/table_waiting_attribution.csv | waiting time per activity/resource, late vs on-time |
| results/table_bottleneck_hotspots.csv | CART waiting hotspots + log-based what-if bound |
| results/table_wait_resource_hour.csv | mean waiting per resource x ready hour |
| results/metrics.json | all headline numbers |
| results/raw_checkpoint.pkl | pickled intermediate results (out-of-fold predictions, curves) used by FIGS_ONLY=1 |
| figures/*.png (300 dpi), results/*.jpg | Fig. 1 flowchart, Fig. 2 main comparison, Fig. 3 PR curves + scatter, Fig. 4 sensitivity + learning curves, Fig. 5 waiting heat map, Fig. 6 hotspots |

`fig1_flowchart.dot` is the Graphviz source of Fig. 1 (rendered by run_all.py).
