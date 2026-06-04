| Timestamp| Dataset | Samples | Threshold | ThresholdStrategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | EER  | FPR    |
| -------- | ------- | ------- | --------- | ----------------- | -------- | --------- | ----- | --- | ------- | ------ | ---- | ------ |
| 2026-06-04 15:10:23 | av1 | 1 | 0.5000 | fixed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA | NA |
| 2026-06-04 15:12:48 | av1 | 3 | 0.5000 | fixed | 0.6667 | 0.6667 | 1.0000 | 0.8000 | 0.5000 | 0.8333 | 0.2500 | 0.0000 |
| 2026-06-04 15:16:38 | av1 | 5 | 0.5000 | fixed | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.5000 | 0.8875 | 0.2500 | 0.0000 |

conda activate auvire; python detectors\auvire\scripts\eval_datasubset.py --datasets av1 --max_videos 5 --device cuda:0 --progress_every 2 --results_path C:\t309\results\avuire\smoke_results.mdconda activate auvire; python detectors\auvire\scripts\eval_datasubset.py --datasets av1 --max_videos 5 --device cuda:0 --progress_every 2 --results_path C:\t309\results\avuire\smoke_results.md