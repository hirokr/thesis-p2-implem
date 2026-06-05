Review and fix the result-generation code for this detector the same way you fixed FGI.

Files:
- Results file: [sadd.md](thesis-p2-implem/results/sadd/sadd.md) 
- Runner/evaluation code: [run_all_datasets.py](thesis-p2-implem/detectors/SADD/custom/run_all_datasets.py) 
- Metadata files:
  - thesis-p2-implem/dataSubset/av1.metadata.json
  - thesis-p2-implem/dataSubset/dfdc.metadata.json
  - thesis-p2-implem/dataSubset/faceavceleb.metadata.json

Goal:
The results are not trustworthy. Find bugs in the code that cause wrong labels, wrong sample/video counts, wrong threshold usage, or wrong metric calculation. Fix the bugs so the output metrics are correct.

Check specifically:
1. Verify label mapping for all datasets:
   - av1: `modify_type == "real"` means real, `modify_type == "fake"` means fake.
   - dfdc: metadata `label == "REAL"` means real, `label == "FAKE"` means fake.
   - faceavceleb: `method == "real"` or `type == "RealVideo-RealAudio"` means real; all other types are fake.
2. Ensure evaluation uses:
   - fake = 1
   - real = 0
   - higher model score means more fake, unless the model clearly outputs the opposite. If the score direction is opposite, fix the score or label direction.
3. Check that metrics are computed at video level, not accidentally at chunk/frame level unless the detector is designed that way.
4. Check that `Accuracy`, `Precision`, `Recall`, `F1`, `ROC_AUC`, `PR_AUC`, `EER`, and `FPR` are calculated correctly.
5. Reject or avoid `f1` threshold tuning on the test set. Use a fixed threshold or a threshold chosen from validation data.
6. Add logging that prints selected/staged/evaluated counts by label:
   - real count
   - fake count
   - missing/skipped real
   - missing/skipped fake
7. Clean or reset the results markdown if it contains stale incorrect rows from old buggy runs.

Important:
- Do not change metadata paths unless necessary. I will test on Windows, so Windows paths like `C:/t309/...` should continue to work.
- Keep changes scoped to the detector runner/evaluation code and its results file.
- After fixing, run syntax checks and a metadata label-count check.
- Tell me exactly what bugs you found, what you fixed, and what command I should run to generate correct results.