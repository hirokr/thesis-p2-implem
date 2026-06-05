1. Which datasets should I run: run all sequentially  one by one and add a `--dataset [name]` for custom choice
2. Are the raw videos present under: no those are the video metadatas in `.metadata.json` format also contains the location of the actual files, if needed create a new folder and file with required modification.
3. checkpoint: `C:\t309\detectors\SADD\.weights\model_best_epoch50.pth.tar`
4. ThresholdStrategy: `f1`
5. device: `GPU`
6. append on `C:\t309\results\sadd\sadd.md`
7. if needed for for future and fast to test again then save else don't save.

additional command: if one file is preprocessed then it should be saved and not be preprocessed again
env: `conda activate sadd`
add a `--dry_run`  flag to check that this pipeline works for every dataset. this flag should run only 10 videos per datasets

conda activate sadd
python detectors/SADD/custom/run_all_datasets.py --dataset all --dry_run

python detectors/SADD/custom/run_all_datasets.py --dataset av1 --device gpu --checkpoint C:\t309\detectors\SADD\.weights\model_best_epoch50.pth.tar


python detectors/SADD/custom/run_all_datasets.py --dataset faceavceleb --device gpu --checkpoint C:\t309\detectors\SADD\.weights\model_best_epoch50.pth.tar

python C:/t309/detectors/SADD/custom/run_all_datasets.py --dataset all --data_root C:/t309/dataSubset --results_file C:/t309/results/sadd/sadd.md --cache_root C:/t309/results/sadd/cache --checkpoint C:/t309/detectors/SADD/.weights/model_best_epoch50.pth.tar --device gpu --threshold_strategy fixed --threshold 0.5