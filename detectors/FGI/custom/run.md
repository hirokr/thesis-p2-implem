conda activate fgi
python detectors/FGI/run_all_datasets.py --datasets all --checkpoint C:\t309\detectors\FGI\model_best_epoch99.pth.tar --device cuda --dry_run

conda run -n fgi python detectors/FGI/run_all_datasets.py --datasets all --checkpoint "C:\t309\detectors\FGI\model_best_epoch99.pth.tar" --device cuda

conda run -n fgi python detectors/FGI/run_all_datasets.py --datasets all --checkpoint "C:\t309\detectors\FGI\model_best_epoch99.pth.tar" --device cuda --preprocess full

Only one dataset:
conda run -n fgi python detectors/FGI/run_all_datasets.py --datasets dfdc --checkpoint "C:\t309\detectors\FGI\model_best_epoch99.pth.tar" --device cuda


python detectors/FGI/run_all_datasets.py --datasets av1,dfdc,faceavceleb --threshold_strategy fixed --threshold 0.5

python detectors/FGI/run_all_datasets.py --datasets dfdc --threshold_strategy fixed --threshold 0.5