````cli
cmd
python custom/dfdc/preprocess.py --auto_download_assets --dry_run
python custom/dfdc/preprocess.py --auto_download_assets

python custom/dfdc/preprocess.py --auto_download_assets --base_dataset_folder "C:\t309\dataSubset" --preprocessed_root "C:\t309\results\avh_aligned\preprocessed" --features_root "C:\t309\results\avh_aligned\features" --results_file "C:\t309\results\avh_aligned\result.md"

python custom/dfdc/preprocess.py --auto_download_assets --datasets av1,dfdc

python custom/dfdc/preprocess.py --auto_download_assets --threshold_strategy youden


```cli
cmd
python .\custom\dfdc\preprocess.py --auto_download_assets --dry_run
python .\custom\dfdc\preprocess.py --auto_download_assets

python .\custom\dfdc\preprocess.py --auto_download_assets --base_dataset_folder "C:\t309\dataSubset" --dfdc_root "C:\t309\dataset\dfdc" --preprocessed_root "C:\t309\results\avh_aligned\preprocessed" --features_root "C:\t309\results\avh_aligned\features" --results_file "C:\t309\results\avh_aligned\result.md"
````



```cli
new
python custom/dfdc/preprocess.py --auto_download_assets --base_dataset_folder "C:\t309\dataSubset" --preprocessed_root "C:\t309\results\avh_aligned\preprocessed" --features_root "C:\t309\results\avh_aligned\features" --results_file "C:\t309\results\avh_aligned\result.md" --max_workers 16
```

python detectors/AVH-Align/custom/dfdc/preprocess.py --datasets av1,dfdc,faceavceleb --threshold_strategy fixed --threshold 0.5