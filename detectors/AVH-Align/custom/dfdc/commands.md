```cli
cmd
python custom/dfdc/preprocess.py

python custom/dfdc/preprocess.py --base_dataset_folder "C:\t309\dataSubset" --preprocessed_root "C:\t309\results\avh_aligned\preprocessed"  --features_root "C:\t309\results\avh_aligned\features" --results_file "C:\t309\results\avh_aligned\result.md"

python custom/dfdc/preprocess.py --datasets av1,dfdc

python custom/dfdc/preprocess.py --threshold_strategy youden


```cli
cmd
python .\custom\dfdc\preprocess.py

python .\custom\dfdc\preprocess.py --base_dataset_folder "C:\t309\dataSubset" --dfdc_root "C:\t309\dataset\dfdc" --preprocessed_root "C:\t309\results\avh_aligned\preprocessed" --features_root "C:\t309\results\avh_aligned\features" --results_file "C:\t309\results\avh_aligned\result.md"