```cli
# From repo root (C:\t309)
python detectors/FGI/custom/run_all.py --with_att

# Limit videos per dataset
python detectors/FGI/custom/run_all.py --with_att --max_videos 200

# Skip preprocessing if already done
python detectors/FGI/custom/run_all.py --with_att --skip_preprocess

# Only run a subset
python detectors/FGI/custom/run_all.py --with_att --datasets av1,dfdc,lavdf

# Force using all items, ignoring split fields
python detectors/FGI/custom/run_all.py --with_att --split all
```
