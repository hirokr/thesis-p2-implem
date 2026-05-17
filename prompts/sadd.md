```I have the conda sadd env for this repo. and the dataset metadata on the C:\t309\dataSubset on this folder.
Check if i can test this model on the dataset or not```

```
i want to custome code that can run this model test it on all the given dataset and save the results in the C:\t309\results\sadd\sadd.md folder on the folloing metrics | Timestamp | Dataset | Samples | Threshold | ThresholdStrategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | EER | FPR |
before writing any code plan and ask me for clarification
```

```
Which datasets should I run: run all sequentially one by one and add a --dataset [name] for custom choice
Are the raw videos present under: no those are the video metadatas in .metadata.json format also contains the location of the actual files, if needed create a new folder and file with required modification.
checkpoint: C:\t309\detectors\SADD\.weights\model_best_epoch50.pth.tar
ThresholdStrategy: f1
device: GPU
append on C:\t309\results\sadd\sadd.md
if needed for for future and fast to test again then save else don't save.
additional command: if one file is preprocessed then it should be saved and not be preprocessed again
env: conda activate sadd
add a --dry_run flag to check that this pipeline works for every dataset. this flag should run only 10 videos per datasets
```