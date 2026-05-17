
I have the conda fgi env for this repo. and the dataset metadata on the C:\t309\dataSubset on this folder.
Check if i can test this model on the dataset or not

---

i want to custome code that can run this model test it on all the given dataset and save the results in the C:\t309\results\fgi\fgi.md folder on the folloing metrics | Timestamp | Dataset | Samples | Threshold | ThresholdStrategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | EER | FPR |
before writing any code plan and ask me for clarification

---

1. Which datasets should I run: run all sequentially one by one and add a --dataset [name] for custom choice
2. Are the raw videos present under: no those are the video metadatas in .metadata.json format also contains the location of the actual files, if needed create a new folder and file with required modification.
3. checkpoint: C:\t309\detectors\SADD\.weights\model_best_epoch50.pth.tar
4. ThresholdStrategy: f1
5. device: GPU
6. append on C:\t309\results\sadd\sadd.md
7. if needed for for future and fast to test again then save else don't save.
8. additional command: if one file is preprocessed then it should be saved and not be preprocessed again
9. env: conda activate sadd
10. add a --dry_run flag to check that this pipeline works for every dataset. this flag should run only 10 videos per datasets

---

1. Which datasets should I run: run all sequentially one by one and add a --dataset [name] for custom choice
2. I have not done preprocessing for this mode you that in the pipeline
3. ThresholdStrategy: f1
4. checkpoint: C:\t309\detectors\FGI\model_best_epoch99.pth.tar
5. device: GPU
6. a script to run end-to-end. it should do everything in one pass.

add a --dry_run flag to check that this pipeline works for every dataset. this flag should run only 10 videos per datasets