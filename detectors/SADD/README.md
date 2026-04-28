# Statistics-aware Audio-visual DeepFake Detector
This repository contains the implementation of Audio Visual Deepfake Detection method proposed in the paper  
  
Marcella Astrid, Enjie Ghorbel, and Djamila Aouada, Statistics-aware Audio-visual DeepFake Detector, ICIP 2024.  
Links: [[PDF]](https://www.arxiv.org/pdf/2407.11650) 

# Dependencies
Create conda environment with package inside the package list `conda create -n myenv --file package-list.txt`
  
# Prepare data (DFDC)
1) Download DFDC dataset from [here](https://www.kaggle.com/c/deepfake-detection-challenge/data). 
  
2) Store the train and test videos as follows:  

   ```
   train/real/{videoname}.mp4  
   train/fake/{videoname}.mp4  
   test/real/{videoname}.mp4  
   test/fake/{videoname}.mp4
   ```
  
   If you wish to use the same videos as used by the paper authors, please refer to `train_fake_list.txt`,  `train_real_list.txt`,  `test_fake_list.txt`, and `test_real_list.txt`. 
  
3) Once the videos have been placed at the above mentioned paths, run `python pre-process.py --out_dir train` and `python pre-process.py --out_dir test` for pre-processing the videos.  
  
4) After the above step, you can delete `pyavi`, `pywork`, `pyframes` and `pycrop` directories under `train` and `test` folders. (Do not delete `pytmp` folder please!)  
  
5) Collect video paths in csv files by running `python write_csv.py --out_dir . ` command. Also can create the small set version (e.g., used in Table 3) with `python write_csv.py --out_dir . --small`  

# Prepare data (FakeAVCeleb)
1) Download FakeAVCeleb dataset from [here](https://github.com/DASH-Lab/FakeAVCeleb/blob/main/dataset/README.md)
2) Run `python preprocess_FakeAVCeleb_to_DFDCformat.py`
3) Use instructions in Prepare data (DFDC). Adjust the --out_dir in step 3 respectively. Also add --dont_crop_face option in step 3. Use write_csv_fakeavceleb.py for step 5

# Training
```
python train.py --out_dir . --half_network --audio_format waveform --epochs 50 --num_workers 7 --mean_separation_loss_weight 1 
```
Remove `--audio_format waveform` for frequency based input.
Remove `--half_network` for deeper network.
Change value in `--mean_separation_loss_weight` to change the α in Eq. (3). E.g., to 0.01 `--mean_separation_loss_weight 0.01`

# Testing
Weight file for Table 1:

(a) and (b): [drive](https://drive.google.com/drive/folders/1zqA90iJrpZBMCJonkd5zHD_XlzAGL6W2?usp=sharing)

(c) [drive](https://drive.google.com/drive/folders/1lpVyge7oQuqNVEHaWnyGe9ORhwP2AmJv?usp=sharing) 

(d) [drive](https://drive.google.com/drive/folders/1pcc6k_9a2w-QqdL4NTCW6-lhGSdyBAEA?usp=sharing)

(e) [drive](https://drive.google.com/drive/folders/1MJUKdyh7RyD7OyUeJBa8NTgYWZXhty2M?usp=sharing)

```
python train.py --half_network --num_workers 7 --test log_tmp/v5_deepfake_audio-waveform-msl1.0-224_r18_bs8_lr0.001_half/model/model_best_epoch50.pth.tar --out_dir . --audio_format waveform
```
Change the path of model file accordingly in the --test argument.  
  
For computing AUC score, run `python test.py --folder test_results/v5_deepfake_audio-waveform-msl1.0-224_r18_bs8_lr0.001_half/model_best_epoch50/normal --normalize train` after executing the above command, and see the result in test_results folder. 
Use `--normalize none` for without score normalization during test. 
See the results in the `output.txt` inside `test_results/v5_deepfake_audio-waveform-msl1.0-224_r18_bs8_lr0.001_half/model_best_epoch50/normal`
  
# Reference
If you use the code, please cite the paper -
```
@InProceedings{astrid2024statistics,
  author       = "Astrid, Marcella and Ghorbel, Enjie and Aouada, Djamila",
  title        = "Statistics-aware Audio-visual DeepFake Detector",
  booktitle    = "IEEE International Conference on Image Processing (ICIP)",
  year         = "2024",
}
```
# Acknowledgements
Thanks to the code available at https://github.com/abhinavdhall/deepfake/tree/main/ACM_MM_2020, https://github.com/TengdaHan/DPC and https://github.com/joonson/syncnet_python.  
  



