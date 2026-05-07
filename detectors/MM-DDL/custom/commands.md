```cli
cmd
# Environment setup
pip install -r requirements.txt

# Build NMS extension (Linux)
cd .\libs\utils
python setup.py install --user
cd ..\..

# Download weights (Linux)
cd .\.weights
bash hfd.sh openai/clip-vit-base-patch16 --tool wget
bash hfd.sh microsoft/xclip-base-patch16 --tool wget
cd ..

# Train (default configs)
python train-audio.py
python train-video.py

# Test (default configs)
python test-audio.py
python test-video.py

# Combine results
python combine_results.py
```

```cli
powershell
# Example with explicit configs
python train-audio.py --config configs_train/ijcai25audio-wavLM.yaml
python train-video.py --config configs_train/ijcai25video-CLIP16.yaml

python test-audio.py --config configs_test/ijcai25audio-wavLM.yaml
python test-video.py --config configs_test/ijcai25video-CLIP16.yaml
```
