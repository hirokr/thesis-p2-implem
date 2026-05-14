import os
import gc
import json
import time
import numpy as np
import glob
import random
import librosa
from pprint import pprint
from tqdm import tqdm
# from moviepy.editor import VideoFileClip
import subprocess

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torch.nn import functional as F
from functools import partial

from .datasets import register_dataset
from .data_utils import (
    truncate_feats,
    build_precompute_loader,
    get_cuda_mem_mb,
    get_cpu_rss_mb,
    get_loader_queue_size,
    is_pin_memory_ok,
    set_pin_memory_ok,
)
import torchaudio.transforms as T


def _collate_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)


def _is_oom_error(exc):
    message = str(exc).lower()
    return "out of memory" in message or "cuda" in message and "memory" in message


def _is_pin_memory_error(exc):
    message = str(exc).lower()
    return "pin memory" in message or "pinned memory" in message


class AudioPrecomputeDataset(Dataset):
    def __init__(self, data_list, dataset_root, sample_rate):
        self.data_list = data_list
        self.dataset_root = dataset_root
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        audio_path = os.path.join(self.dataset_root, video_item["id"])
        try:
            waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            waveform = torch.from_numpy(waveform)
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        except Exception:
            with open("audios.txt", "a") as f:
                f.write(f"{audio_path}\n")
            return None
        return waveform, sample_rate, video_item["id"]


@register_dataset("ijcai25audio")
class IJCAI25(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
        devices,
        num_workers,
        ssl_model,
        dataset_root,
        feat_stride,        # temporal stride of the feats
        num_frames,         # number of frames for each feat
        downsample_rate,    # downsample rate for feats
        max_seq_len,        # maximum sequence length during training
        trunc_thresh,       # threshold for truncate an action segment
        crop_ratio,         # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,          # input feat dim
        num_classes,        # number of action categories
        featureMapIndex,
        max_items=None,
        precompute_batch_size=1,
        precompute_log_every=25,
        precompute_cleanup_every=50,
        precompute_prefetch_factor=2,
    ):
        assert crop_ratio == None or len(crop_ratio) == 2
        self.is_training = is_training
        self.ssl_model = ssl_model
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
        self.featureMapIndex = featureMapIndex
        self.label_dict = None

        self.max_items = max_items
        self.precompute_batch_size = max(1, int(precompute_batch_size))
        if self.precompute_batch_size not in (1, 2):
            self.precompute_batch_size = 1
        self.precompute_log_every = max(1, int(precompute_log_every))
        self.precompute_cleanup_every = max(1, int(precompute_cleanup_every))
        self.precompute_prefetch_factor = max(1, int(precompute_prefetch_factor))

        # 读取数据集信息，并存储为list
        self.dataset_root = dataset_root
        dict_db = self._load_label_db()
        self.data_list = dict_db
        if self.max_items:
            self.data_list = self.data_list[: self.max_items]
        self.device = torch.device(devices[0])
        self.num_workers = num_workers

        # 设置数据集属性
        self.db_attributes = {
            'dataset_name': 'ijcai25',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            'empty_label_ids': [],
        }

        # 设置音频预训练模型model
        self.bundle = getattr(torchaudio.pipelines, f"{self.ssl_model}")
        self.bundle_sample_rate = self.bundle.sample_rate
        self.transform = None

        # 新增缓存相关配置
        self.cache_dir = os.path.join('./.cached_data/', f'ijcai25_{ssl_model}_feat_{self.featureMapIndex}')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(f'{self.cache_dir}/train_data/fake_audio_real_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/train_data/real_audio_real_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/train_data/fake_audio_fake_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/train_data/real_audio_fake_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/val_data/fake_audio_real_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/val_data/real_audio_real_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/val_data/fake_audio_fake_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/val_data/real_audio_fake_visual/', exist_ok=True)

        # 检查是否需要预计算特征
        self.need_precompute = any(
            not os.path.exists(self._get_feature_path(video_item['id']))
            for video_item in self.data_list
        )

        # 预计算特征（仅在需要时执行）
        if self.need_precompute:
            self._precompute_features()

    def delta(sef,mat):
        '''
        有限差分滤波器，计算它相对于列的一阶导数。
        win: 滤波器组的系数
        '''
        assert mat.ndim == 2
        win = np.array([-1.0, 0.0, 1.0]).reshape(3, 1)
        mat = np.concatenate((mat[:, :1], mat, mat[:, -1:]), axis=-1)
        mat = np.expand_dims(mat, 2)
        mat = np.concatenate((mat[:, :-2], mat[:, 1:-1], mat[:, 2:]), axis=2)
        t, v = mat.shape[:2]
        mat = np.dot(mat.reshape(-1, 3), win).reshape(t, v)
        return mat

    def get_attributes(self):
        return self.db_attributes

    def _load_label_db(self):
        file_paths = []
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                if file.endswith(".json"):
                    file_paths.append(os.path.join(root, file))

        dict_db = []  # 改用列表存储
        split = 'val' if self.is_training == False else 'train'
        for json_path in tqdm(file_paths, desc='Loading Dataset'):
            with open(json_path, 'r') as f:
                data = json.load(f)
                if data["split"] != split:
                    continue  # 根据split过滤
                if 'fake_audio' in data["modify_type"]:
                    segments = data.get("audio_fake_segments", []), # 音频伪造片段
                    segments = np.array(segments, dtype=np.float32).reshape(-1, 2)
                    labels = np.zeros(len(segments), dtype=np.int64)
                else:
                    segments = np.empty((0, 2))
                    labels = np.empty((0, 1))

                # 构建样本信息
                video_id = 'train_data/' + data["file"] if split == 'train' else 'val_data/' + data["file"]
                dict_db.append({
                    'id': video_id,
                    'fps': 50, # 设置fps，音频帧率为50
                    'duration': data["video_frames"] / 25.0, # 25为视频帧率,用视频帧数计算时长
                    'segments': segments,
                    'labels': labels
                })
                if self.max_items and len(dict_db) >= self.max_items:
                    break
            if self.max_items and len(dict_db) >= self.max_items:
                break

        return dict_db

    def _get_feature_path(self, audio_id):
        """生成特征文件的路径"""
        dirname = os.path.dirname(audio_id)
        basename = os.path.basename(audio_id)
        feature_dir = os.path.join(self.cache_dir, dirname)
        root, _ = os.path.splitext(basename)
        return os.path.join(feature_dir, f"{root}.npy")

    def _precompute_features(self):
        """Audio feature precompute with safe workers and logging."""

        def log_status(tag, batch_idx, processed, start_time, loader):
            elapsed = max(1e-6, time.perf_counter() - start_time)
            rate = processed / elapsed
            vram = get_cuda_mem_mb(self.device)
            cpu = get_cpu_rss_mb()
            queue_size = get_loader_queue_size(loader)
            if vram:
                vram_text = f"{vram[0]:.0f}/{vram[1]:.0f}MB"
            else:
                vram_text = "n/a"
            cpu_text = f"{cpu:.0f}MB" if cpu is not None else "n/a"
            queue_text = str(queue_size) if queue_size is not None else "n/a"
            print(
                f"[PRECOMPUTE][{tag}] batch {batch_idx} "
                f"samples={processed} rate={rate:.2f}/s vram={vram_text} "
                f"cpu={cpu_text} queue={queue_text}"
            )

        def run_precompute(batch_size, pin_memory):
            pending = [
                item for item in self.data_list
                if not os.path.exists(self._get_feature_path(item["id"]))
            ]
            if not pending:
                return "ok"

            workers = self.num_workers or 0
            if os.name == "nt":
                workers = max(4, min(workers or 4, 8))

            dataset = AudioPrecomputeDataset(
                pending,
                dataset_root=self.dataset_root,
                sample_rate=self.bundle.sample_rate,
            )
            loader = build_precompute_loader(
                dataset,
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=pin_memory,
                collate_fn=_collate_skip_none,
                prefetch_factor=self.precompute_prefetch_factor,
            )

            processed = 0
            start_time = time.perf_counter()
            try:
                for batch_idx, batch in enumerate(loader, start=1):
                    if batch is None:
                        continue
                    waveform, sample_rate, audio_id = batch
                    if not isinstance(audio_id, (list, tuple)):
                        audio_ids = [audio_id]
                    else:
                        audio_ids = list(audio_id)
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)

                    for idx, audio_key in enumerate(audio_ids):
                        feature_path = self._get_feature_path(audio_key)
                        try:
                            waveform_i = waveform[idx].unsqueeze(0).to(self.device, non_blocking=True)
                            device_type = "cuda" if self.device.type == "cuda" else "cpu"
                            with torch.inference_mode(), torch.autocast(
                                device_type=device_type,
                                enabled=self.device.type == "cuda",
                            ):
                                feats, _ = self.model.extract_features(waveform_i)
                                feats = torch.stack(feats, dim=0).squeeze()
                                feats = feats[self.featureMapIndex].transpose(1, 0)
                            np.save(feature_path, feats.cpu().numpy().astype(np.float32))
                        except RuntimeError as exc:
                            if _is_oom_error(exc):
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                if batch_size > 1:
                                    return "oom"
                                with open("audios.txt", "a") as f:
                                    f.write(f"GPU memory is insufficient: {audio_key}\n")
                                continue
                            raise
                        finally:
                            if "waveform_i" in locals():
                                del waveform_i
                            if "feats" in locals():
                                del feats

                        processed += 1
                    if processed == 1 or processed % self.precompute_log_every == 0:
                        log_status("audio", batch_idx, processed, start_time, loader)
                    if processed % self.precompute_cleanup_every == 0:
                        gc.collect()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
            except RuntimeError as exc:
                if _is_pin_memory_error(exc):
                    return "pin_memory"
                if _is_oom_error(exc):
                    return "oom"
                raise
            return "ok"

        self.model = self.bundle.get_model()
        self.model.eval().to(self.device)

        pin_memory = self.device.type == "cuda" and is_pin_memory_ok()
        batch_size = self.precompute_batch_size
        while True:
            result = run_precompute(batch_size, pin_memory)
            if result == "ok":
                break
            if result == "pin_memory" and pin_memory:
                set_pin_memory_ok(False)
                pin_memory = False
                continue
            if result == "oom" and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"[WARN] OOM detected, reducing batch_size to {batch_size}")
                continue
            if result != "ok":
                print("[WARN] Precompute stopped early due to repeated OOM.")
            break

        self.model.cpu()
        del self.model

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        audio_id = video_item['id']

        feature_path = self._get_feature_path(video_item['id'])
        feats = torch.from_numpy(np.ascontiguousarray(np.load(feature_path, mmap_mode='r')))
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride

        segments = torch.from_numpy(video_item['segments'] * video_item['fps'] / feat_stride - feat_offset)
        labels = torch.from_numpy(video_item['labels'])

        data_dict = {
            'video_id': audio_id,
            'feats': feats,  # C x T
            'segments': segments,  # N x 2
            'labels': labels,  # N
            'fps': video_item['fps'],
            'duration': video_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames
        }

        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh,
                feat_offset, self.crop_ratio
            )

        return data_dict
