import os
import gc
from pathlib import Path
import cv2
import json
import math
import time
import numpy as np
import librosa
import albumentations
from tqdm import tqdm

import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel

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

import json

CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cached_data" / "test"


def _debug(message):
    if os.getenv("MMDDL_DEBUG", "0") == "1":
        print(message)


def _warn_if_lfs_pointer(path):
    try:
        if path.exists():
            with path.open("rb") as handle:
                head = handle.read(64)
            if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
                print(f"[WARN] LFS pointer detected at {path}. Run 'git lfs pull' in that folder.")
    except OSError:
        return


def _normalize_rel_path(path_value, dataset_root):
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        try:
            return os.path.relpath(path_value, dataset_root)
        except ValueError:
            return os.path.basename(path_value)
    return path_value


def _needs_rebuild_dict_db(entries, dataset_root):
    if not entries:
        return True
    sample = entries[0]
    if "rel_path" not in sample or "video_path" not in sample:
        return True
    for item in entries[:5]:
        rel_path = item.get("rel_path")
        if not rel_path:
            return True
        if not os.path.exists(os.path.join(dataset_root, rel_path)):
            return True
    return False


def _make_feature_path(cache_root, rel_path):
    rel_path = rel_path.replace("\\", "/")
    root, _ = os.path.splitext(rel_path)
    root = root.replace(":", "")
    feature_path = os.path.join(cache_root, f"{root}.npy")
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    return feature_path


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

# 保存 dict_db 到 JSON 文件
def save_dict_db_to_json(dict_db, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_db, f, indent=4)

# 从 JSON 文件读取 dict_db
def load_dict_db_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


Transformers = [
    'CLIP-16',
    'XCLIP-16',
]

def choose_CLIP_model(clip_model):
    weights_root = Path(__file__).resolve().parents[2] / ".weights"
    def _load_model(model_cls, model_dir, model_name):
        _debug(f"[DEBUG] Loading {model_name} from {model_dir}")
        _warn_if_lfs_pointer(model_dir / "model.safetensors")
        _warn_if_lfs_pointer(model_dir / "pytorch_model.bin")
        try:
            return model_cls.from_pretrained(str(model_dir))
        except Exception as exc:
            message = str(exc).lower()
            if "safetensor" in message or "headertoolarge" in message:
                print(f"[WARN] {model_name} safetensors load failed; retrying with pytorch weights.")
                return model_cls.from_pretrained(str(model_dir), use_safetensors=False)
            raise

    match clip_model:
        case 'CLIP-16':
            return _load_model(CLIPVisionModel, weights_root / "clip-vit-base-patch16", "CLIP-16")
        case 'XCLIP-16':
            return _load_model(XCLIPVisionModel, weights_root / "xclip-base-patch16", "XCLIP-16")

def pad_tensor_to_multiple_of_8(x):
    T = x.size(0)
    remainder = T % 8
    if remainder == 0:
        return x
    pad_size = 8 - remainder
    pad_tensor = x[-1].unsqueeze(0).repeat(pad_size, 1, 1, 1)
    x_padded = torch.cat([x, pad_tensor], dim=0)
    return x_padded


class WaveformDataset(Dataset):
    def __init__(self, data_list, dataset_root, sample_rate):
        self.data_list = data_list
        self.dataset_root = dataset_root
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        audio_path = video_item.get("video_path") or os.path.join(self.dataset_root, video_item["id"])
        try:
            waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        except Exception:
            with open("audios.txt", "a") as f:
                f.write(f"{audio_path}\n")
            return None

        waveform = torch.from_numpy(waveform)
        waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform, sample_rate, video_item["id"]


class VideoFrameDataset(Dataset):
    def __init__(self, data_list, dataset_root, sample_rate):
        self.data_list = data_list
        self.dataset_root = dataset_root
        self.sample_rate = sample_rate
        self.trans = self._default_transform()

    def __len__(self):
        return len(self.data_list)

    def _default_transform(self):
        return albumentations.Compose([
            albumentations.Resize(224, 224),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        video_path = video_item.get("video_path") or os.path.join(self.dataset_root, video_item["id"])
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("Unable to open video")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return None
            frames = []
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                augmented = self.trans(image=frame)
                image = augmented["image"]
                image = image.transpose(2, 0, 1)[np.newaxis, :]
                frames.append(image)
            if not frames:
                return None
            frames = np.concatenate(frames, axis=0)
            frames = torch.tensor(frames).unsqueeze(0).squeeze(0)
            return frames, video_item["id"]
        except Exception:
            with open("videos.txt", "a") as f:
                f.write(f"{video_path}\n")
            return None
        finally:
            cap.release()


class VideoChunkDataset(Dataset):
    def __init__(self, data_list, dataset_root, chunk_size):
        self.data_list = data_list
        self.dataset_root = dataset_root
        self.chunk_size = max(1, int(chunk_size))
        self.trans = self._default_transform()
        self.tasks = []
        for item in self.data_list:
            total_frames = int(item.get("frame_count") or 0)
            if total_frames <= 0:
                continue
            num_chunks = max(1, (total_frames + self.chunk_size - 1) // self.chunk_size)
            for chunk_idx in range(num_chunks):
                start = chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, total_frames)
                self.tasks.append((item, start, end, total_frames))

    def _default_transform(self):
        return albumentations.Compose([
            albumentations.Resize(224, 224),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        video_item, start, end, total_frames = self.tasks[idx]
        video_path = video_item.get("video_path") or os.path.join(self.dataset_root, video_item["id"])
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("Unable to open video")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            current = start
            while current < end:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                augmented = self.trans(image=frame)
                image = augmented["image"].transpose(2, 0, 1)
                frames.append(image)
                current += 1
            if not frames:
                return None
            frames = np.ascontiguousarray(np.stack(frames, axis=0))
            frames = torch.from_numpy(frames)
            return frames, video_item["id"], start, total_frames
        except Exception:
            with open("videos.txt", "a") as f:
                f.write(f"{video_path}\n")
            return None
        finally:
            cap.release()

@register_dataset("ijcai25testvideo")
class IJCAI25testvideo(Dataset):
    def __init__(
        self,
        is_training,        
        devices,
        num_workers,
        ssl_model,
        clip_model,
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
        precompute_chunk_size=32,
        precompute_log_every=25,
        precompute_cleanup_every=50,
        precompute_prefetch_factor=2,
    ):
        assert crop_ratio == None or len(crop_ratio) == 2
        self.is_training = is_training
        self.ssl_model = ssl_model
        self.clip_model = clip_model
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
        self.precompute_chunk_size = max(1, int(precompute_chunk_size))
        self.precompute_log_every = max(1, int(precompute_log_every))
        self.precompute_cleanup_every = max(1, int(precompute_cleanup_every))
        self.precompute_prefetch_factor = max(1, int(precompute_prefetch_factor))

        # 读取数据集信息，并存储为list
        self.dataset_root = dataset_root
        _debug(f"[DEBUG] Dataset root: {self.dataset_root}")
        dict_db = self._load_label_db()
        self.data_list = dict_db
        if self.max_items:
            self.data_list = self.data_list[: self.max_items]
        _debug(f"[DEBUG] Loaded {len(self.data_list)} items")
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
        self.video_cache_dir = os.path.join(str(CACHE_ROOT), f'ijcai25_{clip_model}_feats')
        self.make_cache_dir(self.video_cache_dir)
        self.audio_cache_dir = os.path.join(str(CACHE_ROOT), f'ijcai25_{ssl_model}_feat_{self.featureMapIndex}')
        self.make_cache_dir(self.audio_cache_dir)
        _debug(f"[DEBUG] Video cache: {self.video_cache_dir}")
        _debug(f"[DEBUG] Audio cache: {self.audio_cache_dir}")

        # 检查是否需要预计算特征
        self.video_need_precompute = any(
            not os.path.exists(self._get_feature_path(video_item['id'],modal='video'))
            for video_item in self.data_list
        )
        self.audio_need_precompute = any(
            not os.path.exists(self._get_feature_path(video_item['id'],modal='audio'))
            for video_item in self.data_list
        )
        _debug(f"[DEBUG] Precompute video: {self.video_need_precompute}, audio: {self.audio_need_precompute}")

        # 预计算特征（仅在需要时执行）
        if self.video_need_precompute and clip_model is not None:
            self._precompute_features(modal='video')
        if self.audio_need_precompute and ssl_model is not None:
            self._precompute_features(modal='audio')

    def make_cache_dir(self,cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    def _load_label_db(self):
        file_paths = []
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                if file.endswith(".mp4"):
                    file_paths.append(os.path.join(root, file))
                    if self.max_items and len(file_paths) >= self.max_items:
                        break
            if self.max_items and len(file_paths) >= self.max_items:
                break

        db_path = os.path.join(str(CACHE_ROOT), 'dict_db.json')
        if os.path.exists(db_path):
            _debug(f"[DEBUG] Using cached labels: {db_path}")
            dict_db = load_dict_db_from_json(db_path)
            if not _needs_rebuild_dict_db(dict_db, self.dataset_root):
                return dict_db
            _debug("[DEBUG] Cached labels invalid; rebuilding.")
        else:
            _debug(f"[DEBUG] Scanning {len(file_paths)} videos for metadata")
            dict_db = []
            for video_path in tqdm(file_paths, desc='Loading Dataset'):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Unable to open video file: {video_path}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()

                # 确保帧率和帧数有效
                if fps <= 0 or frame_count <= 0:
                    duration = 0.0
                else:
                    duration = frame_count / fps

                # 构造 video_id：可以是文件名，也可以是相对路径
                rel_path = _normalize_rel_path(video_path, self.dataset_root)

                # 构建样本字典，仅保留基本字段
                dict_db.append({
                    'id': rel_path,
                    'rel_path': rel_path,
                    'video_path': video_path,
                    'fps': fps,
                    'duration': duration,
                    'frame_count': int(frame_count)
                })
                if self.max_items and len(dict_db) >= self.max_items:
                    break
            if self.max_items and len(dict_db) >= self.max_items:
                _debug(f"[DEBUG] Reached max_items limit: {self.max_items}")
                return dict_db
            save_dict_db_to_json(dict_db,db_path)
            _debug(f"[DEBUG] Saved labels: {db_path}")

        return dict_db


    def delta(self,mat):
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


    def _get_feature_path(self, video_id, modal):
        """生成特征文件的路径"""
        if modal == 'video':
            rel_path = _normalize_rel_path(video_id, self.dataset_root)
            return _make_feature_path(self.video_cache_dir, rel_path)
        elif modal == 'audio':
            rel_path = _normalize_rel_path(video_id, self.dataset_root)
            return _make_feature_path(self.audio_cache_dir, rel_path)


    def get_attributes(self):
        return self.db_attributes


    def _precompute_features(self, modal):
        """无填充的特征预计算"""
        if modal == 'video':
            _debug("[DEBUG] Precomputing video features")
            self.model = choose_CLIP_model(self.clip_model)
            self.model.eval().to(self.device)
            filtered_data = [
                item for item in self.data_list
                if not os.path.exists(self._get_feature_path(item['id'],modal=modal))
            ]
            _debug(f"[DEBUG] Video features pending: {len(filtered_data)}")

            def finalize_video(memmap, tmp_path, final_path, written_frames):
                if memmap is None or not tmp_path or not final_path:
                    return 0
                memmap.flush()
                expected_frames, feat_dim = memmap.shape
                del memmap
                if written_frames <= 0:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    return 0
                if written_frames == expected_frames:
                    os.replace(tmp_path, final_path)
                    return 1
                src = np.lib.format.open_memmap(tmp_path, mode="r")
                dst = np.lib.format.open_memmap(
                    final_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(written_frames, feat_dim),
                )
                step = 1024
                for start in range(0, written_frames, step):
                    end = min(start + step, written_frames)
                    dst[start:end] = src[start:end]
                dst.flush()
                del src
                del dst
                os.remove(tmp_path)
                return 1

            def log_status(batch_idx, processed, completed, start_time, loader):
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
                    f"[PRECOMPUTE][video] batch {batch_idx} "
                    f"samples={processed} videos={completed} rate={rate:.2f}/s "
                    f"vram={vram_text} cpu={cpu_text} queue={queue_text}"
                )

            pin_memory = self.device.type == "cuda" and is_pin_memory_ok()
            chunk_size = self.precompute_chunk_size
            while True:
                filtered_data = [
                    item for item in self.data_list
                    if not os.path.exists(self._get_feature_path(item['id'],modal=modal))
                ]
                if not filtered_data:
                    break
                workers = self.num_workers or 0
                if os.name == "nt":
                    workers = max(4, min(workers or 4, 8))

                dataset = VideoChunkDataset(filtered_data, self.dataset_root, chunk_size)
                loader = build_precompute_loader(
                    dataset,
                    batch_size=1,
                    num_workers=workers,
                    pin_memory=pin_memory,
                    collate_fn=_collate_skip_none,
                    prefetch_factor=self.precompute_prefetch_factor,
                )

                current_video = None
                current_tmp = None
                current_path = None
                memmap = None
                written_frames = 0
                completed_videos = 0
                processed = 0
                start_time = time.perf_counter()
                try:
                    for batch_idx, batch in enumerate(loader, start=1):
                        if batch is None:
                            continue
                        frames, video_id, start_idx, total_frames = batch
                        if isinstance(video_id, (list, tuple)):
                            video_id = video_id[0]
                        if torch.is_tensor(start_idx):
                            start_idx = int(start_idx[0])
                        if torch.is_tensor(total_frames):
                            total_frames = int(total_frames[0])

                        if current_video != video_id:
                            completed_videos += finalize_video(
                                memmap, current_tmp, current_path, written_frames
                            )
                            memmap = None
                            written_frames = 0
                            current_video = video_id
                            current_path = self._get_feature_path(video_id, modal=modal)
                            current_tmp = f"{current_path}.partial"
                            if os.path.exists(current_tmp):
                                os.remove(current_tmp)

                        try:
                            frames = frames.to(self.device, non_blocking=True)
                            frames = frames.squeeze(0)
                            t, _, h, w = frames.shape
                            frames = frames.reshape(-1, 3, h, w)
                            if self.clip_model in {"XCLIP-16", "XCLIP-32"}:
                                frames = pad_tensor_to_multiple_of_8(frames)
                            device_type = "cuda" if self.device.type == "cuda" else "cpu"
                            with torch.no_grad(), torch.autocast(
                                device_type=device_type,
                                enabled=self.device.type == "cuda",
                            ):
                                if self.clip_model in Transformers:
                                    feats = self.model(frames, output_hidden_states=True).pooler_output
                                else:
                                    feats = self.model(frames)
                            feats = feats[:t]
                            feats = feats.detach().cpu().numpy().astype(np.float32)
                            if memmap is None:
                                feat_dim = feats.shape[1]
                                memmap = np.lib.format.open_memmap(
                                    current_tmp,
                                    mode="w+",
                                    dtype=np.float32,
                                    shape=(total_frames, feat_dim),
                                )
                            end_idx = start_idx + feats.shape[0]
                            memmap[start_idx:end_idx] = feats
                            written_frames = max(written_frames, end_idx)
                        except RuntimeError as exc:
                            if _is_oom_error(exc):
                                if memmap is not None:
                                    del memmap
                                    memmap = None
                                if current_tmp and os.path.exists(current_tmp):
                                    os.remove(current_tmp)
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                raise
                            raise
                        finally:
                            del frames
                            if "feats" in locals():
                                del feats

                        processed += 1
                        if processed == 1 or processed % self.precompute_log_every == 0:
                            log_status(batch_idx, processed, completed_videos, start_time, loader)
                        if processed % self.precompute_cleanup_every == 0:
                            gc.collect()
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                except RuntimeError as exc:
                    if _is_pin_memory_error(exc) and pin_memory:
                        set_pin_memory_ok(False)
                        pin_memory = False
                        continue
                    if _is_oom_error(exc) and chunk_size > 1:
                        chunk_size = max(1, chunk_size // 2)
                        print(f"[WARN] OOM detected, reducing chunk_size to {chunk_size}")
                        continue
                    raise
                finally:
                    completed_videos += finalize_video(
                        memmap, current_tmp, current_path, written_frames
                    )
                break

            _debug("[DEBUG] Video feature precompute done")

        elif modal == 'audio':
            _debug("[DEBUG] Precomputing audio features")
            self.model = self.bundle.get_model()
            self.model.eval().to(self.device)
            # 先过滤掉已经生成特征的视频项
            filtered_data = [
                item for item in self.data_list
                if not os.path.exists(self._get_feature_path(item['id'],modal=modal))
            ]
            _debug(f"[DEBUG] Audio features pending: {len(filtered_data)}")
            pin_memory = self.device.type == "cuda" and is_pin_memory_ok()
            batch_size = self.precompute_batch_size
            while True:
                filtered_data = [
                    item for item in self.data_list
                    if not os.path.exists(self._get_feature_path(item['id'],modal=modal))
                ]
                if not filtered_data:
                    break
                dataset = WaveformDataset(filtered_data, dataset_root = self.dataset_root, sample_rate = self.bundle.sample_rate)
                workers = self.num_workers or 0
                if os.name == "nt":
                    workers = max(4, min(workers or 4, 8))

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
                            feature_path = self._get_feature_path(audio_key, modal=modal)
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
                                    raise
                                raise
                            finally:
                                if "waveform_i" in locals():
                                    del waveform_i
                                if "feats" in locals():
                                    del feats

                            processed += 1
                        if processed == 1 or processed % self.precompute_log_every == 0:
                            elapsed = max(1e-6, time.perf_counter() - start_time)
                            rate = processed / elapsed
                            vram = get_cuda_mem_mb(self.device)
                            cpu = get_cpu_rss_mb()
                            queue_size = get_loader_queue_size(loader)
                            vram_text = f"{vram[0]:.0f}/{vram[1]:.0f}MB" if vram else "n/a"
                            cpu_text = f"{cpu:.0f}MB" if cpu is not None else "n/a"
                            queue_text = str(queue_size) if queue_size is not None else "n/a"
                            print(
                                f"[PRECOMPUTE][audio] batch {batch_idx} "
                                f"samples={processed} rate={rate:.2f}/s vram={vram_text} "
                                f"cpu={cpu_text} queue={queue_text}"
                            )
                        if processed % self.precompute_cleanup_every == 0:
                            gc.collect()
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                except RuntimeError as exc:
                    if _is_pin_memory_error(exc) and pin_memory:
                        set_pin_memory_ok(False)
                        pin_memory = False
                        continue
                    if _is_oom_error(exc) and batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        print(f"[WARN] OOM detected, reducing batch_size to {batch_size}")
                        continue
                    raise
                break
            self.model.cpu()
            del self.model
            _debug("[DEBUG] Audio feature precompute done")


    def __len__(self):
        return len(self.data_list)


    def extract_feature(self, idx, feat_stride, feat_offset, modal=None):
        video_item = self.data_list[idx]
        feature_path = self._get_feature_path(video_item['id'], modal=modal)
        feats = torch.from_numpy(np.ascontiguousarray(np.load(feature_path, mmap_mode='r')))

        if modal == 'video':
            feats = feats.transpose(1, 0)
        data_dict = {
            'feats': feats,  # C x T
        }

        return data_dict['feats']

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        video_feats = self.extract_feature(idx, feat_stride, feat_offset, modal='video')
        
        video_dict = {
            'video_id': video_item['id'],
            'feats': video_feats,  # C x T
            'fps': video_item['fps'],
            'duration': video_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames
        }

        return video_dict
