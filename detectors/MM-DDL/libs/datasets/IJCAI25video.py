import os
import gc
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

Transformers = [
    'CLIP-16',
    'XCLIP-16',
]

def choose_CLIP_model(clip_model):
    match clip_model:
        case 'CLIP-16':
            return CLIPVisionModel.from_pretrained(".weights/clip-vit-base-patch16")
        case 'XCLIP-16':
            return XCLIPVisionModel.from_pretrained(".weights/xclip-base-patch16")


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
        video_path = os.path.join(self.dataset_root, video_item["id"])
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

@register_dataset("ijcai25video")
class IJCAI25Video(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
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
        # file path
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
        self.cache_dir = os.path.join('./.cached_data/', f'ijcai25_{clip_model}_feats')
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

        dict_db = []  
        split = 'val' if self.is_training == False else 'train'
        for json_path in tqdm(file_paths, desc='Loading Dataset'):
            with open(json_path, 'r') as f:
                data = json.load(f)
                if data["split"] != split:
                    continue  # 根据split过滤
                if 'fake_visual' in data["modify_type"]:
                    segments = data.get("visual_fake_segments", []), # 音频伪造片段
                    segments = np.array(segments, dtype=np.float32).reshape(-1, 2)
                    labels = np.zeros(len(segments), dtype=np.int64)
                else:
                    segments = np.empty((0, 2))
                    labels = np.empty((0, 1))

                # 构建样本信息
                video_id = 'train_data/' + data["file"] if split == 'train' else 'val_data/' + data["file"]
                dict_db.append({
                    'id': video_id,
                    'fps': 25,
                    'duration': data["video_frames"] / 25.0,
                    'frame_count': int(data.get("video_frames", 0)),
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
        """Chunked feature precompute with streaming writes."""

        def pad_tensor_to_multiple_of_8(x):
            t = x.size(0)
            remainder = t % 8
            if remainder == 0:
                return x
            pad_size = 8 - remainder
            pad_tensor = x[-1].unsqueeze(0).repeat(pad_size, 1, 1, 1)
            return torch.cat([x, pad_tensor], dim=0)

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

        def log_status(tag, batch_idx, processed_batches, completed_videos, start_time, loader):
            elapsed = max(1e-6, time.perf_counter() - start_time)
            rate = processed_batches / elapsed
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
                f"samples={processed_batches} videos={completed_videos} "
                f"rate={rate:.2f}/s vram={vram_text} cpu={cpu_text} queue={queue_text}"
            )

        def run_precompute(chunk_size, pin_memory):
            pending = [
                item for item in self.data_list
                if not os.path.exists(self._get_feature_path(item["id"]))
            ]
            if not pending:
                return "ok"

            workers = self.num_workers or 0
            if os.name == "nt":
                workers = max(4, min(workers or 4, 8))

            dataset = VideoChunkDataset(pending, self.dataset_root, chunk_size)
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
            processed_batches = 0
            start_time = time.perf_counter()
            skipped_videos = set()

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

                    if video_id in skipped_videos:
                        continue

                    if current_video != video_id:
                        completed_videos += finalize_video(
                            memmap, current_tmp, current_path, written_frames
                        )
                        memmap = None
                        written_frames = 0
                        current_video = video_id
                        current_path = self._get_feature_path(video_id)
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
                            if chunk_size > 1:
                                return "oom"
                            skipped_videos.add(video_id)
                            with open("videos.txt", "a") as f:
                                f.write(f"GPU memory is insufficient: {video_id}\n")
                            continue
                        raise
                    finally:
                        del frames
                        if "feats" in locals():
                            del feats

                    processed_batches += 1
                    if processed_batches == 1 or processed_batches % self.precompute_log_every == 0:
                        log_status("video", batch_idx, processed_batches, completed_videos, start_time, loader)
                    if processed_batches % self.precompute_cleanup_every == 0:
                        gc.collect()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
            except RuntimeError as exc:
                if _is_pin_memory_error(exc):
                    return "pin_memory"
                if _is_oom_error(exc):
                    return "oom"
                raise
            finally:
                completed_videos += finalize_video(
                    memmap, current_tmp, current_path, written_frames
                )

            return "ok"

        self.model = choose_CLIP_model(self.clip_model)
        self.model.eval().to(self.device)

        pin_memory = self.device.type == "cuda" and is_pin_memory_ok()
        chunk_size = self.precompute_chunk_size
        while True:
            result = run_precompute(chunk_size, pin_memory)
            if result == "ok":
                break
            if result == "pin_memory" and pin_memory:
                set_pin_memory_ok(False)
                pin_memory = False
                continue
            if result == "oom" and chunk_size > 1:
                chunk_size = max(1, chunk_size // 2)
                print(f"[WARN] OOM detected, reducing chunk_size to {chunk_size}")
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

        feature_path = self._get_feature_path(video_item['id'])
        feats = torch.from_numpy(np.ascontiguousarray(np.load(feature_path, mmap_mode='r')))

        feats = feats.transpose(1, 0)
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride

        segments = torch.from_numpy(video_item['segments'] * video_item['fps'] / feat_stride - feat_offset)
        labels = torch.from_numpy(video_item['labels'])

        data_dict = {
            'video_id': video_item['id'],
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
