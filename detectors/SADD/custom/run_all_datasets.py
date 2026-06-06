r"""Run SADD evaluation across metadata-driven datasets and append summary metrics.

This runner:
- reads the metadata files in C:\t309\dataSubset
- preprocesses each video once into a reusable cache
- reuses cached chunks on later runs
- supports --dataset for a single dataset or all datasets sequentially
- supports --dry_run to limit each dataset to 10 videos
- writes one markdown row per dataset to C:\t309\results\sadd\sadd.md

Evaluation convention:
- fake = 1
- real = 0
- SADD mean audio-visual dissimilarity is used as the score, so higher means
  more fake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SADD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SADD_DIR.parents[1]
FGI_ROOT = REPO_ROOT / "detectors" / "FGI"

for path in (SADD_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torchvision import transforms

from augmentation import Normalize, Scale, ToTensor
from model import Audio_RNN, Raw_Audio_RNN
from utils import min_max_normalize


DATA_ROOT = REPO_ROOT / "dataSubset"
RESULTS_ROOT = REPO_ROOT / "results" / "sadd"
RESULTS_FILE = RESULTS_ROOT / "sadd.md"
DEFAULT_CHECKPOINT = SADD_DIR / ".weights" / "model_best_epoch50.pth.tar"
AUDIO_WAVEFORM_SAMPLES = 48000

SUPPORTED_DATASETS = ("av1", "dfdc", "faceavceleb")
REAL_LABEL = 0
FAKE_LABEL = 1


@dataclass(frozen=True)
class MetadataItem:
    dataset: str
    video_id: str
    label: int  # 1 = fake, 0 = real
    video_path: Path
    raw_entry: Dict[str, object]


@dataclass(frozen=True)
class ChunkItem:
    dataset: str
    video_id: str
    chunk_id: str
    label: int  # 1 = fake, 0 = real
    chunk_dir: Path
    audio_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SADD on metadata-driven datasets")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all",) + SUPPORTED_DATASETS,
        help="Dataset to run. Default: all datasets sequentially.",
    )
    parser.add_argument(
        "--data_root",
        default=str(DATA_ROOT),
        help="Root folder containing dataSubset metadata and dataset files.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=("gpu", "cpu"),
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--audio_format",
        default="waveform",
        choices=("waveform", "mfcc"),
        help="Audio feature type expected by the checkpoint.",
    )
    parser.add_argument(
        "--half_network",
        action="store_true",
        default=True,
        help="Use the half network checkpoint architecture.",
    )
    parser.add_argument(
        "--full_network",
        dest="half_network",
        action="store_false",
        help="Use the full network architecture instead of half network.",
    )
    parser.add_argument(
        "--net",
        default="resnet18",
        help="Backbone name used when instantiating the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size over preprocessed chunks.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows for stability.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only 10 videos per dataset to validate the pipeline.",
    )
    parser.add_argument(
        "--results_file",
        default=str(RESULTS_FILE),
        help="Markdown file to append summary rows to.",
    )
    parser.add_argument(
        "--cache_root",
        default=str(RESULTS_ROOT / "cache"),
        help="Folder used to store reusable preprocessed chunks.",
    )
    parser.add_argument(
        "--keep_cache",
        dest="keep_cache",
        action="store_true",
        help="Keep cached preprocessed data after the run.",
    )
    parser.add_argument(
        "--no_keep_cache",
        dest="keep_cache",
        action="store_false",
        help="Delete cached preprocessed data after the run.",
    )
    parser.set_defaults(keep_cache=True)
    parser.add_argument(
        "--threshold_strategy",
        default="fixed",
        choices=("fixed", "f1"),
        help="Use a fixed threshold for unbiased test metrics. f1 is rejected because it tunes on the evaluated set.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used when --threshold_strategy fixed.",
    )
    parser.add_argument(
        "--crop_face_av1_dfdc",
        dest="crop_face_av1_dfdc",
        action="store_true",
        help="Force face cropping for AV1 and DFDC.",
    )
    parser.add_argument(
        "--dont_crop_face_av1_dfdc",
        dest="crop_face_av1_dfdc",
        action="store_false",
        help="Disable face cropping for AV1 and DFDC.",
    )
    parser.set_defaults(crop_face_av1_dfdc=True)
    return parser.parse_args()


def load_metadata_items(dataset: str, data_root: Path) -> List[MetadataItem]:
    metadata_map = {
        "av1": data_root / "av1.metadata.json",
        "dfdc": data_root / "dfdc.metadata.json",
        "faceavceleb": data_root / "faceavceleb.metadata.json",
    }

    metadata_path = metadata_map[dataset]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    items: List[MetadataItem] = []

    if isinstance(payload, list):
        for index, entry in enumerate(payload):
            if not isinstance(entry, dict):
                continue
            video_path = resolve_video_path(dataset, data_root, str(index), entry)
            video_id = make_video_id(video_path, index)
            label = resolve_label(dataset, str(index), entry)
            items.append(MetadataItem(dataset, video_id, label, video_path, entry))
    elif isinstance(payload, dict):
        for index, (key, entry) in enumerate(payload.items()):
            if not isinstance(entry, dict):
                continue
            video_path = resolve_video_path(dataset, data_root, key, entry)
            video_id = make_video_id(video_path, index)
            label = resolve_label(dataset, key, entry)
            items.append(MetadataItem(dataset, video_id, label, video_path, entry))
    else:
        raise ValueError(f"Unsupported metadata format in {metadata_path}")

    return items


def resolve_video_path(dataset: str, data_root: Path, key: str, entry: Dict[str, object]) -> Path:
    file_value = entry.get("file")
    if isinstance(file_value, str) and file_value:
        return Path(file_value)

    if dataset == "dfdc":
        return data_root.parent / "dataset" / "dfdc" / key

    if dataset == "faceavceleb":
        if key:
            candidate = Path(key)
            if candidate.is_absolute():
                return candidate
            return data_root.parent / "dataset" / "FakeAVCeleb" / key

    raise ValueError(f"Cannot resolve file path for dataset={dataset}, key={key}")


def resolve_label(dataset: str, key: str, entry: Dict[str, object]) -> int:
    if dataset == "dfdc":
        label = str(entry.get("label", "")).upper()
        if label == "FAKE":
            return FAKE_LABEL
        if label == "REAL":
            return REAL_LABEL
        raise ValueError(f"Unknown DFDC label for {key}: {entry.get('label')!r}")

    if dataset == "av1":
        modify_type = str(entry.get("modify_type", "")).lower()
        if modify_type == "fake":
            return FAKE_LABEL
        if modify_type == "real":
            return REAL_LABEL
        raise ValueError(f"Unknown AV1 modify_type for {key}: {entry.get('modify_type')!r}")

    if dataset == "faceavceleb":
        entry_type = str(entry.get("type", "")).lower()
        method = str(entry.get("method", "")).lower()
        if entry_type == "realvideo-realaudio" or method == "real":
            return REAL_LABEL
        return FAKE_LABEL

    raise ValueError(f"Unsupported dataset: {dataset}")


def make_video_id(video_path: Path, index: int) -> str:
    digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:10]
    stem = sanitize_name(video_path.stem)
    return f"{index:06d}_{stem}_{digest}"


def sanitize_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "sample"


def choose_crop_face(dataset: str, args: argparse.Namespace) -> bool:
    if dataset in {"av1", "dfdc"}:
        return bool(args.crop_face_av1_dfdc)
    return False


def select_subset(items: Sequence[MetadataItem], dry_run: bool) -> List[MetadataItem]:
    if not dry_run:
        return list(items)

    selected: List[MetadataItem] = []
    per_label_target = 5
    per_label_counts = {REAL_LABEL: 0, FAKE_LABEL: 0}

    for item in items:
        if per_label_counts[item.label] < per_label_target:
            selected.append(item)
            per_label_counts[item.label] += 1
        if len(selected) >= 10:
            break

    if len(selected) < 10:
        for item in items:
            if item not in selected:
                selected.append(item)
            if len(selected) >= 10:
                break

    return selected


def log_skip(item: MetadataItem, reason: str) -> None:
    print(f"Skipping {item.video_id}: {reason}")


def is_missing_audio_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "does not contain any stream" in lowered
        or "matches no streams" in lowered
        or "no audio" in lowered
    )


def preprocess_item(item: MetadataItem, dataset_cache_root: Path, crop_face: bool) -> None:
    run_pipeline_module = load_run_pipeline_module()

    label_name = label_to_name(item.label)
    done_marker = dataset_cache_root / "pytmp" / label_name / item.video_id / ".sadd_done"
    if done_marker.exists():
        return

    if not item.video_path.exists():
        log_skip(item, f"missing video file: {item.video_path}")
        return

    run_pipeline_module.scene_detect = make_full_video_scene_detect(run_pipeline_module)
    try:
        run_pipeline_module.run_pipeline(
            str(dataset_cache_root),
            str(item.video_path),
            item.video_id,
            label_name,
            crop_face=crop_face,
        )
    except RuntimeError as exc:
        if is_missing_audio_error(str(exc)):
            log_skip(item, f"missing audio stream in {item.video_path}")
            return
        raise

    crop_dir = dataset_cache_root / "pycrop" / label_name / item.video_id
    if not crop_dir.exists():
        raise RuntimeError(f"Cropping did not produce output: {crop_dir}")

    cropped_videos = sorted(
        path
        for path in crop_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    )
    if not cropped_videos:
        log_skip(item, f"no cropped clips found in {crop_dir}")
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.write_text(
            json.dumps(
                {
                    "video_path": str(item.video_path),
                    "crop_face": crop_face,
                    "chunks": 0,
                    "skipped": "no cropped clips",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    chunk_root = dataset_cache_root / "pytmp" / label_name / item.video_id
    chunk_root.mkdir(parents=True, exist_ok=True)

    frames_dir = crop_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    audio_path = crop_dir / "audio.wav"

    crop_video_path = cropped_videos[0]
    run_ffmpeg([
        "-y",
        "-i",
        str(crop_video_path),
        "-qscale:v",
        "2",
        "-threads",
        "1",
        "-f",
        "image2",
        str(frames_dir / "%06d.jpg"),
    ])
    try:
        run_ffmpeg([
            "-y",
            "-i",
            str(crop_video_path),
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "48000",
            str(audio_path),
        ])
    except RuntimeError as exc:
        if is_missing_audio_error(str(exc)):
            log_skip(item, f"missing audio stream in {crop_video_path}")
            return
        raise

    if not audio_path.exists():
        log_skip(item, f"missing audio output: {audio_path}")
        return

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if len(frame_files) < 30:
        raise RuntimeError(f"Not enough frames for chunking in {frames_dir} ({len(frame_files)} frames)")

    for frame_num in range(0, len(frame_files), 30):
        if frame_num + 30 > len(frame_files):
            continue

        chunk_id = f"{frame_num // 30:05d}"
        video_chunk_dir = chunk_root / chunk_id
        video_chunk_dir.mkdir(parents=True, exist_ok=True)

        for i in range(frame_num + 1, frame_num + 31):
            src = frames_dir / f"{i:06d}.jpg"
            dst = video_chunk_dir / f"{i:06d}.jpg"
            shutil.copy2(src, dst)

        chunk_audio = chunk_root / f"{chunk_id}.wav"
        start_time = frame_num / 30.0
        end_time = (frame_num + 30) / 30.0
        run_ffmpeg([
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            f"{start_time:.3f}",
            "-to",
            f"{end_time:.3f}",
            str(chunk_audio),
        ])

    done_marker.write_text(
        json.dumps(
            {
                "video_path": str(item.video_path),
                "crop_face": crop_face,
                "chunks": len(list(chunk_root.glob("[0-9][0-9][0-9][0-9][0-9]"))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_run_pipeline_module():
    install_scenedetect_stubs()
    import importlib.util

    module_path = SADD_DIR / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("sadd_run_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    sys.path.insert(0, str(FGI_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(FGI_ROOT):
            sys.path.pop(0)

    return module


def install_scenedetect_stubs() -> None:
    if "scenedetect" in sys.modules:
        return

    scenedetect_module = types.ModuleType("scenedetect")
    video_manager_module = types.ModuleType("scenedetect.video_manager")
    scene_manager_module = types.ModuleType("scenedetect.scene_manager")
    frame_timecode_module = types.ModuleType("scenedetect.frame_timecode")
    stats_manager_module = types.ModuleType("scenedetect.stats_manager")
    detectors_module = types.ModuleType("scenedetect.detectors")

    class FrameTimecode:
        def __init__(self, frame_num: int = 0, *args, **kwargs):
            self.frame_num = int(frame_num)

    class VideoManager:
        def __init__(self, video_paths):
            self.video_paths = video_paths

        def get_base_timecode(self):
            return FrameTimecode(0)

        def set_downscale_factor(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def get_current_timecode(self):
            return FrameTimecode(0)

    class SceneManager:
        def __init__(self, stats_manager):
            self.stats_manager = stats_manager

        def add_detector(self, detector):
            return None

        def detect_scenes(self, frame_source=None):
            return None

        def get_scene_list(self, base_timecode):
            return []

    class StatsManager:
        pass

    class ContentDetector:
        def __init__(self, *args, **kwargs):
            return None

    video_manager_module.VideoManager = VideoManager
    scene_manager_module.SceneManager = SceneManager
    frame_timecode_module.FrameTimecode = FrameTimecode
    stats_manager_module.StatsManager = StatsManager
    detectors_module.ContentDetector = ContentDetector

    scenedetect_module.video_manager = video_manager_module
    scenedetect_module.scene_manager = scene_manager_module
    scenedetect_module.frame_timecode = frame_timecode_module
    scenedetect_module.stats_manager = stats_manager_module
    scenedetect_module.detectors = detectors_module

    sys.modules["scenedetect"] = scenedetect_module
    sys.modules["scenedetect.video_manager"] = video_manager_module
    sys.modules["scenedetect.scene_manager"] = scene_manager_module
    sys.modules["scenedetect.frame_timecode"] = frame_timecode_module
    sys.modules["scenedetect.stats_manager"] = stats_manager_module
    sys.modules["scenedetect.detectors"] = detectors_module


def make_full_video_scene_detect(run_pipeline_module):
    def _scene_detect():
        frames_dir = Path(run_pipeline_module.frames_dir) / run_pipeline_module.reference
        frame_files = sorted(frames_dir.glob("*.jpg"))
        total_frames = len(frame_files)
        return [
            (
                run_pipeline_module.FrameTimecode(0),
                run_pipeline_module.FrameTimecode(total_frames),
            )
        ]

    return _scene_detect


def run_ffmpeg(args: Sequence[str]) -> None:
    command = ["ffmpeg", *args]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed with exit code %s\nCOMMAND: %s\nSTDOUT:\n%s\nSTDERR:\n%s"
            % (completed.returncode, " ".join(command), completed.stdout, completed.stderr)
        )


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunk_items: Sequence[ChunkItem], audio_format: str):
        self.chunk_items = list(chunk_items)
        self.audio_format = audio_format
        self.transform = transforms.Compose(
            [
                Scale(size=(224, 224)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.chunk_items)

    def __getitem__(self, index: int):
        item = self.chunk_items[index]
        frame_paths = sorted(item.chunk_dir.glob("*.jpg"))
        if len(frame_paths) != 30:
            raise RuntimeError(f"Expected 30 frames in {item.chunk_dir}, found {len(frame_paths)}")

        frames = [load_rgb_image(path) for path in frame_paths]
        transformed = self.transform(frames)
        channels, height, width = transformed[0].shape
        video_tensor = torch.stack(transformed, 0).view(1, 30, channels, height, width).transpose(1, 2)

        sample_rate, audio = wavfile.read(item.audio_path)
        if self.audio_format == "waveform":
            if audio.size == 0:
                normalized_audio = np.zeros(AUDIO_WAVEFORM_SAMPLES, dtype=np.float32)
            else:
                normalized_audio = min_max_normalize(audio, int(audio.min()), int(audio.max()))
            if normalized_audio.shape[0] < AUDIO_WAVEFORM_SAMPLES:
                padded_audio = np.zeros(AUDIO_WAVEFORM_SAMPLES, dtype=normalized_audio.dtype)
                padded_audio[: normalized_audio.shape[0]] = normalized_audio
                normalized_audio = padded_audio
            elif normalized_audio.shape[0] > AUDIO_WAVEFORM_SAMPLES:
                normalized_audio = normalized_audio[:AUDIO_WAVEFORM_SAMPLES]
            audio_tensor = torch.from_numpy(normalized_audio.astype(float)).float()
        else:
            try:
                import python_speech_features

                mfcc = python_speech_features.mfcc(audio, sample_rate, nfft=2048)
                mfcc = np.stack([np.array(chunk) for chunk in zip(*mfcc)])
            except ModuleNotFoundError:
                import librosa

                mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sample_rate, n_mfcc=13, n_fft=2048)
            audio_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0).astype(float)).float()

        label_tensor = torch.LongTensor([item.label])
        return video_tensor, audio_tensor, label_tensor, item.video_id


def load_rgb_image(path: Path) -> Image.Image:
    with path.open("rb") as handle:
        with Image.open(handle) as image:
            return image.convert("RGB")


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.audio_format == "waveform":
        model = Raw_Audio_RNN(
            img_dim=224,
            network=args.net,
            num_layers_in_fc_layers=1024,
            dropout=0.5,
            full_network=not args.half_network,
        )
    else:
        model = Audio_RNN(
            img_dim=224,
            network=args.net,
            num_layers_in_fc_layers=1024,
            dropout=0.5,
            full_network=not args.half_network,
        )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def discover_chunk_items(dataset_cache_root: Path, selected_video_ids: Iterable[str]) -> List[ChunkItem]:
    selected_set = set(selected_video_ids)
    chunk_items: List[ChunkItem] = []
    for label_name, label_value in (("fake", FAKE_LABEL), ("real", REAL_LABEL)):
        label_root = dataset_cache_root / "pytmp" / label_name
        if not label_root.exists():
            continue
        for video_dir in sorted(path for path in label_root.iterdir() if path.is_dir() and path.name in selected_set):
            for chunk_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
                audio_path = video_dir / f"{chunk_dir.name}.wav"
                if not audio_path.exists():
                    continue
                chunk_items.append(
                    ChunkItem(
                        dataset=dataset_cache_root.name,
                        video_id=video_dir.name,
                        chunk_id=chunk_dir.name,
                        label=label_value,
                        chunk_dir=chunk_dir,
                        audio_path=audio_path,
                    )
                )
    return chunk_items


def evaluate_dataset(
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
) -> Dict[str, object]:
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    dataset_cache_root = cache_root / dataset
    dataset_cache_root.mkdir(parents=True, exist_ok=True)

    items = select_subset(load_metadata_items(dataset, data_root), args.dry_run)
    if not items:
        raise RuntimeError(f"No metadata items found for dataset {dataset}")

    selected_counts = count_items_by_label(items)
    crop_face = choose_crop_face(dataset, args)
    for item in items:
        preprocess_item(item, dataset_cache_root, crop_face=crop_face)

    chunk_items = discover_chunk_items(dataset_cache_root, [item.video_id for item in items])
    if not chunk_items:
        raise RuntimeError(f"No chunk items discovered for dataset {dataset}")

    dataset_obj = ChunkDataset(chunk_items, audio_format=args.audio_format)
    loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
    )

    video_score_sum: Dict[str, float] = {}
    video_chunk_count: Dict[str, int] = {}
    video_target: Dict[str, int] = {}

    with torch.no_grad():
        for video_batch, audio_batch, target_batch, video_ids in loader:
            video_batch = video_batch.to(device)
            audio_batch = audio_batch.to(device)
            target_batch = target_batch.to(device)
            sadd_target_batch = 1 - target_batch

            vid_class, aud_class, loss1, vid_out, aud_out, mean_separation_loss, kl_loss = model(
                video_batch,
                audio_batch,
                0.99,
                sadd_target_batch,
                calc_mean_separation_loss=False,
                calc_kl_loss=False,
            )

            distance_batch = torch.nn.PairwiseDistance(p=2)(
                vid_out[0].view(video_batch.size(0), -1),
                aud_out[0].view(video_batch.size(0), -1),
            )
            distances = distance_batch.detach().cpu().tolist()
            targets = target_batch.view(-1).detach().cpu().tolist()

            for video_id, distance, target in zip(video_ids, distances, targets):
                if video_id not in video_score_sum:
                    video_score_sum[video_id] = 0.0
                    video_chunk_count[video_id] = 0
                    video_target[video_id] = int(target)
                video_score_sum[video_id] += float(distance)
                video_chunk_count[video_id] += 1

    video_scores: Dict[str, float] = {
        video_id: video_score_sum[video_id] / max(1, video_chunk_count[video_id])
        for video_id in video_score_sum
    }

    ordered_video_ids = sorted(video_scores)
    y_true = np.array([video_target[video_id] for video_id in ordered_video_ids], dtype=int)
    y_score = np.array([video_scores[video_id] for video_id in ordered_video_ids], dtype=float)
    threshold = resolve_threshold(
        strategy=args.threshold_strategy,
        fixed_threshold=args.threshold,
    )
    y_pred = (y_score >= threshold).astype(int)

    evaluated_counts = count_labels(y_true)
    missing_skipped_counts = {
        label: selected_counts[label] - evaluated_counts[label]
        for label in (REAL_LABEL, FAKE_LABEL)
    }
    log_dataset_counts(
        dataset=dataset,
        selected_counts=selected_counts,
        evaluated_counts=evaluated_counts,
        missing_skipped_counts=missing_skipped_counts,
        chunk_count=len(chunk_items),
    )
    metrics = compute_metrics(y_true, y_score, y_pred)
    metrics.update(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset,
            "videos": int(len(y_true)),
            "chunks": int(len(chunk_items)),
            "threshold": float(threshold),
            "threshold_strategy": args.threshold_strategy,
        }
    )

    save_dataset_cache_index(dataset_cache_root, items, chunk_items, metrics)
    return metrics


def resolve_threshold(strategy: str, fixed_threshold: float) -> float:
    if strategy == "fixed":
        return float(fixed_threshold)
    if strategy == "f1":
        raise ValueError(
            "threshold_strategy=f1 tunes on evaluation labels and produces biased metrics. "
            "Use --threshold_strategy fixed with a threshold chosen before evaluation."
        )
    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    has_both_classes = len(np.unique(y_true)) > 1
    try:
        roc_auc = float(roc_auc_score(y_true, y_score)) if has_both_classes else float("nan")
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_score)) if has_both_classes else float("nan")
    except ValueError:
        pr_auc = float("nan")

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else float("nan")

    eer = calculate_eer(y_true, y_score) if has_both_classes else float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "eer": float(eer),
        "fpr": fpr,
    }


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr_values, tpr_values, _ = roc_curve(y_true, y_score, pos_label=FAKE_LABEL)
    fnr_values = 1.0 - tpr_values
    index = int(np.nanargmin(np.abs(fpr_values - fnr_values)))
    return float((fpr_values[index] + fnr_values[index]) / 2.0)


def label_to_name(label: int) -> str:
    if label == FAKE_LABEL:
        return "fake"
    if label == REAL_LABEL:
        return "real"
    raise ValueError(f"Unknown binary label: {label!r}")


def count_items_by_label(items: Sequence[MetadataItem]) -> Dict[int, int]:
    counts = {REAL_LABEL: 0, FAKE_LABEL: 0}
    for item in items:
        counts[item.label] += 1
    return counts


def count_labels(labels: np.ndarray) -> Dict[int, int]:
    return {
        REAL_LABEL: int(np.sum(labels == REAL_LABEL)),
        FAKE_LABEL: int(np.sum(labels == FAKE_LABEL)),
    }


def log_dataset_counts(
    dataset: str,
    selected_counts: Dict[int, int],
    evaluated_counts: Dict[int, int],
    missing_skipped_counts: Dict[int, int],
    chunk_count: int,
) -> None:
    print(
        f"{dataset} counts: "
        f"selected real={selected_counts[REAL_LABEL]} fake={selected_counts[FAKE_LABEL]}; "
        f"evaluated videos real={evaluated_counts[REAL_LABEL]} fake={evaluated_counts[FAKE_LABEL]}; "
        f"missing/skipped real={missing_skipped_counts[REAL_LABEL]} fake={missing_skipped_counts[FAKE_LABEL]}; "
        f"evaluated chunks={chunk_count}"
    )


def save_dataset_cache_index(
    dataset_cache_root: Path,
    items: Sequence[MetadataItem],
    chunk_items: Sequence[ChunkItem],
    metrics: Dict[str, object],
) -> None:
    summary_dir = dataset_cache_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metrics": metrics,
        "videos": [
            {
                "video_id": item.video_id,
                "video_path": str(item.video_path),
                "label": item.label,
            }
            for item in items
        ],
        "chunks": [
            {
                "video_id": item.video_id,
                "chunk_id": item.chunk_id,
                "label": item.label,
                "chunk_dir": str(item.chunk_dir),
                "audio_path": str(item.audio_path),
            }
            for item in chunk_items
        ],
    }
    (summary_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_results_row(results_file: Path, metrics: Dict[str, object]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| Model       | Dataset     | Samples/Videos | Threshold | ThresholdStrategy | Accuracy | Precision | Recall |     F1 | ROC_AUC | PR_AUC |    EER |    FPR |\n"
    )
    separator = (
        "| ----------- | ----------- | -------------: | --------: | ----------------- | -------: | --------: | -----: | -----: | ------: | -----: | -----: | -----: |\n"
    )
    needs_header = not results_file.exists() or results_file.stat().st_size == 0
    if not needs_header:
        with results_file.open("r", encoding="utf-8") as handle:
            needs_header = handle.readline() != header

    if needs_header:
        results_file.write_text(
            header + separator,
            encoding="utf-8",
        )

    row = (
        f"| SADD | {metrics['dataset']} | {metrics['videos']} | "
        f"{metrics['threshold']:.6f} | {metrics['threshold_strategy']} | "
        f"{metrics['accuracy']:.6f} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | "
        f"{metrics['f1']:.6f} | {metrics['roc_auc']:.6f} | {metrics['pr_auc']:.6f} | "
        f"{metrics['eer']:.6f} | {metrics['fpr']:.6f} |\n"
    )

    existing = results_file.read_text(encoding="utf-8")
    needs_newline = not existing.endswith("\n")
    with results_file.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(row)


def cleanup_cache(dataset_cache_root: Path) -> None:
    if dataset_cache_root.exists():
        shutil.rmtree(dataset_cache_root)


def run() -> None:
    args = parse_args()
    if args.threshold_strategy == "f1":
        raise SystemExit(
            "--threshold_strategy f1 tunes on the evaluated labels and produces biased metrics; "
            "use --threshold_strategy fixed with a preselected --threshold."
        )
    datasets = list(SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset])
    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")

    model = build_model(args, device)
    rows: List[Dict[str, object]] = []

    for dataset in datasets:
        print(f"Running {dataset} on {device} ...")
        metrics = evaluate_dataset(dataset, args, device, model)
        rows.append(metrics)
        append_results_row(Path(args.results_file), metrics)
        print(
            f"{dataset}: videos={metrics['videos']} chunks={metrics['chunks']} threshold={metrics['threshold']:.6f} "
            f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}"
        )

        if not args.keep_cache:
            cleanup_cache(Path(args.cache_root) / dataset)

    print(f"Appended {len(rows)} row(s) to {args.results_file}")


if __name__ == "__main__":
    run()
