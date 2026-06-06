import argparse
import importlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import torch
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AVH_HUBERT_ROOT = PROJECT_ROOT / "av_hubert"
for path in (AVH_HUBERT_ROOT / "avhubert", AVH_HUBERT_ROOT / "fairseq", PROJECT_ROOT):
	path_str = str(path)
	if path_str not in sys.path:
		sys.path.insert(0, path_str)

landmarks_interpolate = None
crop_patch = None
write_video_ffmpeg = None
feature_extraction = None
eval_runner = None


base_dataset_folder = r"C:\t309\dataSubset"
results_file = r"C:\t309\results\avh_aligned\result.md"

# Backward compatibility of np.float and np.int
np.float = np.float64
np.int = np.int_

# Defaults from AVH-Align preprocessing
DEFAULT_FACE_PREDICTOR_PATH = "av_hubert/avhubert/content/data/misc/shape_predictor_68_face_landmarks.dat"
DEFAULT_MEAN_FACE_PATH = "av_hubert/avhubert/content/data/misc/20words_mean_face.npy"
STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]


def _load_json(path):
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _load_align_mouth_helpers():
	global landmarks_interpolate, crop_patch, write_video_ffmpeg
	if landmarks_interpolate is not None and crop_patch is not None and write_video_ffmpeg is not None:
		return
	try:
		from ...av_hubert.avhubert.preparation.align_mouth import (
			landmarks_interpolate as loaded_landmarks_interpolate,
			crop_patch as loaded_crop_patch,
			write_video_ffmpeg as loaded_write_video_ffmpeg,
		)
	except Exception:
		try:
			from avhubert.preparation.align_mouth import (
				landmarks_interpolate as loaded_landmarks_interpolate,
				crop_patch as loaded_crop_patch,
				write_video_ffmpeg as loaded_write_video_ffmpeg,
			)
		except Exception:
			align_mouth_path = AVH_HUBERT_ROOT / "avhubert" / "preparation" / "align_mouth.py"
			if not align_mouth_path.exists():
				raise FileNotFoundError(f"AV-HuBERT align_mouth.py not found: {align_mouth_path}")
			spec = importlib.util.spec_from_file_location("avh_align_mouth", align_mouth_path)
			module = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(module)
			loaded_landmarks_interpolate = module.landmarks_interpolate
			loaded_crop_patch = module.crop_patch
			loaded_write_video_ffmpeg = module.write_video_ffmpeg
	landmarks_interpolate = loaded_landmarks_interpolate
	crop_patch = loaded_crop_patch
	write_video_ffmpeg = loaded_write_video_ffmpeg


def _load_feature_extraction():
	global feature_extraction
	if feature_extraction is None:
		feature_extraction = importlib.import_module("deepfake_feature_extraction")
	return feature_extraction


def _load_eval_runner():
	global eval_runner
	if eval_runner is None:
		eval_runner = importlib.import_module("eval")
	return eval_runner


def _normalize_label(label):
	if isinstance(label, str):
		label_upper = label.strip().upper()
		if label_upper == "REAL":
			return 0
		if label_upper == "FAKE":
			return 1
	if isinstance(label, (bool, np.bool_)):
		return int(label)
	if isinstance(label, (int, float, np.integer, np.floating)):
		return int(label > 0)
	return int(bool(label))


def _label_from_av1(row):
	modify_type = str(row.get("modify_type", "")).strip().lower()
	if modify_type == "real":
		return 0
	if modify_type == "fake":
		return 1
	raise ValueError(f"Unknown av1 modify_type: {row.get('modify_type')!r}")


def _label_from_dfdc(info):
	label = str(info.get("label", "")).strip().upper()
	if label == "REAL":
		return 0
	if label == "FAKE":
		return 1
	raise ValueError(f"Unknown DFDC label: {info.get('label')!r}")


def _label_from_faceavceleb(row):
	method = str(row.get("method", "")).strip().lower()
	category = str(row.get("type", "")).strip()
	return 0 if method == "real" or category == "RealVideo-RealAudio" else 1


def _guess_common_root(paths):
	normalized = [os.path.normpath(p) for p in paths if p]
	return os.path.commonpath(normalized) if normalized else ""


def _ensure_dir(path):
	os.makedirs(path, exist_ok=True)


def _resolve_path(path_value, base_dir):
	if not path_value:
		return path_value
	path_value = os.path.expanduser(path_value)
	if os.path.isabs(path_value):
		return path_value
	return os.path.abspath(os.path.join(base_dir, path_value))


def _replace_ext(path_value, new_ext):
	root, _ = os.path.splitext(path_value)
	return root + new_ext


def _append_failure_log(log_path, video_path, reason, error=None, extra=None):
	if not log_path:
		return
	_ensure_dir(os.path.dirname(log_path))
	stamp = datetime.now().isoformat(timespec="seconds")
	message = f"[{stamp}] {reason} | {video_path}"
	if error:
		message += f" | error={error}"
	if extra:
		message += f" | {extra}"
	with open(log_path, "a", encoding="utf-8") as handle:
		handle.write(message + "\n")


def _center_crop_frames(frames, crop_height, crop_width):
	if frames is None or len(frames) == 0:
		return frames
	_, height, width, _ = frames.shape
	if crop_height > height or crop_width > width:
		return frames
	y_start = (height - crop_height) // 2
	x_start = (width - crop_width) // 2
	return frames[:, y_start : y_start + crop_height, x_start : x_start + crop_width]


def _fallback_mouth_crop(frames, crop_height=96, crop_width=96):
	if frames is None or len(frames) == 0:
		return None

	import cv2

	gray_frames = []
	for frame in frames:
		gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	prev_box = None
	rois = []

	for gray, frame in zip(gray_frames, frames):
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
		if len(faces) > 0:
			x, y, w, h = faces[0]
			prev_box = (x, y, w, h)
		elif prev_box is not None:
			x, y, w, h = prev_box
		else:
			rois.append(None)
			continue

		mouth_y = y + int(h * 0.55)
		mouth_h = int(h * 0.40)
		mouth_x = x + int(w * 0.10)
		mouth_w = int(w * 0.80)

		mouth_y = max(0, min(mouth_y, frame.shape[0] - 1))
		mouth_x = max(0, min(mouth_x, frame.shape[1] - 1))
		mouth_h = max(1, min(mouth_h, frame.shape[0] - mouth_y))
		mouth_w = max(1, min(mouth_w, frame.shape[1] - mouth_x))

		crop = frame[mouth_y : mouth_y + mouth_h, mouth_x : mouth_x + mouth_w]
		crop = cv2.resize(crop, (crop_width, crop_height), interpolation=cv2.INTER_AREA)
		rois.append(crop)

	if not rois or all(r is None for r in rois):
		return None

	valid = [r for r in rois if r is not None]
	if valid:
		last = valid[-1]
		rois = [r if r is not None else last for r in rois]

	return np.stack(rois, axis=0)


def _resolve_ffmpeg_path(explicit_path=None):
	if explicit_path:
		return explicit_path
	return shutil.which("ffmpeg") or "ffmpeg"


def _collect_missing_inputs(args, dataset_map):
	missing = []
	if not args.ffmpeg_path:
		missing.append("ffmpeg path is empty.")
	elif os.path.isabs(args.ffmpeg_path):
		if not os.path.exists(args.ffmpeg_path):
			missing.append(f"ffmpeg not found at '{args.ffmpeg_path}'.")
	else:
		if not shutil.which(args.ffmpeg_path):
			missing.append(f"ffmpeg not found on PATH as '{args.ffmpeg_path}'.")
	if not os.path.exists(args.face_predictor_path):
		missing.append(f"face predictor not found: {args.face_predictor_path}")
	if not os.path.exists(args.mean_face_path):
		missing.append(f"mean face landmarks not found: {args.mean_face_path}")
	if not os.path.exists(args.checkpoint_path):
		missing.append(f"fusion checkpoint not found: {args.checkpoint_path}")
	if not os.path.exists(args.avhubert_ckpt):
		missing.append(f"AVHubert checkpoint not found: {args.avhubert_ckpt}")
	for name, path in dataset_map.items():
		if not os.path.exists(path):
			print(f"[WARN] Metadata not found for '{name}': {path}")
	return missing


def _validate_inputs(args, dataset_map):
	missing = _collect_missing_inputs(args, dataset_map)
	if missing:
		raise FileNotFoundError("\n".join(missing))


def _download_file(url, output_path):
	import urllib.request

	_ensure_dir(os.path.dirname(output_path))
	urllib.request.urlretrieve(url, output_path)


def _maybe_download_assets(face_predictor_path, mean_face_path, auto_download):
	if not auto_download:
		return

	predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
	mean_face_url = "https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy"

	if face_predictor_path and not os.path.exists(face_predictor_path):
		import bz2
		import tempfile

		_ensure_dir(os.path.dirname(face_predictor_path))
		with tempfile.NamedTemporaryFile(delete=False, suffix=".bz2") as temp_handle:
			bz2_path = temp_handle.name
		_download_file(predictor_url, bz2_path)
		with bz2.open(bz2_path, "rb") as source, open(face_predictor_path, "wb") as dest:
			shutil.copyfileobj(source, dest)
		try:
			os.remove(bz2_path)
		except OSError:
			pass

	if mean_face_path and not os.path.exists(mean_face_path):
		_download_file(mean_face_url, mean_face_path)


def _detect_landmark(image, detector, predictor):
	import cv2

	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	rects = detector(gray, 1)
	coords = None
	for (_, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		coords = np.zeros((68, 2), dtype=np.int32)
		for i in range(0, 68):
			coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


def _preprocess_video(
	input_video_dir,
	video_filename,
	output_video_dir,
	face_predictor_path,
	mean_face_path,
	ffmpeg_path,
	failure_log,
):
	_load_align_mouth_helpers()
	import cv2
	import dlib
	import skvideo.io

	roi_path = os.path.join(output_video_dir, _replace_ext(video_filename, "_roi.mp4"))
	if os.path.exists(roi_path):
		return True

	_ensure_dir(output_video_dir)

	input_path = os.path.join(input_video_dir, video_filename)
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(face_predictor_path)
	mean_face_landmarks = np.load(mean_face_path)

	try:
		videogen = skvideo.io.vread(input_path)
	except Exception as exc:
		print(f"Failed to read video: {input_path}")
		_append_failure_log(failure_log, input_path, "vread_failed", error=repr(exc))
		return False

	frames = np.array([frame for frame in videogen])
	landmarks = []
	missing_landmarks = 0
	for frame in frames:
		coords = _detect_landmark(frame, detector, predictor)
		if coords is None:
			missing_landmarks += 1
		landmarks.append(coords)
	try:
		preprocessed_landmarks = landmarks_interpolate(landmarks)
	except Exception as exc:
		_append_failure_log(
			failure_log,
			input_path,
			"landmarks_interpolate_failed",
			error=repr(exc),
			extra=f"missing_landmarks={missing_landmarks}/{len(landmarks)}",
		)
		preprocessed_landmarks = None

	try:
		if preprocessed_landmarks is None:
			rois = _fallback_mouth_crop(frames)
			if rois is None:
				raise ValueError("No valid landmarks to crop")
			print(f"[WARN] Fallback mouth crop used for {input_path}")
		else:
			rois = crop_patch(
				input_path,
				preprocessed_landmarks,
				mean_face_landmarks,
				STABLE_PNTS_IDS,
				STD_SIZE,
				window_margin=12,
				start_idx=48,
				stop_idx=68,
				crop_height=96,
				crop_width=96,
			)
			if rois is None or len(rois) == 0:
				rois = _fallback_mouth_crop(frames)
				if rois is None:
					raise ValueError("No valid mouth crops from crop_patch")
				print(f"[WARN] Fallback mouth crop used for {input_path}")
	except Exception as exc:
		print(f"[ERROR] crop_patch failed for {input_path}: {exc}")
		print(f"Failed to preprocess video: {input_path}; skipping")
		_append_failure_log(
			failure_log,
			input_path,
			"crop_patch_failed",
			error=repr(exc),
			extra=f"missing_landmarks={missing_landmarks}/{len(landmarks)}",
		)
		return False

	audio_fn = os.path.join(output_video_dir, _replace_ext(video_filename, ".wav"))
	write_video_ffmpeg(rois, roi_path, ffmpeg_path)

	import subprocess

	subprocess.run(
		[
			ffmpeg_path,
			"-i",
			input_path,
			"-f",
			"wav",
			"-vn",
			"-y",
			audio_fn,
			"-loglevel",
			"quiet",
		],
		check=False,
	)
	if not os.path.exists(audio_fn):
		duration_sec = max(len(frames) / 25.0, 0.04)
		print(f"[WARN] Audio missing for {input_path}; generating silence")
		subprocess.run(
			[
				ffmpeg_path,
				"-f",
				"lavfi",
				"-i",
				"anullsrc=r=16000:cl=mono",
				"-t",
				f"{duration_sec:.3f}",
				"-q:a",
				"9",
				"-acodec",
				"pcm_s16le",
				"-y",
				audio_fn,
				"-loglevel",
				"quiet",
			],
			check=False,
		)
	return True


def _load_av1m(metadata_path, split_filter=None):
	items = []
	data = _load_json(metadata_path)
	paths = [row.get("file") for row in data if row.get("file")]
	root = _guess_common_root(paths)

	for row in data:
		split = row.get("split")
		if split_filter and split != split_filter:
			continue
		video_path = row.get("file")
		if not video_path:
			continue
		label = _label_from_av1(row)
		rel_path = os.path.relpath(video_path, root) if root else os.path.basename(video_path)
		items.append({"video_path": video_path, "label": label, "split": split, "rel_path": rel_path})

	return items, root


def _load_fakeavceleb(metadata_path, split_filter=None):
	items = []
	data = _load_json(metadata_path)
	paths = [row.get("file") for row in data if row.get("file")]
	root = _guess_common_root(paths)

	for row in data:
		video_path = row.get("file")
		if not video_path:
			continue
		label = _label_from_faceavceleb(row)
		rel_path = os.path.relpath(video_path, root) if root else os.path.basename(video_path)
		items.append({"video_path": video_path, "label": label, "split": None, "rel_path": rel_path})

	return items, root


def _load_faceforensics(metadata_path, split_filter=None):
	items = []
	data = _load_json(metadata_path)
	paths = [row.get("file") for row in data if row.get("file")]
	root = _guess_common_root(paths)

	for row in data:
		video_path = row.get("file")
		if not video_path:
			continue
		label = _normalize_label(row.get("label"))
		rel_path = os.path.relpath(video_path, root) if root else os.path.basename(video_path)
		items.append({"video_path": video_path, "label": label, "split": None, "rel_path": rel_path})

	return items, root


def _load_lavdf(metadata_path, split_filter=None):
	items = []
	data = _load_json(metadata_path)
	paths = [row.get("file") for row in data if row.get("file")]
	root = _guess_common_root(paths)

	for row in data:
		video_path = row.get("file")
		if not video_path:
			continue
		label = 1 if row.get("n_fakes", 0) > 0 else 0
		rel_path = os.path.relpath(video_path, root) if root else os.path.basename(video_path)
		items.append({"video_path": video_path, "label": label, "split": None, "rel_path": rel_path})

	return items, root


def _load_dfdc(metadata_path, dataset_root, split_filter=None):
	items = []
	data = _load_json(metadata_path)
	for rel_path, info in data.items():
		split = info.get("split")
		if split_filter and split != split_filter:
			continue
		video_path = os.path.join(dataset_root, rel_path)
		label = _label_from_dfdc(info)
		items.append({"video_path": video_path, "label": label, "split": split, "rel_path": rel_path})

	return items, dataset_root


def _count_by_label(items):
	counts = {0: 0, 1: 0}
	for item in items:
		counts[int(item["label"])] += 1
	return counts


def _count_labels(labels):
	labels = np.asarray(labels, dtype=np.int32)
	return {0: int(np.sum(labels == 0)), 1: int(np.sum(labels == 1))}


def _log_label_counts(dataset_name, selected_counts, staged_counts, evaluated_counts):
	missing_counts = {
		0: selected_counts[0] - evaluated_counts[0],
		1: selected_counts[1] - evaluated_counts[1],
	}
	print(
		f"[COUNTS] {dataset_name}: "
		f"selected real={selected_counts[0]} fake={selected_counts[1]}; "
		f"staged real={staged_counts[0]} fake={staged_counts[1]}; "
		f"evaluated real={evaluated_counts[0]} fake={evaluated_counts[1]}; "
		f"missing/skipped real={missing_counts[0]} fake={missing_counts[1]}"
	)


def _has_preprocessed_item(item, preproc_root):
	rel_path = item["rel_path"]
	base_dir = os.path.join(preproc_root, os.path.dirname(rel_path))
	name_root = os.path.splitext(os.path.basename(rel_path))[0]
	return (
		os.path.exists(os.path.join(base_dir, name_root + "_roi.mp4"))
		and os.path.exists(os.path.join(base_dir, name_root + ".wav"))
	)


def _select_threshold(scores, labels, strategy, fixed_threshold):
	if strategy == "fixed":
		return float(fixed_threshold)
	if strategy == "f1":
		raise ValueError(
			"threshold_strategy=f1 tunes on the evaluated labels and produces biased metrics. "
			"Use --threshold_strategy fixed with a threshold chosen before evaluation."
		)
	if strategy == "youden":
		raise ValueError(
			"threshold_strategy=youden tunes on the evaluated labels and produces biased metrics. "
			"Use --threshold_strategy fixed with a threshold chosen before evaluation."
		)
	raise ValueError(f"Unknown threshold strategy: {strategy}")


def _compute_eer(labels, scores):
	fpr, tpr, _ = roc_curve(labels, scores)
	fnr = 1 - tpr
	diff = fpr - fnr
	exact = np.where(diff == 0)[0]
	if len(exact) > 0:
		return float(fpr[int(exact[0])])
	crossing = np.where(diff[:-1] * diff[1:] < 0)[0]
	if len(crossing) > 0:
		idx = int(crossing[0])
		x0, x1 = diff[idx], diff[idx + 1]
		weight = -x0 / (x1 - x0)
		return float(fpr[idx] + weight * (fpr[idx + 1] - fpr[idx]))
	idx = int(np.nanargmin(np.abs(diff)))
	return float((fpr[idx] + fnr[idx]) / 2)


def _compute_metrics(scores, labels, threshold):
	scores = np.asarray(scores, dtype=np.float64)
	labels = np.asarray(labels, dtype=np.int32)
	preds = (scores >= threshold).astype(int)

	accuracy = accuracy_score(labels, preds)
	precision = precision_score(labels, preds, zero_division=0)
	recall = recall_score(labels, preds, zero_division=0)
	f1 = f1_score(labels, preds, zero_division=0)
	roc_auc = float("nan")
	pr_auc = float("nan")
	eer = float("nan")
	fpr_at_threshold = float("nan")

	if len(np.unique(labels)) >= 2:
		try:
			roc_auc = roc_auc_score(labels, scores)
		except ValueError:
			roc_auc = float("nan")
		try:
			pr_auc = average_precision_score(labels, scores)
		except ValueError:
			pr_auc = float("nan")
		eer = _compute_eer(labels, scores)

	tn = int(np.sum((labels == 0) & (preds == 0)))
	fp = int(np.sum((labels == 0) & (preds == 1)))
	if fp + tn > 0:
		fpr_at_threshold = float(fp / (fp + tn))

	return {
		"Accuracy": accuracy,
		"Precision": precision,
		"Recall": recall,
		"F1": f1,
		"ROC_AUC": roc_auc,
		"PR_AUC": pr_auc,
		"EER": eer,
		"FPR": fpr_at_threshold,
	}


def _append_results(path, dataset_name, metrics, threshold, total_count, threshold_strategy):
	_ensure_dir(os.path.dirname(path))
	header = (
		"| Model       | Dataset     | Samples/Videos | Threshold | ThresholdStrategy | Accuracy | Precision | Recall |     F1 | ROC_AUC | PR_AUC |    EER |    FPR |\n"
	)
	divider = (
		"| ----------- | ----------- | -------------: | --------: | ----------------- | -------: | --------: | -----: | -----: | ------: | -----: | -----: | -----: |\n"
	)
	row = (
		f"| AVH-Align | {dataset_name} | {total_count} | {threshold:.6f} | {threshold_strategy} "
		f"| {metrics['Accuracy']:.6f} | {metrics['Precision']:.6f} | {metrics['Recall']:.6f} | {metrics['F1']:.6f} "
		f"| {metrics['ROC_AUC']:.6f} | {metrics['PR_AUC']:.6f} | {metrics['EER']:.6f} | {metrics['FPR']:.6f} |\n"
	)

	write_header = not os.path.exists(path) or os.path.getsize(path) == 0
	if not write_header:
		with open(path, "r", encoding="utf-8") as handle:
			write_header = handle.readline() != header

	if write_header:
		with open(path, "w", encoding="utf-8") as handle:
			handle.write(header)
			handle.write(divider)
			handle.write(row)
		return

	with open(path, "a", encoding="utf-8") as handle:
		handle.write(row)


def _load_completed_datasets(results_path):
	if not results_path or not os.path.exists(results_path):
		return set()

	completed = set()
	try:
		with open(results_path, "r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if not line.startswith("|"):
					continue
				if line.startswith("| Model") or line.startswith("| Timestamp") or line.startswith("| ---"):
					continue
				parts = [part.strip() for part in line.strip("|").split("|")]
				if len(parts) < 2:
					continue
				dataset = parts[1]
				if dataset:
					completed.add(dataset)
	except OSError:
		return set()

	return completed


def _preprocess_items(items, output_root, face_predictor_path, mean_face_path, ffmpeg_path, failure_log, max_workers):
	if not items:
		return

	if not max_workers or max_workers <= 1:
		for item in items:
			input_dir = os.path.dirname(item["video_path"])
			filename = os.path.basename(item["video_path"])
			rel_dir = os.path.dirname(item["rel_path"])
			output_dir = os.path.join(output_root, rel_dir)
			try:
				result = _preprocess_video(
					input_dir,
					filename,
					output_dir,
					face_predictor_path,
					mean_face_path,
					ffmpeg_path,
					failure_log,
				)
				if result:
						print(f"[DONE] Preprocessed: {item['video_path']}")
	
			except Exception as exc:
				print(f"[ERROR] Exception while preprocessing {item['video_path']}: {exc}")
		return

	executor_cls = ThreadPoolExecutor if os.name == "nt" else ProcessPoolExecutor
	with executor_cls(max_workers=max_workers) as executor:
		futures = {}
		for item in items:
			input_dir = os.path.dirname(item["video_path"])
			filename = os.path.basename(item["video_path"])
			rel_dir = os.path.dirname(item["rel_path"])
			output_dir = os.path.join(output_root, rel_dir)

			future = executor.submit(
				_preprocess_video,
				input_dir,
				filename,
				output_dir,
				face_predictor_path,
				mean_face_path,
				ffmpeg_path,
				failure_log,
			)
			futures[future] = item["video_path"]

		for future in as_completed(futures):
			video_path = futures[future]
			try:
				result = future.result()
				if result:
						print(f"[DONE] Preprocessed: {video_path}")
				else:
						print(f"[WARN] Failed to preprocess: {video_path}")
			except Exception as exc:
				print(f"[ERROR] Exception while preprocessing {video_path}: {exc}")


def _extract_features(items, preproc_root, feature_root, model, transform, trimmed):
	feature_module = _load_feature_extraction()
	for item in items:
		rel_path = item["rel_path"]
		base_dir = os.path.join(preproc_root, os.path.dirname(rel_path))
		filename = os.path.basename(rel_path)
		name_root = os.path.splitext(filename)[0]
		mouth_roi_path = os.path.join(base_dir, name_root + "_roi.mp4")
		audio_path = os.path.join(base_dir, name_root + ".wav")

		if not os.path.exists(mouth_roi_path) or not os.path.exists(audio_path):
			print(f"[WARN] Missing preprocessed files for {item['video_path']}")
			continue

		try:
			feature_audio, feature_vid, feature_multimodal = feature_module.extract_features(
				model, mouth_roi_path, audio_path, transform, trimmed
			)
		except Exception as exc:
			print(f"[WARN] Unprocessed for file: {mouth_roi_path}; error {exc}")
			continue

		save_dict = {
			"visual": feature_vid,
			"audio": feature_audio,
			"multimodal": feature_multimodal,
		}
		save_path = os.path.join(feature_root, _replace_ext(rel_path, ".npz"))
		_ensure_dir(os.path.dirname(save_path))
		np.savez(save_path, **save_dict)
		print(f"[DONE] Features extracted: {item['video_path']}")


def _evaluate(items, feature_root, checkpoint_path, dataset_name, threshold_strategy, threshold_value):
	eval_module = _load_eval_runner()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	try:
		fusion_model_weights = torch.load(checkpoint_path, map_location=device, weights_only=False)
	except TypeError:
		fusion_model_weights = torch.load(checkpoint_path, map_location=device)

	fusion_model = eval_module.FusionModel().to(device)
	state_dict = fusion_model_weights.get("state_dict") if isinstance(fusion_model_weights, dict) else None
	if state_dict is None:
		state_dict = fusion_model_weights
	fusion_model.load_state_dict(state_dict)
	fusion_model.eval()

	outputs = []
	ground_truths = []
	for item in items:
		feature_path = os.path.join(feature_root, _replace_ext(item["rel_path"], ".npz"))
		if not os.path.exists(feature_path):
			print(f"[WARN] Missing features for {item['video_path']}")
			continue
		data = np.load(feature_path, allow_pickle=True)
		score = eval_module.process_video(data, fusion_model, device)
		score = float(score.detach().cpu().item() if torch.is_tensor(score) else score)
		outputs.append(score)
		ground_truths.append(item["label"])
		print(f"[DONE] Evaluated: {item['video_path']} | Score: {score:.4f}")

	outputs = np.array(outputs, dtype=np.float64)
	ground_truths = np.array(ground_truths, dtype=np.int32)
	if len(outputs) == 0:
		raise RuntimeError(f"No features available for evaluation on {dataset_name}.")

	threshold = _select_threshold(outputs, ground_truths, threshold_strategy, threshold_value)
	metrics = _compute_metrics(outputs, ground_truths, threshold)
	return metrics, threshold, len(ground_truths), _count_labels(ground_truths)


def _run_dataset(dataset_name, items, dataset_root, args, model, transform):
	preproc_root = os.path.join(args.preprocessed_root, dataset_name)
	feature_root = os.path.join(args.features_root, dataset_name)
	selected_counts = _count_by_label(items)

	_preprocess_items(
		items,
		preproc_root,
		args.face_predictor_path,
		args.mean_face_path,
		args.ffmpeg_path,
		args.failure_log,
		args.max_workers,
	)
	staged_items = [item for item in items if _has_preprocessed_item(item, preproc_root)]
	staged_counts = _count_by_label(staged_items)
	_extract_features(items, preproc_root, feature_root, model, transform, args.trimmed)
	try:
		metrics, threshold, total_count, evaluated_counts = _evaluate(
			items,
			feature_root,
			args.checkpoint_path,
			dataset_name,
			args.threshold_strategy,
			args.threshold,
		)
		_log_label_counts(dataset_name, selected_counts, staged_counts, evaluated_counts)
		_append_results(args.results_file, dataset_name, metrics, threshold, total_count, args.threshold_strategy)
	except (RuntimeError, ValueError) as exc:
		_log_label_counts(dataset_name, selected_counts, staged_counts, {0: 0, 1: 0})
		print(f"[WARN] Skipping evaluation for {dataset_name}: {exc}")


def _load_dataset_items(dataset_name, metadata_path, args):
	if dataset_name == "av1":
		return _load_av1m(metadata_path, args.split)[0]
	if dataset_name == "dfdc":
		return _load_dfdc(metadata_path, args.dfdc_root, args.split)[0]
	if dataset_name == "faceavceleb":
		return _load_fakeavceleb(metadata_path, args.split)[0]
	if dataset_name == "faceforensics":
		return _load_faceforensics(metadata_path, args.split)[0]
	if dataset_name == "lavdf":
		return _load_lavdf(metadata_path, args.split)[0]
	return []


def main():
	parser = argparse.ArgumentParser(description="Run AVH-Align preprocess/features/eval across datasets")
	parser.add_argument("--base_dataset_folder", default=base_dataset_folder, help="Folder with metadata JSON files")
	parser.add_argument("--results_file", default=results_file, help="Path to append results markdown")
	parser.add_argument("--datasets", default="av1,dfdc,faceavceleb,faceforensics,lavdf", help="Comma list of datasets to run")
	parser.add_argument("--split", default=None, help="Optional split filter (e.g., train, val, test)")
	parser.add_argument("--dfdc_root", default=r"C:\t309\dataset\dfdc", help="Root folder for DFDC videos")
	parser.add_argument("--preprocessed_root", default=r"C:\t309\results\avh_aligned\preprocessed", help="Output folder for preprocessed ROI/audio")
	parser.add_argument("--features_root", default=r"C:\t309\results\avh_aligned\features", help="Output folder for feature files")
	parser.add_argument("--checkpoint_path", default="checkpoints/AVH-Align_AV1M.pt", help="Fusion model checkpoint")
	parser.add_argument(
		"--avhubert_ckpt",
		default="av_hubert/avhubert/self_large_vox_433h.pt",
		help="AVHubert checkpoint path",
	)
	parser.add_argument("--face_predictor_path", default=DEFAULT_FACE_PREDICTOR_PATH, help="Dlib face predictor")
	parser.add_argument("--mean_face_path", default=DEFAULT_MEAN_FACE_PATH, help="Mean face landmarks")
	parser.add_argument("--ffmpeg_path", default=None, help="Explicit ffmpeg path")
	parser.add_argument("--max_workers", type=int, default=8, help="Parallel workers for preprocessing")
	parser.add_argument(
		"--failure_log",
		default=r"C:\t309\results\avh_aligned\preprocess_failures.log",
		help="Path to log preprocessing failures",
	)
	parser.add_argument("--trimmed", action="store_true", help="Trim audio to starting silence")
	parser.add_argument(
		"--threshold_strategy",
		choices=["fixed", "youden", "f1"],
		default="fixed",
		help="Use fixed for unbiased test metrics. f1/youden are rejected because they tune on evaluated labels.",
	)
	parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for fixed strategy")
	parser.add_argument("--auto_download_assets", action="store_true", help="Download missing landmark/mean-face assets")
	parser.add_argument("--dry_run", action="store_true", help="Validate inputs and exit without processing")
	parser.add_argument("--metadata_label_count_check", action="store_true", help="Print metadata label counts and exit")
	parser.add_argument("--max_items", type=int, default=None, help="Limit items per dataset for debugging")
	parser.add_argument("--skip_completed", action="store_true", help="Skip datasets already present in the results markdown")
	args = parser.parse_args()

	args.ffmpeg_path = _resolve_ffmpeg_path(args.ffmpeg_path)
	if args.ffmpeg_path and (os.path.sep in args.ffmpeg_path or "/" in args.ffmpeg_path):
		args.ffmpeg_path = _resolve_path(args.ffmpeg_path, PROJECT_ROOT)
	args.base_dataset_folder = _resolve_path(args.base_dataset_folder, PROJECT_ROOT)
	args.results_file = _resolve_path(args.results_file, PROJECT_ROOT)
	args.dfdc_root = _resolve_path(args.dfdc_root, PROJECT_ROOT)
	args.preprocessed_root = _resolve_path(args.preprocessed_root, PROJECT_ROOT)
	args.features_root = _resolve_path(args.features_root, PROJECT_ROOT)
	args.checkpoint_path = _resolve_path(args.checkpoint_path, PROJECT_ROOT)
	args.avhubert_ckpt = _resolve_path(args.avhubert_ckpt, PROJECT_ROOT)
	args.face_predictor_path = _resolve_path(args.face_predictor_path, PROJECT_ROOT)
	args.mean_face_path = _resolve_path(args.mean_face_path, PROJECT_ROOT)
	args.failure_log = _resolve_path(args.failure_log, PROJECT_ROOT)

	_maybe_download_assets(args.face_predictor_path, args.mean_face_path, args.auto_download_assets)

	dataset_map = {
		"av1": os.path.join(args.base_dataset_folder, "av1.metadata.json"),
		"dfdc": os.path.join(args.base_dataset_folder, "dfdc.metadata.json"),
		"faceavceleb": os.path.join(args.base_dataset_folder, "faceavceleb.metadata.json"),
		"faceforensics": os.path.join(args.base_dataset_folder, "faceforensics.metadata.json"),
		"lavdf": os.path.join(args.base_dataset_folder, "lavdf.metadata.json"),
	}

	selected = [name.strip() for name in args.datasets.split(",") if name.strip()]

	if args.threshold_strategy in ("f1", "youden"):
		raise ValueError(
			f"--threshold_strategy {args.threshold_strategy} tunes on evaluation labels and produces biased metrics; "
			"use --threshold_strategy fixed with a preselected --threshold."
		)

	if args.metadata_label_count_check:
		for dataset_name in selected:
			if dataset_name not in dataset_map:
				print(f"[WARN] Unknown dataset '{dataset_name}', skipping.")
				continue
			metadata_path = dataset_map[dataset_name]
			if not os.path.exists(metadata_path):
				print(f"[WARN] Metadata not found for '{dataset_name}': {metadata_path}")
				continue
			items = _load_dataset_items(dataset_name, metadata_path, args)
			if args.max_items:
				items = items[: args.max_items]
			counts = _count_by_label(items)
			print(f"[METADATA] {dataset_name}: total={len(items)} real={counts[0]} fake={counts[1]}")
		return

	missing = _collect_missing_inputs(args, dataset_map)
	if args.dry_run:
		if missing:
			print("[DRY RUN] Missing required files:")
			for entry in missing:
				print(f"  - {entry}")
		else:
			print("[DRY RUN] All required files found.")
		print(f"[DRY RUN] Selected datasets: {', '.join(selected)}")
		return

	if missing:
		raise FileNotFoundError("\n".join(missing))

	print(f"CUDA available: {torch.cuda.is_available()}")

	feature_module = _load_feature_extraction()
	model, task = feature_module.load_model(args.avhubert_ckpt)
	transform = feature_module.load_transforms(task)
	completed = {name.lower() for name in _load_completed_datasets(args.results_file)} if args.skip_completed else set()

	for dataset_name in selected:
		if dataset_name.lower() in completed:
			print(f"[SKIP] Results already recorded for '{dataset_name}' in {args.results_file}.")
			continue
		if dataset_name not in dataset_map:
			print(f"[WARN] Unknown dataset '{dataset_name}', skipping.")
			continue
		metadata_path = dataset_map[dataset_name]
		if not os.path.exists(metadata_path):
			print(f"[WARN] Metadata not found for '{dataset_name}': {metadata_path}")
			continue

		items = _load_dataset_items(dataset_name, metadata_path, args)

		if not items:
			print(f"[WARN] No items for dataset '{dataset_name}'.")
			continue

		if args.max_items:
			items = items[: args.max_items]

		_run_dataset(dataset_name, items, None, args, model, transform)


if __name__ == "__main__":
	main()
