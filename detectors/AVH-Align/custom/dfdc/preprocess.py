import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

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

try:
	from ...av_hubert.avhubert.preparation.align_mouth import (
		landmarks_interpolate,
		crop_patch,
		write_video_ffmpeg,
	)
except Exception:
	try:
		from avhubert.preparation.align_mouth import (
			landmarks_interpolate,
			crop_patch,
			write_video_ffmpeg,
		)
	except Exception:
		import importlib.util

		align_mouth_path = AVH_HUBERT_ROOT / "avhubert" / "preparation" / "align_mouth.py"
		spec = importlib.util.spec_from_file_location("avh_align_mouth", align_mouth_path)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		landmarks_interpolate = module.landmarks_interpolate
		crop_patch = module.crop_patch
		write_video_ffmpeg = module.write_video_ffmpeg

import deepfake_feature_extraction as feature_extraction
import eval as eval_runner


base_dataset_folder = r"C:\t309\dataSubset"
results_file = r"C:\t309\results\avh_aligned\result.md"

# Backward compatibility of np.float and np.int
np.float = np.float64
np.int = np.int_

# Defaults from AVH-Align preprocessing
DEFAULT_FACE_PREDICTOR_PATH = "content/data/misc/shape_predictor_68_face_landmarks.dat"
DEFAULT_MEAN_FACE_PATH = "content/data/misc/20words_mean_face.npy"
STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]


def _load_json(path):
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


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


def _resolve_ffmpeg_path(explicit_path=None):
	if explicit_path:
		return explicit_path
	return shutil.which("ffmpeg") or "ffmpeg"


def _validate_inputs(args, dataset_map):
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
	if missing:
		raise FileNotFoundError("\n".join(missing))

	for name, path in dataset_map.items():
		if not os.path.exists(path):
			print(f"[WARN] Metadata not found for '{name}': {path}")


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
):
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
	except Exception:
		print(f"Failed to read video: {input_path}")
		return False

	frames = np.array([frame for frame in videogen])
	landmarks = [_detect_landmark(frame, detector, predictor) for frame in frames]
	preprocessed_landmarks = landmarks_interpolate(landmarks)

	try:
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
	except Exception:
		print(f"Failed to preprocess video: {input_path}; passing whole video")
		rois = frames[..., ::-1]

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
		modify_type = row.get("modify_type", "")
		label = 0 if modify_type == "real" else 1
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
		category = row.get("type", "")
		label = 0 if category == "RealVideo-RealAudio" else 1
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
		label = _normalize_label(info.get("label"))
		items.append({"video_path": video_path, "label": label, "split": split, "rel_path": rel_path})

	return items, dataset_root


def _select_threshold(scores, labels, strategy, fixed_threshold):
	if len(np.unique(labels)) < 2:
		return fixed_threshold
	if strategy == "fixed":
		return fixed_threshold
	fpr, tpr, thresholds = roc_curve(labels, scores)
	if strategy == "youden":
		j_scores = tpr - fpr
		return thresholds[int(np.argmax(j_scores))]
	if strategy == "f1":
		best_f1 = -1.0
		best_threshold = thresholds[0]
		for threshold in thresholds:
			preds = (scores >= threshold).astype(int)
			value = f1_score(labels, preds, zero_division=0)
			if value > best_f1:
				best_f1 = value
				best_threshold = threshold
		return best_threshold
	return fixed_threshold


def _compute_metrics(scores, labels, threshold):
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
		fpr, tpr, thresholds = roc_curve(labels, scores)
		fnr = 1 - tpr
		eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
		eer = float((fpr[eer_index] + fnr[eer_index]) / 2)
		fpr_at_threshold = float(np.interp(threshold, thresholds[::-1], fpr[::-1]))

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
	header = "| Timestamp | Dataset | Samples | Threshold | ThresholdStrategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | EER | FPR |\n"
	divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
	row = (
		f"| {datetime.now().isoformat(timespec='seconds')} | {dataset_name} | {total_count} | {threshold:.6f} | {threshold_strategy} "
		f"| {metrics['Accuracy']:.6f} | {metrics['Precision']:.6f} | {metrics['Recall']:.6f} | {metrics['F1']:.6f} "
		f"| {metrics['ROC_AUC']:.6f} | {metrics['PR_AUC']:.6f} | {metrics['EER']:.6f} | {metrics['FPR']:.6f} |\n"
	)

	if not os.path.exists(path):
		with open(path, "w", encoding="utf-8") as handle:
			handle.write(header)
			handle.write(divider)
			handle.write(row)
		return

	with open(path, "a", encoding="utf-8") as handle:
		handle.write(row)


def _preprocess_items(items, output_root, face_predictor_path, mean_face_path, ffmpeg_path, max_workers):
	if not items:
		return

	with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
			)
			futures[future] = item["video_path"]

		for future in as_completed(futures):
			video_path = futures[future]
			try:
				result = future.result()
				if not result:
					print(f"[WARN] Failed to preprocess: {video_path}")
			except Exception as exc:
				print(f"[ERROR] Exception while preprocessing {video_path}: {exc}")


def _extract_features(items, preproc_root, feature_root, model, transform, trimmed):
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
			feature_audio, feature_vid, feature_multimodal = feature_extraction.extract_features(
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


def _evaluate(items, feature_root, checkpoint_path, dataset_name, threshold_strategy, threshold_value):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	try:
		fusion_model_weights = torch.load(checkpoint_path, map_location=device, weights_only=False)
	except TypeError:
		fusion_model_weights = torch.load(checkpoint_path, map_location=device)

	fusion_model = eval_runner.FusionModel().to(device)
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
		score = eval_runner.process_video(data, fusion_model, device)
		outputs.append(score)
		ground_truths.append(item["label"])

	outputs = np.array(outputs, dtype=np.float64)
	ground_truths = np.array(ground_truths, dtype=np.int32)
	if len(outputs) == 0:
		raise RuntimeError(f"No features available for evaluation on {dataset_name}.")

	threshold = _select_threshold(outputs, ground_truths, threshold_strategy, threshold_value)
	metrics = _compute_metrics(outputs, ground_truths, threshold)
	return metrics, threshold, len(ground_truths)


def _run_dataset(dataset_name, items, dataset_root, args, model, transform):
	preproc_root = os.path.join(args.preprocessed_root, dataset_name)
	feature_root = os.path.join(args.features_root, dataset_name)

	_preprocess_items(
		items,
		preproc_root,
		args.face_predictor_path,
		args.mean_face_path,
		args.ffmpeg_path,
		args.max_workers,
	)
	_extract_features(items, preproc_root, feature_root, model, transform, args.trimmed)
	metrics, threshold, total_count = _evaluate(
		items,
		feature_root,
		args.checkpoint_path,
		dataset_name,
		args.threshold_strategy,
		args.threshold,
	)
	_append_results(args.results_file, dataset_name, metrics, threshold, total_count, args.threshold_strategy)


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
	parser.add_argument("--avhubert_ckpt", default="self_large_vox_433h.pt", help="AVHubert checkpoint path")
	parser.add_argument("--face_predictor_path", default=DEFAULT_FACE_PREDICTOR_PATH, help="Dlib face predictor")
	parser.add_argument("--mean_face_path", default=DEFAULT_MEAN_FACE_PATH, help="Mean face landmarks")
	parser.add_argument("--ffmpeg_path", default=None, help="Explicit ffmpeg path")
	parser.add_argument("--max_workers", type=int, default=8, help="Parallel workers for preprocessing")
	parser.add_argument("--trimmed", action="store_true", help="Trim audio to starting silence")
	parser.add_argument("--threshold_strategy", choices=["fixed", "youden", "f1"], default="f1", help="Threshold strategy")
	parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for fixed strategy")
	args = parser.parse_args()

	args.ffmpeg_path = _resolve_ffmpeg_path(args.ffmpeg_path)
	args.base_dataset_folder = _resolve_path(args.base_dataset_folder, PROJECT_ROOT)
	args.results_file = _resolve_path(args.results_file, PROJECT_ROOT)
	args.dfdc_root = _resolve_path(args.dfdc_root, PROJECT_ROOT)
	args.preprocessed_root = _resolve_path(args.preprocessed_root, PROJECT_ROOT)
	args.features_root = _resolve_path(args.features_root, PROJECT_ROOT)
	args.checkpoint_path = _resolve_path(args.checkpoint_path, PROJECT_ROOT)
	args.avhubert_ckpt = _resolve_path(args.avhubert_ckpt, PROJECT_ROOT)
	args.face_predictor_path = _resolve_path(args.face_predictor_path, PROJECT_ROOT)
	args.mean_face_path = _resolve_path(args.mean_face_path, PROJECT_ROOT)

	dataset_map = {
		"av1": os.path.join(args.base_dataset_folder, "av1.metadata.json"),
		"dfdc": os.path.join(args.base_dataset_folder, "dfdc.metadata.json"),
		"faceavceleb": os.path.join(args.base_dataset_folder, "faceavceleb.metadata.json"),
		"faceforensics": os.path.join(args.base_dataset_folder, "faceforensics.metadata.json"),
		"lavdf": os.path.join(args.base_dataset_folder, "lavdf.metadata.json"),
	}

	_validate_inputs(args, dataset_map)

	selected = [name.strip() for name in args.datasets.split(",") if name.strip()]

	model, task = feature_extraction.load_model(args.avhubert_ckpt)
	transform = feature_extraction.load_transforms(task)

	for dataset_name in selected:
		if dataset_name not in dataset_map:
			print(f"[WARN] Unknown dataset '{dataset_name}', skipping.")
			continue
		metadata_path = dataset_map[dataset_name]
		if not os.path.exists(metadata_path):
			print(f"[WARN] Metadata not found for '{dataset_name}': {metadata_path}")
			continue

		if dataset_name == "av1":
			items, _ = _load_av1m(metadata_path, args.split)
		elif dataset_name == "dfdc":
			items, _ = _load_dfdc(metadata_path, args.dfdc_root, args.split)
		elif dataset_name == "faceavceleb":
			items, _ = _load_fakeavceleb(metadata_path, args.split)
		elif dataset_name == "faceforensics":
			items, _ = _load_faceforensics(metadata_path, args.split)
		elif dataset_name == "lavdf":
			items, _ = _load_lavdf(metadata_path, args.split)
		else:
			items = []

		if not items:
			print(f"[WARN] No items for dataset '{dataset_name}'.")
			continue

		_run_dataset(dataset_name, items, None, args, model, transform)


if __name__ == "__main__":
	main()