import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.core import load_config_without_merge
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed


DEFAULT_DATASET_FOLDER = r"C:\t309\dataSubset"
DEFAULT_RESULTS_FILE = r"C:\t309\results\mm_dl\results.md"


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


def _ensure_results_header(path):
    _ensure_dir(os.path.dirname(path))
    if os.path.exists(path):
        return
    header = "| Timestamp | Dataset | Samples | Threshold | ThresholdStrategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | EER | FPR |\n"
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(divider)


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
                if line.startswith("| Timestamp") or line.startswith("| ---"):
                    continue
                parts = [part.strip() for part in line.strip("|").split("|")]
                if len(parts) < 2:
                    continue
                dataset = parts[1]
                if dataset:
                    completed.add(dataset.lower())
    except OSError:
        return set()

    return completed


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


def _resolve_ckpt(ckpt_path, epoch):
    if ckpt_path.endswith(".pth.tar"):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint folder not found: {ckpt_path}")
    if epoch > 0:
        ckpt_file = os.path.join(ckpt_path, f"epoch_{epoch:03d}.pth.tar")
    else:
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_path, "*.pth.tar")))
        ckpt_file = ckpt_files[-1] if ckpt_files else None
    if not ckpt_file or not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found in: {ckpt_path}")
    return ckpt_file


def _clear_cached_data():
    cached_root = os.path.join(PROJECT_ROOT, ".cached_data", "test")
    dict_db = os.path.join(cached_root, "dict_db.json")
    if os.path.exists(dict_db):
        os.remove(dict_db)
    if os.path.exists(cached_root):
        for entry in os.listdir(cached_root):
            full = os.path.join(cached_root, entry)
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
    _ensure_dir(cached_root)


def _load_model_and_loader(cfg, ckpt_file):
    _ = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

    val_dataset = make_dataset(cfg["dataset_name"], is_training=False, **cfg["dataset"])
    val_loader = make_data_loader(
        dataset=val_dataset,
        is_training=False,
        generator=None,
        batch_size=1,
        num_workers=cfg["loader"]["num_workers"],
        shuffle=False,
    )

    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    device = torch.device(cfg["devices"][0])
    model = model.to(device)
    if device.type != "cpu":
        model = torch.nn.DataParallel(model, device_ids=cfg["devices"])
    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict_ema"])
    del checkpoint
    model.eval()

    return model, val_loader


def _infer_scores(model, val_loader):
    scores = {}
    for audio_list in val_loader:
        with torch.no_grad():
            outputs = model(audio_list)
        for i, _ in enumerate(audio_list):
            segments = outputs[i]["segments"].detach().cpu().numpy()
            segment_scores = outputs[i]["scores"].detach().cpu().numpy()
            max_score = 0.0 if len(segment_scores) == 0 else float(np.max(segment_scores))
            video_id = outputs[i]["video_id"]
            scores[video_id] = {
                "score": max_score,
                "segment_count": len(segments),
            }
    return scores


def _prepare_label_map(items):
    label_map = {}
    collisions = set()
    for item in items:
        basename = os.path.basename(item["video_path"])
        if basename in label_map:
            collisions.add(basename)
            continue
        label_map[basename] = item["label"]
    return label_map, collisions


def _evaluate_dataset(dataset_name, items, dataset_root, run_config, run_ckpt, run_epoch, run_name, args):
    if not items:
        print(f"[WARN] No items for dataset '{dataset_name}'.")
        return

    label_map, collisions = _prepare_label_map(items)
    if collisions:
        print(f"[WARN] {len(collisions)} duplicate basenames in '{dataset_name}', skipping those labels.")

    _clear_cached_data()

    cfg = load_config_without_merge(run_config)
    cfg["dataset"]["dataset_root"] = dataset_root
    if args.device:
        cfg["devices"] = [args.device]
    cfg["opt"]["learning_rate"] *= len(cfg["devices"])
    cfg["loader"]["num_workers"] *= len(cfg["devices"])
    cfg["dataset"]["devices"] = cfg["devices"]
    cfg["dataset"]["num_workers"] = cfg["loader"]["num_workers"]

    ckpt_file = _resolve_ckpt(run_ckpt, run_epoch)
    model, val_loader = _load_model_and_loader(cfg, ckpt_file)
    predictions = _infer_scores(model, val_loader)

    matched_scores = []
    matched_labels = []
    missing = 0
    for item in items:
        basename = os.path.basename(item["video_path"])
        if basename in collisions:
            continue
        pred = predictions.get(basename)
        if pred is None:
            missing += 1
            continue
        matched_scores.append(pred["score"])
        matched_labels.append(item["label"])

    if not matched_scores:
        print(f"[WARN] No matched predictions for dataset '{dataset_name}'.")
        return

    scores = np.array(matched_scores, dtype=np.float64)
    labels = np.array(matched_labels, dtype=np.int32)
    threshold = _select_threshold(scores, labels, args.threshold_strategy, args.threshold)
    metrics = _compute_metrics(scores, labels, threshold)
    result_name = f"{dataset_name}:{run_name}" if run_name else dataset_name
    _append_results(args.results_file, result_name, metrics, threshold, len(labels), args.threshold_strategy)

    if missing > 0:
        print(f"[WARN] Missing predictions for {missing} samples in '{dataset_name}'.")


def _collect_missing_inputs(config_path, ckpt_path):
    missing = []
    if not os.path.exists(config_path):
        missing.append(f"Config not found: {config_path}")
    if ckpt_path.endswith(".pth.tar"):
        if not os.path.exists(ckpt_path):
            missing.append(f"Checkpoint not found: {ckpt_path}")
    else:
        if not os.path.isdir(ckpt_path):
            missing.append(f"Checkpoint folder not found: {ckpt_path}")
    return missing


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Run MM-DDL evaluation across datasets")
    parser.add_argument("--base_dataset_folder", default=DEFAULT_DATASET_FOLDER, help="Folder with metadata JSON files")
    parser.add_argument("--results_file", default=DEFAULT_RESULTS_FILE, help="Path to append results markdown")
    parser.add_argument("--datasets", default="av1,dfdc,faceavceleb,faceforensics,lavdf", help="Comma list of datasets to run")
    parser.add_argument("--split", default=None, help="Optional split filter (e.g., train, val, test)")
    parser.add_argument("--dfdc_root", default=r"C:\t309\dataset\dfdc", help="Root folder for DFDC videos")
    parser.add_argument("--config", default="configs_test/ijcai25video-CLIP16.yaml", help="MM-DDL test config")
    parser.add_argument("--ckpt", default="ckpt/ijcai25video-CLIP16", help="Checkpoint file or folder")
    parser.add_argument("--epoch", type=int, default=-1, help="Checkpoint epoch (folder mode)")
    parser.add_argument("--configs", default=None, help="Comma list of configs for multi-run")
    parser.add_argument("--ckpts", default=None, help="Comma list of checkpoints for multi-run")
    parser.add_argument("--epochs", default=None, help="Comma list of epochs for multi-run")
    parser.add_argument("--names", default=None, help="Comma list of run names for results tagging")
    parser.add_argument("--device", default=None, help="Override device (e.g., cuda:0 or cpu)")
    parser.add_argument("--threshold_strategy", choices=["fixed", "youden", "f1"], default="f1", help="Threshold strategy")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for fixed strategy")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and exit without processing")
    parser.add_argument(
        "--dry_run_write_results",
        action="store_true",
        help="When used with --dry_run, create results markdown header",
    )
    parser.add_argument("--max_items", type=int, default=None, help="Limit items per dataset for debugging")
    args = parser.parse_args()

    args.base_dataset_folder = _resolve_path(args.base_dataset_folder, PROJECT_ROOT)
    args.results_file = _resolve_path(args.results_file, PROJECT_ROOT)
    args.dfdc_root = _resolve_path(args.dfdc_root, PROJECT_ROOT)
    args.config = _resolve_path(args.config, PROJECT_ROOT)
    args.ckpt = _resolve_path(args.ckpt, PROJECT_ROOT)

    dataset_map = {
        "av1": os.path.join(args.base_dataset_folder, "av1.metadata.json"),
        "dfdc": os.path.join(args.base_dataset_folder, "dfdc.metadata.json"),
        "faceavceleb": os.path.join(args.base_dataset_folder, "faceavceleb.metadata.json"),
        "faceforensics": os.path.join(args.base_dataset_folder, "faceforensics.metadata.json"),
        "lavdf": os.path.join(args.base_dataset_folder, "lavdf.metadata.json"),
    }

    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]

    def _split_csv(value):
        return [entry.strip() for entry in value.split(",") if entry.strip()]

    if args.configs or args.ckpts or args.epochs or args.names:
        configs = _split_csv(args.configs or "") or [args.config]
        ckpts = _split_csv(args.ckpts or "") or [args.ckpt]
        epochs = _split_csv(args.epochs or "") or [str(args.epoch)]
        names = _split_csv(args.names or "") or [""]
    else:
        configs = [args.config]
        ckpts = [args.ckpt]
        epochs = [str(args.epoch)]
        names = [""]

    if len(epochs) == 1 and len(configs) > 1:
        epochs = epochs * len(configs)
    if not (len(configs) == len(ckpts) == len(epochs)):
        raise ValueError("configs, ckpts, and epochs must have the same number of entries")
    if names and len(names) == 1 and len(configs) > 1:
        names = names * len(configs)
    if names and len(names) != len(configs):
        raise ValueError("names must be empty or have the same number of entries as configs")

    runs = []
    for config_path, ckpt_path, epoch_value, run_name in zip(configs, ckpts, epochs, names):
        resolved_config = _resolve_path(config_path, PROJECT_ROOT)
        resolved_ckpt = _resolve_path(ckpt_path, PROJECT_ROOT)
        runs.append((resolved_config, resolved_ckpt, int(epoch_value), run_name))

    missing = []
    for resolved_config, resolved_ckpt, _, _ in runs:
        missing.extend(_collect_missing_inputs(resolved_config, resolved_ckpt))
    if args.dry_run:
        if args.dry_run_write_results and not missing:
            _ensure_results_header(args.results_file)
            print(f"[DRY RUN] Results header created at {args.results_file}.")
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

    completed = _load_completed_datasets(args.results_file)
    for run_config, run_ckpt, run_epoch, run_name in runs:
        for dataset_name in selected:
            dataset_key = dataset_name.lower()
            result_key = f"{dataset_key}:{run_name}" if run_name else dataset_key
            if result_key in completed:
                print(f"[SKIP] Results already recorded for '{result_key}' in {args.results_file}.")
                continue
            if dataset_key not in dataset_map:
                print(f"[WARN] Unknown dataset '{dataset_name}', skipping.")
                continue
            metadata_path = dataset_map[dataset_key]
            if not os.path.exists(metadata_path):
                print(f"[WARN] Metadata not found for '{dataset_name}': {metadata_path}")
                continue

            if dataset_key == "av1":
                items, root = _load_av1m(metadata_path, args.split)
            elif dataset_key == "dfdc":
                items, root = _load_dfdc(metadata_path, args.dfdc_root, args.split)
            elif dataset_key == "faceavceleb":
                items, root = _load_fakeavceleb(metadata_path, args.split)
            elif dataset_key == "faceforensics":
                items, root = _load_faceforensics(metadata_path, args.split)
            elif dataset_key == "lavdf":
                items, root = _load_lavdf(metadata_path, args.split)
            else:
                items, root = [], ""

            if not items:
                print(f"[WARN] No items for dataset '{dataset_name}'.")
                continue

            if args.max_items:
                items = items[: args.max_items]

            dataset_root = root or os.path.dirname(items[0]["video_path"])
            _evaluate_dataset(dataset_key, items, dataset_root, run_config, run_ckpt, run_epoch, run_name, args)


if __name__ == "__main__":
    main()
