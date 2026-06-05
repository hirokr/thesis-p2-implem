import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FGI_ROOT = REPO_ROOT / "detectors" / "FGI"

DEFAULT_DATASET_FILES = {
    "av1": "av1.metadata.json",
    "dfdc": "dfdc.metadata.json",
    "faceavceleb": "faceavceleb.metadata.json",
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _norm_path(path_value):
    if not path_value:
        return path_value
    path_value = os.path.expanduser(path_value)
    return os.path.normpath(path_value)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_default_split(name):
    if name == "av1":
        return "val"
    if name == "dfdc":
        return "test"
    if name == "lavdf":
        return "test"
    return None


def _resolve_path(path_value, dataset_root=None):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return _norm_path(path_value)
    if dataset_root:
        return _norm_path(os.path.join(dataset_root, path_value))
    return _norm_path(path_value)


def _load_av1(metadata_path):
    items = []
    for row in _load_json(metadata_path):
        path_value = _resolve_path(row.get("file"))
        modify_type = row.get("modify_type", "")
        label = "REAL" if modify_type == "real" else "FAKE"
        items.append((path_value, label, row.get("split")))
    return items


def _load_dfdc(metadata_path, dataset_root):
    items = []
    data = _load_json(metadata_path)
    for rel_path, meta in data.items():
        path_value = _resolve_path(rel_path, os.path.join(dataset_root, "dfdc"))
        label = meta.get("label", "").upper() or "UNKNOWN"
        items.append((path_value, label, meta.get("split")))
    return items


def _load_faceavceleb(metadata_path):
    items = []
    for row in _load_json(metadata_path):
        path_value = _resolve_path(row.get("file"))
        method = (row.get("method") or "").lower()
        label = "REAL" if method == "real" else "FAKE"
        items.append((path_value, label, row.get("split")))
    return items


def _load_faceforensics(metadata_path):
    items = []
    for row in _load_json(metadata_path):
        path_value = _resolve_path(row.get("file"))
        label = (row.get("label") or "").upper() or "UNKNOWN"
        items.append((path_value, label, row.get("split")))
    return items


def _load_lavdf(metadata_path):
    items = []
    for row in _load_json(metadata_path):
        path_value = _resolve_path(row.get("file"))
        split = row.get("split")
        is_fake = bool(row.get("modify_video") or row.get("modify_audio") or row.get("n_fakes"))
        label = "FAKE" if is_fake else "REAL"
        items.append((path_value, label, split))
    return items


def _filter_by_split(items, split_mode, dataset_name):
    if split_mode and split_mode != "auto":
        return [item for item in items if item[2] == split_mode] or items
    preferred = _dataset_default_split(dataset_name)
    if not preferred:
        return items
    filtered = [item for item in items if item[2] == preferred]
    return filtered or items


def _stage_videos(items, dest_root, max_videos=None):
    _ensure_dir(dest_root)
    real_dir = os.path.join(dest_root, "real")
    fake_dir = os.path.join(dest_root, "fake")
    _ensure_dir(real_dir)
    _ensure_dir(fake_dir)

    staged = []
    missing = []
    used = set()
    total = 0

    for path_value, label, _ in items:
        if max_videos is not None and total >= max_videos:
            break
        if not path_value or not os.path.exists(path_value):
            missing.append(path_value)
            continue
        base = os.path.basename(path_value)
        if base in used:
            base = f"{total:06d}_{base}"
        used.add(base)
        target_dir = real_dir if label == "REAL" else fake_dir
        target_path = os.path.join(target_dir, base)
        if not os.path.exists(target_path):
            try:
                os.link(path_value, target_path)
            except OSError:
                shutil.copy2(path_value, target_path)
        staged.append((target_path, label))
        total += 1

    return staged, missing


def _write_test_split_csv(work_dir, csv_name):
    test_root = os.path.join(work_dir, "test", "pytmp")
    real_root = os.path.join(test_root, "real")
    fake_root = os.path.join(test_root, "fake")
    csv_path = os.path.join(work_dir, csv_name)

    rows = []
    for label, root in ("fake", fake_root), ("real", real_root):
        if not os.path.exists(root):
            continue
        for video_id in os.listdir(root):
            video_dir = os.path.join(root, video_id)
            if not os.path.isdir(video_dir):
                continue
            for chunk in os.listdir(video_dir):
                chunk_dir = os.path.join(video_dir, chunk)
                if not os.path.isdir(chunk_dir):
                    continue
                audio_file = os.path.join(video_dir, f"{chunk}.wav")
                rel_chunk = os.path.relpath(chunk_dir, work_dir)
                rel_audio = os.path.relpath(audio_file, work_dir)
                rows.append([rel_chunk, rel_audio, label])

    _ensure_dir(work_dir)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    return csv_path, len(rows)


def _run_command(args, cwd):
    result = subprocess.run(args, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(args))


def _compute_results_folder(model_path, dataset_flag):
    parts = model_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        parts = ["model", "model"]
    test_split = "balance" if dataset_flag == "fakeavceleb" else "imbalance"
    return os.path.join("test_results", parts[1], parts[-1][:-8], test_split)


def _parse_auc(output_path):
    if not os.path.exists(output_path):
        return None
    auc_value = None
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "auc =" in line:
                try:
                    auc_value = float(line.strip().split("=")[-1])
                except ValueError:
                    continue
    return auc_value


def main():
    parser = argparse.ArgumentParser(description="Run FGI evaluation across dataSubset datasets.")
    parser.add_argument("--base_dataset_folder", default=r"C:\\t309\\dataSubset", help="Folder with metadata json files")
    parser.add_argument("--dataset_root", default=r"C:\\t309\\dataset", help="Root folder holding datasets")
    parser.add_argument("--work_root", default=r"C:\\t309\\results\\fgi_custom", help="Where to write staged data")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASET_FILES.keys()))
    parser.add_argument("--model_path", default="detectors/FGI/model_best_epoch99.pth.tar")
    parser.add_argument("--with_att", action="store_true", help="Enable attention flag during testing")
    parser.add_argument("--residual_conn", action="store_true", help="Enable residual connection flag during testing")
    parser.add_argument("--split", default="auto", help="Split filter: auto, train, test, val, or all")
    parser.add_argument("--max_videos", type=int, default=None, help="Max videos per dataset")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--fakeavceleb_crop_face", action="store_true", help="Crop face for FakeAVCeleb (default no crop)")
    args = parser.parse_args()

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    base_dataset_folder = _norm_path(args.base_dataset_folder)
    dataset_root = _norm_path(args.dataset_root)
    work_root = _norm_path(args.work_root)

    summary_rows = [["dataset", "staged", "missing", "csv_rows", "auc", "results_folder"]]

    model_path_posix = args.model_path.replace("\\", "/")

    for name in datasets:
        metadata_file = DEFAULT_DATASET_FILES.get(name)
        if not metadata_file:
            print(f"[WARN] Unknown dataset name: {name}")
            continue
        metadata_path = os.path.join(base_dataset_folder, metadata_file)
        if not os.path.exists(metadata_path):
            print(f"[WARN] Metadata missing: {metadata_path}")
            continue

        if name == "av1":
            items = _load_av1(metadata_path)
        elif name == "dfdc":
            items = _load_dfdc(metadata_path, dataset_root)
        elif name == "faceavceleb":
            items = _load_faceavceleb(metadata_path)
        else:
            print(f"[WARN] No loader for dataset: {name}")
            continue

        if args.split == "all":
            filtered = items
        else:
            filtered = _filter_by_split(items, args.split, name)

        work_dir = os.path.join(work_root, name)
        test_root = os.path.join(work_dir, "test")

        staged, missing = _stage_videos(filtered, test_root, args.max_videos)
        print(f"[{name}] staged={len(staged)} missing={len(missing)}")

        if not args.skip_preprocess:
            preprocess_cmd = [
                sys.executable,
                str(FGI_ROOT / "pre-process.py"),
                "--out_dir",
                test_root,
            ]
            if name == "faceavceleb" and not args.fakeavceleb_crop_face:
                preprocess_cmd.append("--dont_crop_face")
            _run_command(preprocess_cmd, cwd=str(REPO_ROOT))

        csv_name = "test_balance_fakeavceleb_split.csv" if name == "faceavceleb" else "test_imbalance_split.csv"
        csv_path, csv_rows = _write_test_split_csv(work_dir, csv_name)
        print(f"[{name}] wrote {csv_rows} rows to {csv_path}")

        auc_value = None
        results_folder = ""

        if not args.skip_test:
            dataset_flag = "fakeavceleb" if name == "faceavceleb" else "dfdc"
            test_cmd = [
                sys.executable,
                "detectors/FGI/my_train.py",
                "--out_dir",
                work_dir,
                "--test",
                model_path_posix,
                "--dataset",
                dataset_flag,
            ]
            if args.with_att:
                test_cmd.append("--with_att")
            if args.residual_conn:
                test_cmd.append("--residual_conn")
            _run_command(test_cmd, cwd=str(REPO_ROOT))

            results_folder = _compute_results_folder(model_path_posix, dataset_flag)
            my_test_cmd = [
                sys.executable,
                "detectors/FGI/my_test.py",
                "--folder",
                results_folder,
                "--dataset",
                dataset_flag,
            ]
            _run_command(my_test_cmd, cwd=str(REPO_ROOT))

            output_path = os.path.join(results_folder, "output.txt")
            auc_value = _parse_auc(output_path)

        summary_rows.append([
            name,
            str(len(staged)),
            str(len(missing)),
            str(csv_rows),
            "" if auc_value is None else f"{auc_value:.6f}",
            results_folder,
        ])

    summary_path = os.path.join(work_root, "summary.csv")
    _ensure_dir(work_root)
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(summary_rows)

    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
