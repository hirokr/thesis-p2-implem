import argparse
import json
import os
import random
import subprocess
import sys
import shutil
import tempfile
from datetime import datetime
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score

import torch

# Ensure script runs with detectors/FGI on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def normalize_path(path_str):
    return os.path.normpath(path_str.replace('/', os.sep))


def parse_binary_label(raw_label, fake_values, real_values, context):
    label = str(raw_label or '').strip().lower()
    if label in fake_values:
        return 'fake'
    if label in real_values:
        return 'real'
    raise ValueError(f'Unknown label for {context}: {raw_label!r}')


def load_metadata_entries(dataset_name, metadata_root, raw_root):
    meta_path = os.path.join(metadata_root, f'{dataset_name}.metadata.json')
    if not os.path.exists(meta_path):
        return []

    with open(meta_path, 'r') as f:
        data = json.load(f)

    entries = []

    if dataset_name == 'dfdc':
        base = os.path.join(raw_root, 'dfdc')
        for rel, info in data.items():
            label = parse_binary_label(info.get('label'), {'fake'}, {'real'}, rel)
            split = info.get('split')
            full_path = os.path.join(base, normalize_path(rel))
            entries.append({'path': full_path, 'label': label, 'split': split})

    elif dataset_name == 'faceavceleb':
        for item in data:
            file_path = normalize_path(item.get('file', ''))
            method = item.get('method', '').lower()
            dtype = item.get('type', '').lower()
            label = 'real' if method == 'real' or 'realvideo-realaudio' in dtype else 'fake'
            entries.append({'path': file_path, 'label': label, 'split': None})

    elif dataset_name == 'av1':
        for item in data:
            file_path = normalize_path(item.get('file', ''))
            label = parse_binary_label(item.get('modify_type'), {'fake'}, {'real'}, item.get('file', ''))
            split = item.get('split')
            entries.append({'path': file_path, 'label': label, 'split': split})

    return entries


def select_entries(entries, dry_run=False, max_total=10):
    if not entries:
        return []

    # If a test/val split exists, prefer it
    split_tags = {e.get('split') for e in entries}
    if 'test' in split_tags or 'val' in split_tags:
        entries = [e for e in entries if e.get('split') in ('test', 'val')]

    if not dry_run:
        return entries

    reals = [e for e in entries if e['label'] == 'real']
    fakes = [e for e in entries if e['label'] == 'fake']

    take_real = min(len(reals), max_total // 2)
    take_fake = min(len(fakes), max_total // 2)

    selected = reals[:take_real] + fakes[:take_fake]
    if len(selected) < max_total:
        remaining = [e for e in entries if e not in selected]
        selected += remaining[:max_total - len(selected)]

    return selected


def stage_dataset(entries, stage_dir):
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)

    for label in ['real', 'fake']:
        os.makedirs(os.path.join(stage_dir, label), exist_ok=True)

    staged = 0
    staged_by_label = {'real': 0, 'fake': 0}
    missing_by_label = {'real': 0, 'fake': 0}
    for idx, entry in enumerate(entries):
        src = entry['path']
        if not os.path.exists(src):
            missing_by_label[entry['label']] += 1
            continue
        label = entry['label']
        ext = os.path.splitext(src)[1] or '.mp4'
        dst_name = f'{idx:05d}_{os.path.basename(src)}'
        if not dst_name.lower().endswith(ext.lower()):
            dst_name += ext
        dst = os.path.join(stage_dir, label, dst_name)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        staged += 1
        staged_by_label[label] += 1

    return staged, staged_by_label, missing_by_label

from dataset_3d import deepfake_3d_rawaudio


def collate_rawaudio_fixed(batch, target_len=48000):
    from torch.utils.data.dataloader import default_collate
    fixed = []
    for item in batch:
        if item is None:
            continue
        video_seq, audio_seq, target, audiopath = item
        if audio_seq is None or audio_seq.numel() == 0:
            continue
        if audio_seq.shape[0] > target_len:
            audio_seq = audio_seq[:target_len]
        elif audio_seq.shape[0] < target_len:
            pad = target_len - audio_seq.shape[0]
            audio_seq = torch.nn.functional.pad(audio_seq, (0, pad))
        fixed.append((video_seq, audio_seq, target, audiopath))

    if len(fixed) == 0:
        return [[], [], [], []]

    return default_collate(fixed)


def get_rawaudio_data(transform, args, mode='test'):
    from torch.utils import data
    print('Loading data for "%s" ...' % mode)
    dataset = deepfake_3d_rawaudio(out_dir=args.out_dir, mode=mode,
                                   transform=transform,
                                   vis_min_fake_len=args.vis_min_fake_len, vis_max_fake_len=args.vis_max_fake_len,
                                   aud_min_fake_len=args.aud_min_fake_len, aud_max_fake_len=args.aud_max_fake_len,
                                   using_pseudo_fake=args.using_pseudo_fake, dataset_name=args.dataset)

    sampler = data.RandomSampler(dataset)

    if mode == 'test':
        data_loader = data.DataLoader(dataset,
                          batch_size=1,
                          sampler=sampler,
                          shuffle=False,
                          num_workers=args.num_workers,
                          pin_memory=True,
                          collate_fn=collate_rawaudio_fixed)
    else:
        raise ValueError('only test mode supported in run script')

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader
from model import My_Network


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    diff = fpr - fnr

    exact = np.where(diff == 0)[0]
    if len(exact) > 0:
        idx = exact[0]
        return float(fpr[idx]), float(thresholds[idx])

    crossing = np.where(diff[:-1] * diff[1:] < 0)[0]
    if len(crossing) > 0:
        idx = crossing[0]
        x0, x1 = diff[idx], diff[idx + 1]
        weight = -x0 / (x1 - x0)
        eer = fpr[idx] + weight * (fpr[idx + 1] - fpr[idx])
        thr = thresholds[idx] + weight * (thresholds[idx + 1] - thresholds[idx])
        return float(eer), float(thr)

    idx = np.nanargmin(np.abs(diff))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def resolve_threshold(y_true, y_score, strategy='fixed', fixed_threshold=0.5):
    if strategy == 'fixed':
        return float(fixed_threshold)
    if strategy == 'f1':
        raise ValueError(
            'threshold_strategy=f1 tunes on the evaluation labels and produces biased metrics. '
            'Use --threshold_strategy fixed with a threshold chosen before evaluation.'
        )
    raise ValueError(f'Unknown threshold strategy: {strategy}')


def evaluate_from_pickles(
    pred_dict,
    target_dict,
    num_chunks_dict,
    threshold_strategy='fixed',
    threshold=0.5,
):
    vids = list(pred_dict.keys())
    scores = []
    targets = []
    for v in vids:
        total_score = pred_dict[v]
        n_chunks = num_chunks_dict.get(v, 1)
        mean_score = float(total_score) / float(n_chunks)
        scores.append(mean_score)
        tar = int(target_dict[v])
        # In this codebase test_target: 1 -> fake, 0 -> real
        targets.append(1 if tar == 1 else 0)

    y_score = np.array(scores)
    y_true = np.array(targets)
    if len(y_true) == 0:
        raise ValueError('No valid predictions were collected; cannot compute evaluation metrics.')

    has_both_classes = len(np.unique(y_true)) > 1
    roc_auc = roc_auc_score(y_true, y_score) if has_both_classes else float('nan')
    pr_auc = average_precision_score(y_true, y_score) if has_both_classes else float('nan')

    thr = resolve_threshold(y_true, y_score, strategy=threshold_strategy, fixed_threshold=threshold)
    y_pred = (y_score >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if has_both_classes:
        eer, eer_thr = compute_eer(y_true, y_score)
    else:
        eer, eer_thr = float('nan'), float('nan')

    # FPR at chosen threshold
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else float('nan')

    return {
        'Samples': len(y_true),
        'Videos': len(y_true),
        'Chunks': int(sum(num_chunks_dict.get(v, 1) for v in vids)),
        'Threshold': float(thr),
        'ThresholdStrategy': threshold_strategy,
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1': float(f1),
        'ROC_AUC': float(roc_auc),
        'PR_AUC': float(pr_auc),
        'EER': float(eer),
        'EER_threshold': float(eer_thr),
        'FPR': float(fpr)
    }


def prepare_splits(out_dir, dataset_name, dry_run=False):
    # Build CSV splits directly from the pytmp folder structure that preprocessing created.
    def build_split(root_split_dir):
        rows = []
        for label in ['fake', 'real']:
            split_dir = os.path.join(root_split_dir, label)
            if not os.path.exists(split_dir):
                continue
            for video_id in os.listdir(split_dir):
                video_dir = os.path.join(split_dir, video_id)
                if not os.path.isdir(video_dir):
                    continue
                for entry in os.listdir(video_dir):
                    entry_path = os.path.join(video_dir, entry)
                    if os.path.isdir(entry_path):
                        # chunk folder
                        audio_path = os.path.join(video_dir, entry + '.wav')
                        rows.append([entry_path, audio_path, label])
        return rows

    # Prefer root pytmp if present (pre-process.py output)
    root_pytmp = os.path.join(out_dir, 'pytmp')
    if os.path.exists(root_pytmp):
        test_rows = build_split(root_pytmp)
        train_rows = test_rows
    else:
        train_root = os.path.join(out_dir, 'train', 'pytmp')
        test_root = os.path.join(out_dir, 'test', 'pytmp')
        train_rows = build_split(train_root)
        test_rows = build_split(test_root)

    # write train_split.csv
    train_csv = os.path.join(out_dir, 'train_split.csv')
    with open(train_csv, 'w') as f:
        for r in train_rows:
            f.write(','.join(r) + '\n')

    # If dry_run, keep only first chunk per video for up to 10 videos
    if dry_run:
        limited = []
        seen = {}
        for r in test_rows:
            video_id = os.path.basename(os.path.dirname(r[0]))
            if video_id not in seen and len(seen) >= 10:
                continue
            count = seen.get(video_id, 0)
            if count >= 1:
                continue
            seen[video_id] = count + 1
            limited.append(r)
        test_rows = limited

    # write test split depending on dataset naming the loader expects
    if dataset_name == 'dfdc':
        test_csv = os.path.join(out_dir, 'test_imbalance_split.csv')
    else:
        test_csv = os.path.join(out_dir, 'test_balance_fakeavceleb_split.csv')

    with open(test_csv, 'w') as f:
        for r in test_rows:
            f.write(','.join(r) + '\n')

    return len(test_rows)


def run_preprocess_for_dir(raw_dir, dont_crop_face=False):
    # call pre-process.py with out_dir = raw_dir
    cmd = [sys.executable, os.path.join(HERE, 'pre-process.py'), '--out_dir', raw_dir]
    if dont_crop_face:
        cmd.append('--dont_crop_face')
    subprocess.run(cmd, cwd=HERE, check=True)


def simple_preprocess_for_dir(raw_dir):
    """Lightweight preprocessing for dry-run: split videos into 30-frame chunks and audio snippets using ffmpeg.
    Produces structure: <raw_dir>/pytmp/{real,fake}/{video_id}/{chunk_id}/[frames].jpg and <raw_dir>/pytmp/{real,fake}/{video_id}/{chunk_id}.wav
    """
    for split in ['real', 'fake']:
        src_dir = os.path.join(raw_dir, split)
        if not os.path.exists(src_dir):
            continue
        out_root = os.path.join(raw_dir, 'pytmp', split)
        os.makedirs(out_root, exist_ok=True)

        for video in os.listdir(src_dir):
            if not video.lower().endswith('.mp4'):
                continue
            video_path = os.path.join(src_dir, video)
            video_id = os.path.basename(video)[:-4]
            work_frames = os.path.join(raw_dir, 'tmp_frames', video_id)
            os.makedirs(work_frames, exist_ok=True)

            # extract frames
            cmd_frames = f"ffmpeg -hide_banner -loglevel error -y -i \"{video_path}\" -qscale:v 2 -threads 1 -f image2 \"{os.path.join(work_frames, '%06d.jpg')}\""
            subprocess.call(cmd_frames, shell=True)

            # extract full audio
            full_audio = os.path.join(work_frames, 'audio.wav')
            cmd_audio = f"ffmpeg -hide_banner -loglevel error -y -i \"{video_path}\" -ac 1 -vn -acodec pcm_s16le -ar 48000 \"{full_audio}\""
            subprocess.call(cmd_audio, shell=True)

            frames = sorted([f for f in os.listdir(work_frames) if f.endswith('.jpg')])
            total_frames = len(frames)
            if total_frames == 0:
                shutil.rmtree(work_frames, ignore_errors=True)
                continue
            video_out_dir = os.path.join(out_root, video_id)
            os.makedirs(video_out_dir, exist_ok=True)

            if total_frames < 30:
                videonum = '00000'
                chunk_dir = os.path.join(video_out_dir, videonum)
                os.makedirs(chunk_dir, exist_ok=True)
                for j in range(30):
                    src_idx = min(j, total_frames - 1)
                    src = os.path.join(work_frames, frames[src_idx])
                    dst = os.path.join(chunk_dir, '%06d.jpg' % (j + 1))
                    shutil.copyfile(src, dst)

                audiotmp = os.path.join(video_out_dir, videonum + '.wav')
                cmd_chunk_audio = f"ffmpeg -hide_banner -loglevel error -y -i \"{full_audio}\" -ss 0.000 -to 1.000 \"{audiotmp}\""
                subprocess.call(cmd_chunk_audio, shell=True)
                shutil.rmtree(work_frames, ignore_errors=True)
                continue

            for frameNum in range(0, total_frames, 30):
                if frameNum + 30 > total_frames:
                    continue
                videonum = '%05d' % (frameNum // 30)
                chunk_dir = os.path.join(video_out_dir, videonum)
                os.makedirs(chunk_dir, exist_ok=True)
                # copy frames
                for j in range(30):
                    src = os.path.join(work_frames, '%06d.jpg' % (frameNum + j + 1))
                    dst = os.path.join(chunk_dir, '%06d.jpg' % (frameNum + j + 1))
                    if os.path.exists(src):
                        shutil.copyfile(src, dst)

                # create audio snippet for this chunk
                audiotmp = os.path.join(video_out_dir, videonum + '.wav')
                audiostart = frameNum / 30.0
                audioend = (frameNum + 30) / 30.0
                cmd_chunk_audio = f"ffmpeg -hide_banner -loglevel error -y -i \"{full_audio}\" -ss {audiostart:.3f} -to {audioend:.3f} \"{audiotmp}\""
                subprocess.call(cmd_chunk_audio, shell=True)

            shutil.rmtree(work_frames, ignore_errors=True)


def run_inference_and_collect(out_dir, dataset_name, checkpoint, device='cuda'):
    # load model
    args = argparse.Namespace()
    args.with_att = True
    args.residual_conn = False
    args.spatial_size = 28
    args.img_dim = 224
    model = My_Network(with_attention=args.with_att, residual_conn=args.residual_conn, spatial_size=args.spatial_size)
    map_loc = None if device == 'cuda' else torch.device('cpu')
    ckpt = torch.load(checkpoint, map_location=map_loc)
    state = ckpt.get('state_dict', ckpt)
    try:
        model.load_state_dict(state)
    except Exception:
        # try to remove DataParallel prefix
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state)

    if device == 'cuda':
        model = model.cuda()
    model.eval()

    # prepare transform matching my_train.py
    from torchvision import transforms
    from augmentation import Scale, ToTensor, Normalize
    transform = transforms.Compose([
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # build dataloader
    class ArgsForLoader:
        pass

    loader_args = ArgsForLoader()
    loader_args.out_dir = out_dir
    loader_args.batch_size = 1
    loader_args.num_workers = 4
    loader_args.vis_min_fake_len = 2
    loader_args.vis_max_fake_len = -1
    loader_args.aud_min_fake_len = 2
    loader_args.aud_max_fake_len = -1
    loader_args.using_pseudo_fake = False
    loader_args.dataset = dataset_name

    # We will reuse get_rawaudio_data from this module by simple wrapper
    test_loader = get_rawaudio_data(transform, loader_args, mode='test')

    test_pred = {}
    test_target = {}
    test_num_chunks = {}

    with torch.no_grad():
        for idx, (video_seq, audio_seq, target, audiopath) in enumerate(test_loader):
            if len(video_seq) == 0:
                continue
            # flip label like training/test code
            target = 1 - target

            # use first sample in batch
            video = video_seq[0].unsqueeze(0).to(device)
            audio = audio_seq[0].unsqueeze(0).to(device)

            out, vid_aud_dist, att = model(video, audio)

            pred_val = out[0].view(-1).item()
            tar = int(target[0, :].view(-1).item())
            vid_name = os.path.basename(os.path.dirname(audiopath[0]))

            if vid_name in test_pred:
                test_pred[vid_name] += pred_val
                test_num_chunks[vid_name] += 1
            else:
                test_pred[vid_name] = pred_val
                test_num_chunks[vid_name] = 1
                test_target[vid_name] = tar

    return test_pred, test_target, test_num_chunks


def append_results_markdown(md_path, row):
    header = '| Timestamp | Dataset | Videos | Chunks | Threshold | ThresholdStrategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | EER | FPR |\n'
    sep = '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'
    exists = os.path.exists(md_path)
    needs_header = True
    if exists and os.path.getsize(md_path) > 0:
        with open(md_path, 'r') as f:
            first_line = f.readline().strip()
        needs_header = first_line != header.strip()
    with open(md_path, 'a') as f:
        if needs_header:
            f.write(header)
            f.write(sep)
        f.write(f"| {row['Timestamp']} | {row['Dataset']} | {row['Videos']} | {row['Chunks']} | {row['Threshold']:.4f} | {row['ThresholdStrategy']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['ROC_AUC']:.4f} | {row['PR_AUC']:.4f} | {row['EER']:.4f} | {row['FPR']:.4f} |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='all', help='Comma separated dataset names or "all"')
    parser.add_argument('--base_data_root', type=str, default=r'C:\t309\dataSubset')
    parser.add_argument('--metadata_root', type=str, default=None)
    parser.add_argument('--raw_root', type=str, default=r'C:\t309\dataset')
    parser.add_argument('--checkpoint', type=str, default=r'C:\t309\detectors\FGI\model_best_epoch99.pth.tar')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_md', type=str, default=r'C:\t309\results\fgi\fgi.md')
    parser.add_argument('--stage_root', type=str, default=r'C:\t309\results\fgi\staging')
    parser.add_argument('--preprocess', type=str, default='simple', choices=['simple', 'full'])
    parser.add_argument(
        '--threshold_strategy',
        type=str,
        default='fixed',
        choices=['fixed', 'f1'],
        help='Use fixed threshold for unbiased test metrics. f1 is rejected because it tunes on the evaluated set.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold used when --threshold_strategy fixed.',
    )
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if args.threshold_strategy == 'f1':
        parser.error(
            '--threshold_strategy f1 tunes on evaluation labels and produces biased metrics; '
            'use --threshold_strategy fixed with a preselected --threshold.'
        )

    metadata_root = args.metadata_root if args.metadata_root else args.base_data_root

    # determine datasets by listing metadata files in metadata_root
    if args.datasets == 'all':
        mets = [f for f in os.listdir(metadata_root) if f.endswith('.metadata.json')]
        datasets = [os.path.splitext(f)[0].replace('.metadata', '') for f in mets]
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    os.makedirs(args.stage_root, exist_ok=True)

    for ds in datasets:
        print('Processing dataset', ds)
        ds_lower = ds.lower()
        model_dataset = 'fakeavceleb' if 'faceav' in ds_lower or 'fakeav' in ds_lower else 'dfdc'

        entries = load_metadata_entries(ds, metadata_root, args.raw_root)
        if not entries:
            print(f'No metadata entries found for {ds}; skipping.')
            continue

        selected = select_entries(entries, dry_run=args.dry_run, max_total=10)
        if not selected:
            print(f'No entries selected for {ds}; skipping.')
            continue

        print(f'Selected {len(selected)} items for {ds}')

        stage_dir = os.path.join(args.stage_root, ds)
        staged_count, staged_by_label, missing_by_label = stage_dataset(selected, stage_dir)
        if staged_count == 0:
            print(f'No files staged for {ds}; skipping.')
            continue
        print(
            f'Staged {staged_count} files for {ds} at {stage_dir} '
            f"(real={staged_by_label['real']}, fake={staged_by_label['fake']}, "
            f"missing_real={missing_by_label['real']}, missing_fake={missing_by_label['fake']})"
        )

        run_preprocess_dir = stage_dir

        # run preprocessing to create pytmp and audio chunks
        print('Running preprocessing for', ds)
        if args.dry_run or args.preprocess == 'simple':
            simple_preprocess_for_dir(run_preprocess_dir)
        else:
            run_preprocess_for_dir(run_preprocess_dir)

        # create csv splits
        print('Creating CSV splits for', ds)
        test_count = prepare_splits(run_preprocess_dir, model_dataset, dry_run=args.dry_run)
        print(f'Generated {test_count} test rows for {ds}')

        if test_count == 0:
            print(f'No test samples generated for dataset {ds}; skipping inference.')
            continue

        # run inference and collect pickles
        print('Running inference for', ds)
        pred, targ, numc = run_inference_and_collect(run_preprocess_dir, model_dataset, args.checkpoint, device=args.device)

        try:
            metrics = evaluate_from_pickles(
                pred,
                targ,
                numc,
                threshold_strategy=args.threshold_strategy,
                threshold=args.threshold,
            )
        except ValueError as exc:
            print(f'Could not compute metrics for dataset {ds}: {exc}')
            continue
        row = {
            'Timestamp': datetime.now().isoformat(),
            'Dataset': ds,
            **metrics
        }
        append_results_markdown(args.out_md, row)
        print('Saved results for', ds)


if __name__ == '__main__':
    main()
