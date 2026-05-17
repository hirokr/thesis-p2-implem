import argparse
import os
import subprocess
import sys
import shutil
import tempfile
from datetime import datetime
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score, accuracy_score

import torch

# Ensure script runs with detectors/FGI on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset_3d import deepfake_3d_rawaudio, my_collate_rawaudio


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
                                      collate_fn=my_collate_rawaudio)
    else:
        raise ValueError('only test mode supported in run script')

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader
from model import My_Network


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # EER where FPR ~= FNR
    abs_diffs = np.abs(fpr - fnr)
    idx = np.nanargmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return eer, thr


def best_threshold_by_f1(y_true, y_score):
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_f1 = -1
    best_thr = 0.5
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def evaluate_from_pickles(pred_dict, target_dict, num_chunks_dict):
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

    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float('nan')
    pr_auc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float('nan')

    thr, _ = best_threshold_by_f1(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    eer, eer_thr = compute_eer(y_true, y_score)

    # FPR at chosen threshold
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'Samples': len(y_true),
        'Threshold': float(thr),
        'ThresholdStrategy': 'f1',
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

    # train and test roots
    train_root = os.path.join(out_dir, 'train', 'pytmp')
    test_root = os.path.join(out_dir, 'test', 'pytmp')

    train_rows = build_split(train_root)
    test_rows = build_split(test_root)

    # write train_split.csv
    train_csv = os.path.join(out_dir, 'train_split.csv')
    with open(train_csv, 'w') as f:
        for r in train_rows:
            f.write(','.join(r) + '\n')

    # write test split depending on dataset naming the loader expects
    if dataset_name == 'dfdc':
        test_csv = os.path.join(out_dir, 'test_imbalance_split.csv')
    else:
        test_csv = os.path.join(out_dir, 'test_balance_fakeavceleb_split.csv')

    with open(test_csv, 'w') as f:
        for r in test_rows:
            f.write(','.join(r) + '\n')

    # If dry_run, truncate test CSV to first 10 lines
    if dry_run:
        if os.path.exists(test_csv):
            with open(test_csv, 'r') as f:
                lines = f.readlines()
            with open(test_csv, 'w') as f:
                f.writelines(lines[:10])

    return len(test_rows)


def run_preprocess_for_dir(raw_dir, dont_crop_face=False):
    # call pre-process.py with out_dir = raw_dir
    cmd = [sys.executable, os.path.join(HERE, 'pre-process.py'), '--out_dir', raw_dir]
    if dont_crop_face:
        cmd.append('--dont_crop_face')
    subprocess.run(cmd, cwd=HERE, check=True)


def simple_preprocess_for_dir(raw_dir):
    """Lightweight preprocessing for dry-run: split videos into 30-frame chunks and audio snippets using ffmpeg.
    Produces structure: <raw_dir>/pytmp/{real,fade}/{video_id}/{chunk_id}/[frames].jpg and <raw_dir>/pytmp/{real,fake}/{video_id}/{chunk_id}.wav
    """
    for split in ['real', 'fake']:
        src_dir = os.path.join(raw_dir, split)
        if not os.path.exists(src_dir):
            continue
        out_root_train = os.path.join(raw_dir, 'train', 'pytmp', split)
        out_root_test = os.path.join(raw_dir, 'test', 'pytmp', split)
        os.makedirs(out_root_train, exist_ok=True)
        os.makedirs(out_root_test, exist_ok=True)

        for video in os.listdir(src_dir):
            if not video.lower().endswith('.mp4'):
                continue
            video_path = os.path.join(src_dir, video)
            video_id = os.path.basename(video)[:-4]
            work_frames = os.path.join(raw_dir, 'tmp_frames', video_id)
            os.makedirs(work_frames, exist_ok=True)

            # extract frames
            cmd_frames = f"ffmpeg -y -i \"{video_path}\" -qscale:v 2 -threads 1 -f image2 \"{os.path.join(work_frames, '%06d.jpg')}\""
            subprocess.call(cmd_frames, shell=True)

            # extract full audio
            full_audio = os.path.join(work_frames, 'audio.wav')
            cmd_audio = f"ffmpeg -y -i \"{video_path}\" -ac 1 -vn -acodec pcm_s16le -ar 48000 \"{full_audio}\""
            subprocess.call(cmd_audio, shell=True)

            frames = sorted([f for f in os.listdir(work_frames) if f.endswith('.jpg')])
            total_frames = len(frames)
            if total_frames < 30:
                shutil.rmtree(work_frames, ignore_errors=True)
                continue

            video_out_dir_train = os.path.join(out_root_train, video_id)
            video_out_dir_test = os.path.join(out_root_test, video_id)
            os.makedirs(video_out_dir_train, exist_ok=True)
            os.makedirs(video_out_dir_test, exist_ok=True)

            for frameNum in range(0, total_frames, 30):
                if frameNum + 30 > total_frames:
                    continue
                videonum = '%05d' % (frameNum // 30)
                chunk_dir_train = os.path.join(video_out_dir_train, videonum)
                chunk_dir_test = os.path.join(video_out_dir_test, videonum)
                os.makedirs(chunk_dir_train, exist_ok=True)
                os.makedirs(chunk_dir_test, exist_ok=True)
                # copy frames
                for i in range(frameNum + 1, frameNum + 31):
                    src = os.path.join(work_frames, '%06d.jpg' % i)
                    dst_train = os.path.join(chunk_dir_train, '%06d.jpg' % i)
                    dst_test = os.path.join(chunk_dir_test, '%06d.jpg' % i)
                    if os.path.exists(src):
                        shutil.copyfile(src, dst_train)
                        shutil.copyfile(src, dst_test)

                # create audio snippet for this chunk
                audiotmp_train = os.path.join(video_out_dir_train, videonum + '.wav')
                audiotmp_test = os.path.join(video_out_dir_test, videonum + '.wav')
                audiostart = frameNum / 30.0
                audioend = (frameNum + 30) / 30.0
                cmd_chunk_audio_train = f"ffmpeg -y -i \"{full_audio}\" -ss {audiostart:.3f} -to {audioend:.3f} \"{audiotmp_train}\""
                cmd_chunk_audio_test = f"ffmpeg -y -i \"{full_audio}\" -ss {audiostart:.3f} -to {audioend:.3f} \"{audiotmp_test}\""
                subprocess.call(cmd_chunk_audio_train, shell=True)
                subprocess.call(cmd_chunk_audio_test, shell=True)

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
            vid_name = audiopath[0].split('/')[-2]

            if vid_name in test_pred:
                test_pred[vid_name] += pred_val
                test_num_chunks[vid_name] += 1
            else:
                test_pred[vid_name] = pred_val
                test_num_chunks[vid_name] = 1
                test_target[vid_name] = tar

    return test_pred, test_target, test_num_chunks


def append_results_markdown(md_path, row):
    header = '| Timestamp | Dataset | Samples | Threshold | ThresholdStrategy | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC | EER | FPR |\n'
    sep = '|---' + '|' * 12 + '\n'
    exists = os.path.exists(md_path)
    with open(md_path, 'a') as f:
        if not exists:
            f.write(header)
            f.write(sep)
        f.write(f"| {row['Timestamp']} | {row['Dataset']} | {row['Samples']} | {row['Threshold']:.4f} | {row['ThresholdStrategy']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['ROC_AUC']:.4f} | {row['PR_AUC']:.4f} | {row['EER']:.4f} | {row['FPR']:.4f} |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='all', help='Comma separated dataset names or "all"')
    parser.add_argument('--base_data_root', type=str, default=r'C:\t309\dataSubset')
    parser.add_argument('--raw_root', type=str, default=r'C:\t309\dataset')
    parser.add_argument('--checkpoint', type=str, default=r'C:\t309\detectors\FGI\model_best_epoch99.pth.tar')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_md', type=str, default=r'C:\t309\results\fgi\fgi.md')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    # determine datasets by listing metadata files in base_data_root
    if args.datasets == 'all':
        mets = [f for f in os.listdir(args.base_data_root) if f.endswith('.metadata.json')]
        datasets = [os.path.splitext(f)[0].replace('.metadata','') for f in mets]
        # fallback simpler names
        datasets = [name.replace('.metadata','') for name in [os.path.splitext(x)[0] for x in mets]]
        # map known metadata filenames
        simple = []
        for m in mets:
            if 'dfdc' in m.lower():
                simple.append('dfdc')
            elif 'fakeavceleb' in m.lower() or 'faceavceleb' in m.lower():
                simple.append('fakeavceleb')
            elif 'av1' in m.lower() or 'av-1m' in m.lower():
                simple.append('av1')
            else:
                simple.append(os.path.splitext(m)[0])
        datasets = list(dict.fromkeys(simple))
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    # mapping to raw folders
    raw_map = {
        'dfdc': os.path.join(args.raw_root, 'dfdc'),
        'fakeavceleb': os.path.join(args.raw_root, 'FakeAVCeleb'),
        'av1': os.path.join(args.raw_root, 'av-1m')
    }

    for ds in datasets:
        print('Processing dataset', ds)
        ds_lower = ds.lower()
        if 'dfdc' in ds_lower:
            ds_key = 'dfdc'
        elif 'fakeav' in ds_lower or 'faceav' in ds_lower:
            ds_key = 'fakeavceleb'
        else:
            print(f'Skipping unsupported dataset: {ds}')
            continue

        # determine raw dir and out_dir (where pre-process/write_csv will write)
        raw_dir = raw_map.get(ds_key, os.path.join(args.raw_root, ds_key))
        out_dir = raw_dir

        if not os.path.exists(raw_dir):
            print(f'Raw dir for {ds} not found: {raw_dir}. Skipping.')
            continue

        # If dry_run, create small temporary folder with subset of videos
        tmp_dir = None
        try:
            if args.dry_run:
                tmp_dir = tempfile.mkdtemp(prefix=f'fgidry_{ds}_')
                for s in ['real', 'fake']:
                    src_dir = os.path.join(raw_dir, s)
                    dst_dir = os.path.join(tmp_dir, s)
                    os.makedirs(dst_dir, exist_ok=True)
                    if os.path.exists(src_dir):
                        files = [f for f in os.listdir(src_dir) if f.lower().endswith('.mp4')]
                        take = max(1, min(5, len(files)))
                        for i, fname in enumerate(files[:take]):
                            shutil.copyfile(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
                run_preprocess_dir = tmp_dir
            else:
                run_preprocess_dir = raw_dir

            # run preprocessing to create pytmp and audio chunks
            print('Running preprocessing for', ds)
            if args.dry_run:
                simple_preprocess_for_dir(run_preprocess_dir)
            else:
                run_preprocess_for_dir(run_preprocess_dir)

            # create csv splits
            print('Creating CSV splits for', ds)
            test_count = prepare_splits(run_preprocess_dir, 'dfdc' if ds == 'dfdc' else 'fakeavceleb' if ds in ('fakeavceleb','faceavceleb') else ds, dry_run=args.dry_run)

            if test_count == 0:
                print(f'No test samples generated for dataset {ds}; skipping inference.')
                continue

            # run inference and collect pickles
            print('Running inference for', ds)
            pred, targ, numc = run_inference_and_collect(run_preprocess_dir, 'dfdc' if ds == 'dfdc' else 'fakeavceleb' if ds in ('fakeavceleb','faceavceleb') else ds, args.checkpoint, device=args.device)

            metrics = evaluate_from_pickles(pred, targ, numc)
            row = {
                'Timestamp': datetime.now().isoformat(),
                'Dataset': ds,
                **metrics
            }
            append_results_markdown(args.out_md, row)
            print('Saved results for', ds)

        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
