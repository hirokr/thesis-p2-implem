import os
import copy
import random
import ctypes
import numpy as np
import random
import time
import torch

_PIN_MEMORY_OK = True


def set_pin_memory_ok(value):
    global _PIN_MEMORY_OK
    _PIN_MEMORY_OK = bool(value)


def is_pin_memory_ok():
    return _PIN_MEMORY_OK


def get_cuda_mem_mb(device):
    if device is None or device.type != "cuda":
        return None
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return allocated, reserved


def _get_cpu_rss_mb_windows():
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return counters.WorkingSetSize / (1024 ** 2)


def get_cpu_rss_mb():
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        if os.name == "nt":
            try:
                return _get_cpu_rss_mb_windows()
            except Exception:
                return None
    return None


def get_loader_queue_size(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return None
    queues = getattr(iterator, "_index_queues", None)
    if not queues:
        return None
    total = 0
    for queue in queues:
        try:
            total += queue.qsize()
        except (NotImplementedError, OSError, AttributeError):
            return None
    return total


def build_precompute_loader(
    dataset,
    batch_size,
    num_workers,
    pin_memory,
    collate_fn=None,
    prefetch_factor=2,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if num_workers and num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    offset,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict
