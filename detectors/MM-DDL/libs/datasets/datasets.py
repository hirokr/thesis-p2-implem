import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed, is_pin_memory_ok

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers, shuffle=True):
    """
        A simple dataloder builder
    """
    pin_memory = torch.cuda.is_available() and is_pin_memory_ok()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": trivial_batch_collator,
        "worker_init_fn": (worker_init_reset_seed if is_training else None),
        "shuffle": shuffle,
        "drop_last": is_training,
        "generator": generator,
        "pin_memory": pin_memory,
    }
    if num_workers and num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    return loader
