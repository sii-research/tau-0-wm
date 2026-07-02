from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from yaml import safe_load

from utils import import_custom_class


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute action/state statistics from a dataset.")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--action_space", type=str, required=True)
    parser.add_argument("--action_type", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--num", "-n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--torch_threads_per_worker", type=int, default=1)
    parser.add_argument("--skip_error", action="store_true")
    return parser.parse_args()


def prepare_dataset(config: Dict[str, Any], action_space: str, action_type: str):
    dataset_class = import_custom_class(config["data_class"], config["data_class_path"])
    dataset_kwargs = dict(config["data"])
    dataset_kwargs.update(
        norm_action=False,
        filter_action=False,
        return_video=False,
        action_space=action_space,
        action_type=action_type,
        split_dataset=False,
        return_caption=False,
    )

    sig = inspect.signature(dataset_class.__init__)
    valid_params = set(sig.parameters) - {"self"}
    dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if k in valid_params}
    return dataset_class(**dataset_kwargs)


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


class IndexedActionStateDataset(Dataset):
    def __init__(self, dataset, indices: List[int], skip_error: bool = False):
        self.dataset = dataset
        self.indices = indices
        self.skip_error = skip_error

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        real_idx = self.indices[i]
        try:
            batch = self.dataset[real_idx]
            action = to_tensor(batch["actions"])
            state = to_tensor(batch["state"])

            if action.ndim != 2:
                raise ValueError(f"action should be (T, C), got {tuple(action.shape)}, idx={real_idx}")
            if state.ndim == 1:
                state = state.unsqueeze(0)
            if state.ndim != 2:
                raise ValueError(f"state should be (1, C), got {tuple(state.shape)}, idx={real_idx}")

            return {
                "action": action.to(torch.float32),
                "state": state.to(torch.float32),
                "idx": real_idx,
                "ok": True,
                "error": "",
            }
        except Exception as exc:
            if not self.skip_error:
                raise
            return {"action": None, "state": None, "idx": real_idx, "ok": False, "error": repr(exc)}


def collate_action_state(samples: List[Dict[str, Any]]):
    valid_actions = []
    valid_states = []
    skipped = 0
    failed_indices = []

    for sample in samples:
        if not sample["ok"]:
            skipped += 1
            failed_indices.append((sample["idx"], sample["error"]))
            continue
        valid_actions.append(sample["action"])
        valid_states.append(sample["state"])

    if not valid_actions:
        return {"action": None, "state": None, "skipped": skipped, "failed_indices": failed_indices}

    return {
        "action": torch.cat(valid_actions, dim=0),
        "state": torch.cat(valid_states, dim=0),
        "skipped": skipped,
        "failed_indices": failed_indices,
    }


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sq_sum = None

    def update(self, x: torch.Tensor):
        if x is None:
            return
        if x.ndim != 2:
            raise ValueError(f"RunningStats expects 2D tensor, got {tuple(x.shape)}")
        x = x.to(torch.float64)
        batch_count = x.shape[0]
        batch_sum = x.sum(dim=0)
        batch_sq_sum = (x * x).sum(dim=0)
        self.sum = batch_sum if self.sum is None else self.sum + batch_sum
        self.sq_sum = batch_sq_sum if self.sq_sum is None else self.sq_sum + batch_sq_sum
        self.count += batch_count

    def mean_std(self):
        if self.count <= 0:
            raise RuntimeError("No valid samples collected.")
        mean = self.sum / self.count
        var = torch.clamp(self.sq_sum / self.count - mean * mean, min=0.0)
        return mean, torch.sqrt(var)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def main() -> None:
    args = get_args()
    torch.set_num_threads(args.torch_threads_per_worker)

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = safe_load(f)

    dataset = prepare_dataset(config, args.action_space, args.action_type)
    dataset_len = len(dataset)
    num_samples = min(args.num, dataset_len)

    print(f"Random sampling {num_samples} / {dataset_len} samples")
    print(f"DataLoader: batch_size={args.batch_size}, num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor}")

    random.seed(args.seed)
    sampled_indices = random.sample(range(dataset_len), num_samples)

    stat_dataset = IndexedActionStateDataset(dataset, sampled_indices, skip_error=args.skip_error)
    dataloader_kwargs = dict(
        dataset=stat_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_action_state,
        worker_init_fn=seed_worker,
        pin_memory=False,
        drop_last=False,
    )
    if args.num_workers > 0:
        dataloader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)

    dataloader = DataLoader(**dataloader_kwargs)
    action_stats = RunningStats()
    state_stats = RunningStats()
    skipped = 0

    total_batches = math.ceil(num_samples / args.batch_size)
    for batch in tqdm(dataloader, total=total_batches):
        skipped += batch["skipped"]
        if batch["action"] is None or batch["state"] is None:
            continue
        action_stats.update(batch["action"])
        state_stats.update(batch["state"])

    action_mean, action_std = action_stats.mean_std()
    state_mean, state_std = state_stats.mean_std()

    result = {
        "action": {"mean": action_mean.tolist(), "std": action_std.tolist()},
        "state": {"mean": state_mean.tolist(), "std": state_std.tolist()},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Saved statistics to: {args.output}")
    print(f"Skipped samples: {skipped}")
    print(f"Action count after streaming: {action_stats.count}")
    print(f"State count after streaming: {state_stats.count}")
    print("Remember to adjust the gripper dimensions if needed.")


if __name__ == "__main__":
    main()
