from torch.utils.data.dataset import Dataset
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

try:
    lerobot_version = version("lerobot")
    print("lerobot version:", lerobot_version)

    if Version(lerobot_version) < Version("0.4.0"):
        USE_LEROBOT_V30 = False
        print("lerobot version < 0.4.0")
    else:
        USE_LEROBOT_V30 = True
        print("lerobot version >= 0.4.0")

except PackageNotFoundError:
    print("lerobot is not installed in the current environment")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset, LeRobotDatasetMetadata
import traceback
import torch
import torch.nn.functional as F
import os
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections.abc import Callable
import logging
import time
import concurrent.futures
import json
import random

from einops import rearrange

import numpy as np
from typing import Sequence, Any, Dict, List, Tuple

from utils.action_space_utils import quaternion_to_euler_xyz, quaternion_to_rotation_6d, abs_eef_to_rela, abs_eef_to_delta
import pandas as pd

class _MyLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, return_video=True, **kwargs):
        self.return_video = return_video
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self.return_video:
            if len(self.meta.video_keys) > 0:
                current_ts = item["timestamp"].item()
                query_timestamps = self._get_query_timestamps(current_ts, query_indices)
                video_frames = self._query_videos(query_timestamps, ep_idx)
                item = {**video_frames, **item}

            if self.image_transforms is not None:
                image_keys = self.meta.camera_keys
                for cam in image_keys:
                    item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item


class _MyLeRobotDatasetv30(LeRobotDataset):
    def __init__(self, *args, return_video=True, **kwargs):
        self.return_video = return_video
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx) -> dict:
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self.return_video:
            if len(self.meta.video_keys) > 0:
                current_ts = item["timestamp"].item()
                query_timestamps = self._get_query_timestamps(current_ts, query_indices)
                video_frames = self._query_videos(query_timestamps, ep_idx)
                item = {**video_frames, **item}

            if self.image_transforms is not None:
                image_keys = self.meta.camera_keys
                for cam in image_keys:
                    item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name
        return item
    
    
def create_dataset(
    repo_id: str,
    delta_timestamps: dict[list[float]] | None,
    video_backend: str | None,
    return_video: bool,
):
    # print(f"Starting creation of dataset for {repo_id}")
    if USE_LEROBOT_V30:
        return _MyLeRobotDatasetv30(
            repo_id,
            delta_timestamps=delta_timestamps,
            tolerance_s=1e-3,
            video_backend=video_backend,
            return_video=return_video,
        ) 
    return _MyLeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        return_video=return_video,
    )

class CustomMultiLerobotDataset(MultiLeRobotDataset):
    def __init__(
        self,
        repo_ids: List[str],
        episodes: Optional[Dict[str, List[int]]] = None,
        image_transforms: Optional[Callable] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        tolerances_s: Optional[Dict[str, float]] = None,
        download_videos: bool = True,
        video_backend: Optional[str] = None,
        max_workers: int = 64,
        return_video: bool = True,
    ) -> None:
        self.repo_ids = repo_ids
        self.tolerances_s = tolerances_s or {rid: 0.0001 for rid in repo_ids}

        self._datasets = [create_dataset(repo_id, delta_timestamps, video_backend, return_video) for repo_id in self.repo_ids]

        # Intersection of feature sets across all datasets.
        self.disabled_features: set[str] = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets[1:]:
            intersection_features.intersection_update(ds.features)

        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )

        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = None


class CustomLeRobotDataset(Dataset):
    def __init__(self,
        data_roots,  # One or more LeRobot dataset root list, or a text file listing them.
        sample_size=(192, 256),  # Resize target for input frames, in (H, W).
        n_view=3,  # Number of camera views returned by the dataset.
        valid_cam=['observation.images.top_head', 'observation.images.hand_left', 'observation.images.hand_right'],  # Camera keys to read.
        fps=30,  # Sampling FPS used to build the temporal window.
        chunk=9,  # Number of video frames used as model input.
        action_chunk=33,  # Number of action steps aligned to the video chunk.
        action_type="absolute",  # Action supervision mode: absolute / relative / delta.
        action_space="joint",  # Action coordinate space, e.g. joint or eef6d.
        ignore_seek=False,  # If True, repeat current frame instead of seeking by timestamp.
        action_key="action",  # Raw action field name in the dataset item.
        state_key="observation.state",  # Raw state field name in the dataset item.
        return_video=True,  # Whether to return video tensors.
        return_action=True,  # Whether to return action/state pairs.
        norm_action=True,  # Whether to normalize actions with statistics.
        filter_action=True,  # Whether to reject outlier action/state samples.
        statistic_files=None,  # Statistics JSON file(s), one per dataset root.
        video_backend=None,  # LeRobot video backend.
        dual_arm=None,  # True for dual-arm layouts, False for single-arm layouts.
        state_index=None,  # Slice indices for the arm pose part of raw state.
        state_gripper_index=None,  # Slice indices for the gripper part of raw state.
        action_index=None,  # Slice indices for the arm pose part of raw action.
        action_gripper_index=None,  # Slice indices for the gripper part of raw action.
    ):
        self.n_view = n_view
        self.valid_cam = valid_cam
        assert len(self.valid_cam) == self.n_view
        
        self.fps = fps
        
        self.action_key = action_key
        self.state_key = state_key
        self.return_video = return_video
        if not self.return_video:
            print("You are not loading video, only recommended when computing statistics.")
                
        if type(data_roots)==str and os.path.exists(data_roots):
            with open(data_roots, 'r', encoding="utf-8") as f:
                paths = [line.strip() for line in f if line.strip()]
        else:
            paths = data_roots
        self.data_roots = paths
        
        self.return_action = return_action
        if self.return_action:
            self.norm_action = norm_action
            self.filter_action = filter_action
            
            self.action_space = action_space
            self.action_type = action_type
        else:
            self.norm_action = False
            self.filter_action = False

        if self.norm_action or self.filter_action:
            assert statistic_files is not None
            
        self.stat_info = []

        if self.norm_action or self.filter_action:
            if isinstance(statistic_files, str) and os.path.exists(statistic_files):
                with open(statistic_files, "r", encoding="utf-8") as f:
                    statistic_file_list = [line.strip() for line in f if line.strip()]
            else:
                statistic_file_list = statistic_files

            assert len(statistic_file_list) == len(self.data_roots), (
                f"statistic_files length mismatch: "
                f"{len(statistic_file_list)} vs {len(self.data_roots)}"
            )

            for stat_path in statistic_file_list:
                with open(stat_path, "r") as f:
                    self.stat_info.append(json.load(f))
        
        # because of the existence of current observation
        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        self.video_temporal_stride = (self.action_chunk - 1) // (self.chunk - 1)
        assert((self.chunk - 1) * self.video_temporal_stride == self.action_chunk - 1)
        
        self.ignore_seek = ignore_seek        
        self.sample_size = tuple(sample_size)
        
        delta_timestamps = self.get_delta_timestamps()
        self.datasets = CustomMultiLerobotDataset(
            repo_ids=self.data_roots,
            delta_timestamps=delta_timestamps,
            return_video=self.return_video,
            video_backend=video_backend,
        )
        
        self.skip_idx = set()
                    
        for i in range(len(self.stat_info)):
            self.stat_info[i]["action"]["mean"] = torch.tensor(self.stat_info[i]["action"]["mean"])
            self.stat_info[i]["action"]["std"] = torch.tensor(self.stat_info[i]["action"]["std"])
            self.stat_info[i]["state"]["mean"] = torch.tensor(self.stat_info[i]["state"]["mean"])
            self.stat_info[i]["state"]["std"] = torch.tensor(self.stat_info[i]["state"]["std"])

        self.dual_arm = dual_arm
        if self.dual_arm:
            assert len(state_gripper_index) == len(action_gripper_index) == 2
        self.state_index = list(state_index)
        self.state_gripper_index = list(state_gripper_index)
        self.action_index = list(action_index)
        self.action_gripper_index = list(action_gripper_index)
        
    def __len__(self):
        return len(self.datasets)
    
    
    def get_delta_timestamps(self):
        # Assume 30 FPS when building the sampling offsets.
        delta_timestamps={
                self.action_key: [i/self.fps for i in range(self.action_chunk)]
        }
        if not self.ignore_seek:
            video_timestamps = {}
            for i in range(self.n_view):
                video_timestamps.update({
                    self.valid_cam[i]: [i/self.fps for i in range(0, self.action_chunk, self.video_temporal_stride)][:self.chunk]
                })
            delta_timestamps.update(video_timestamps)
        return delta_timestamps
    
    def get_state_from_raw_state(self, state):
        state_ = state[:, self.state_index]
        state_gripper = state[:, self.state_gripper_index]
        if not self.dual_arm:
            state_ = torch.cat((state_, state_), dim=-1)
            state_gripper = torch.cat((state_gripper, state_gripper), dim=-1)
        return state_, state_gripper
    
    def get_action_from_raw_action(self, action):
        action_ = action[:, self.action_index]
        action_gripper = action[:, self.action_gripper_index]
        if not self.dual_arm:
            action_ = torch.cat((action_, action_), dim=-1)
            action_gripper = torch.cat((action_gripper, action_gripper), dim=-1)
        return action_, action_gripper
    
    def transform_and_norm_action_state(self, raw_data, stat):
        # action, shape t, c
        action = raw_data[self.action_key]
        
        # state, shape c -> 1, c
        state = raw_data[self.state_key].unsqueeze(0)
        
        state_, state_gripper = self.get_state_from_raw_state(state)
        action_, action_gripper = self.get_action_from_raw_action(action)
        
        if self.action_type == "absolute":
            pass
                
        elif self.action_type == "relative":
            action_ = self.get_relative_action(state_, action_)
            
        elif self.action_type == "delta":
            action_ = self.get_delta_action(state_, action_)
        
        else:
            raise ValueError("Invalid action type")
        
        action_dim = action_.shape[-1]
        state_dim = state_.shape[-1]
        if self.dual_arm:
            action = torch.cat((action_[:,:action_dim//2], action_gripper[:,:1], action_[:,action_dim//2:], action_gripper[:,1:]), dim=-1)
            state = torch.cat((state_[:,:state_dim//2], state_gripper[:,:1], state_[:,state_dim//2:], state_gripper[:,1:]), dim=-1)
        else:
            action = torch.cat((action_[:,:action_dim//2], action_gripper[:,:1]), dim=-1)
            state = torch.cat((state_[:,:state_dim//2], state_gripper[:,:1]), dim=-1)
        
        if self.norm_action:
            action_mean = stat["action"]["mean"]
            action_std = stat["action"]["std"]
            state_mean = stat["state"]["mean"]
            state_std = stat["state"]["std"]

            action = (action-action_mean) / action_std
            state = (state-state_mean) / state_std

            if self.filter_action:
                if (torch.abs(action) > 100).sum() > 0 or (torch.abs(state) > 100).sum() > 0:
                    raise ValueError('Error action value')
                
        return action, state
        
        
    def get_relative_action(self, state_, action_):
        if self.action_space == "joint":
            return action_ - state_
        elif "eef" in self.action_space:
            return abs_eef_to_rela(action_, state_)
        else:
            raise ValueError("Invalid action space")
    
    
    def get_delta_action(self, state_, action_):
        if self.action_space == "joint":
            delta = torch.zeros_like(action_)
            delta[0] = action_[0] - state_[0]
            delta[1:] = action_[1:] - action_[:-1]
            return delta
        
        elif "eef" in self.action_space:
            return abs_eef_to_delta(action_, state_)
        
        else:
            raise ValueError("Invalid action space")
    
    
    def get_video(self, raw_data):        
        videos = []
        for i in range(self.n_view):
            videos.append(self.transform_video(raw_data[self.valid_cam[i]]))
        
        videos = torch.stack(videos, dim=0)    # v, t, c, h, w ->  c, v, t, h, w
        videos = rearrange(videos, "v t c h w -> c v t h w")
        
        if self.ignore_seek:
            videos = videos.repeat(1,1,self.chunk,1,1)
        
        return videos
    
    
    def transform_video(self, video_tensor):
        """
        x: Tensor [C, H, W] or [T, C, H, W] in [0, 1]
        sample_size: int or (H, W)
        """

        # pad shape if necessary
        if self.ignore_seek:
            video_tensor = video_tensor.unsqueeze(0)  # [T, C, H, W]
            
        # resize
        video_tensor = F.interpolate(
            video_tensor,
            size=self.sample_size,
            mode="bilinear",
            align_corners=False,
        )

        # normalize
        video_tensor = video_tensor * 2.0 - 1.0

        return video_tensor
        
        
    def get_batch(self, index):
        results = {}
                
        raw_data = self.datasets[index]
        dataset_index = int(raw_data['dataset_index'])
        
        results['caption'] = raw_data['task']
                                    
        # action
        if self.return_action:
            if self.norm_action or self.filter_action:
                stat = self.stat_info[dataset_index]
            else:
                stat = None
            action, state = self.transform_and_norm_action_state(raw_data, stat)
        
            results['actions'] = action
            results['state'] = state
        
        # video
        if self.return_video:
            results['video'] = self.get_video(raw_data)
        
        return results
        
        
    def __getitem__(self, idx):            
        while True:
            try:
                while idx in self.skip_idx:
                    idx = random.randint(0, self.__len__()-1)
                results = self.get_batch(idx)
                break
            except:
                self.skip_idx.add(idx)
                traceback.print_exc()
                print('drop:', idx)
                idx = random.randint(0, self.__len__()-1)
                    
        return results


if __name__ == "__main__":
    raise SystemExit("This module is intended to be imported, not executed directly.")
