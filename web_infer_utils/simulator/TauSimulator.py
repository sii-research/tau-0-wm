import argparse
import os
import sys
import time
import glob

import cv2
from yaml import Dumper, Loader, dump, load

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.insert(0, current_dir)

root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)

import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from utils.model_utils import load_diffusion_model, count_model_parameters

from utils import import_custom_class, save_video

import json


DEFAULT_HEAD = np.array([0.7060, 0.3267])
DEFAULT_WAIST = np.array([0.0, 0.43633231])
DEFAULT_VELOCITY = np.array([0.0, 0.0])


class TauSimulator:
    def __init__(
        self,
        config_file,
        device = torch.device("cuda:0"),
        rank=0,
    ):
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        
        self.device = device
        self.rank=rank
        
        self.dtype = torch.bfloat16
        self.action_dim = args.action_dim
        self.gripper_dim = args.gripper_dim
        
        self.chunk = args.chunk
        self.action_chunk = args.action_chunk
        self.img_size = args.img_size

        self.action_type = args.action_type
        self.action_space = args.action_space

        print(self.action_type, self.action_space)

        self.args = args

        self.prepare_models()

        with open(args.statistics_file, "r") as f:
            self.StatisticInfo = json.load(f)

        self.norm_type = args.norm_type

        if self.norm_type == "meanstd":
            ### (1,1,C)
            self.act_mean = torch.tensor(self.StatisticInfo["action"]["mean"]).unsqueeze(0).unsqueeze(0)
            self.act_std = torch.tensor(self.StatisticInfo["action"]["std"]).unsqueeze(0).unsqueeze(0)+1e-6
            ### (C, )
            # self.sta_mean = np.array(self.StatisticInfo["state"]["mean"])
            # self.sta_std = np.array(self.StatisticInfo["state"]["std"])+1e-6
        else:
            raise NotImplementedError
        
        self.add_mem = getattr(self.args, "add_mem", False)
        if self.add_mem:
            self.memory = dict({
                "obs": [],
            })
        self.reset()

    def prepare_models(self,):

        print("Initializing models")
        device = self.device
        dtype = self.dtype
        
        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        
        if getattr(self.args, 'vae_path', False):
            vae_path=self.args.vae_path
        else:
            vae_path=os.path.join(self.args.pretrained_model_name_or_path), getattr(self.args, "vae_checkpoint", "Wan2.2_VAE.pth")
        self.vae = vae_class(vae_pth=vae_path, device=device, dtype=dtype)

        self.SPATIAL_DOWN_RATIO = 16
        self.TEMPORAL_DOWN_RATIO = 4
        
        print(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        print(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')

        ### Load Tokenizer
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        self.text_encoder = textenc_class(
            dtype=dtype,
            device=device,
            **self.args.text_encoder
        )

    
        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=True,
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype).eval()
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model:{total_params}')


        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )
        self.pipeline = self.pipeline_class(
            self.text_encoder,
            self.vae,
            self.diffusion_model,
            device_id=self.rank,
            rank=self.rank,
        )

    @torch.no_grad()
    def play(
        self, obs, prompt, actions, num_inference_steps=5, execution_step=99,
        guide_scale=1,
        sim_path="./sim_results", save_sim_video=False,
        sample_solver="unipc",
        n_mem=3, shift=5.0,
        *args, **kwargs
    ):

        """
        Input:
            obs: One of the followings
                1. torch.tensor of shape: {v, 3, h, w}, ranging from -1 to 1
                2. np.array of shape: {v, h, w, 3}, ranging from 0 to 255 (np.uin8)
            prompt: task description
            actions: {t, c}, execution actions to be simulated
        Output:
            video and rewards based on given actions
        """

        if prompt.find("<reset>")>=0:
            self.reset()
            prompt = prompt.replace("<reset>", "")

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        if actions.shape[0] > execution_step:
            actions = actions[:execution_step]

        assert(execution_step<=self.action_chunk)

        assert execution_step >= 1 and execution_step <= 100, "execution_step should be in [1, 100]"

        mem = None

        if obs is not None:
            if isinstance(obs, np.ndarray) and obs.dtype == np.uint8:
                obs = obs.astype(np.float32) / 255 * 2 - 1
                obs = np.transpose(obs, (0,3,1,2))
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)
                
            v, c, h, w = obs.shape
            obs = obs.unsqueeze(2).transpose(0,1)  # v,c,h,w -> c v t h w

            if self.add_mem:
                self.memory["obs"].append(obs)
                n_history = len(self.memory["obs"])
                mem_idx = np.linspace(0, n_history-1, n_mem+1).astype(np.int16).tolist()[:-1]
                # print(mem_idx)
                mem = torch.cat([self.memory["obs"][_mem_idx] for _mem_idx in mem_idx], dim=2).to(self.device, dtype=self.dtype)
            obs = obs.to(self.device, dtype=self.dtype)

        actions = actions.unsqueeze(0)
        actions = (actions - self.act_mean) / self.act_std


        if getattr(self.args, "whole_body", False):
            ### A portion of the training data includes full-body movements. To resolve the ambiguity of the action-conditioned simulator, we incorporate simple control signals for the head and waist joints and robot velocity specifically for this subset of the data. For the data not containing full‑body movements, we can simply use the DEFAULT values.
            head_joint_states = torch.tensor(DEFAULT_HEAD)
            if len(head_joint_states.shape)==1:  
                head_joint_states = head_joint_states.view(1,1,2).repeat(actions.shape[0], actions.shape[1], 1).to(dtype=actions.dtype, device=actions.device)
            waist_joint_states = torch.tensor(DEFAULT_WAIST)
            if len(waist_joint_states.shape)==1:  
                waist_joint_states = waist_joint_states.view(1,1,2).repeat(actions.shape[0], actions.shape[1], 1).to(dtype=actions.dtype, device=actions.device)
            robot_velocity = torch.tensor(DEFAULT_VELOCITY)
            if len(robot_velocity.shape)==1:  
                robot_velocity = robot_velocity.view(1,1,2).repeat(actions.shape[0], actions.shape[1], 1).to(dtype=actions.dtype, device=actions.device)
            actions = torch.cat((
                actions,
                head_joint_states,
                waist_joint_states,
                robot_velocity,
            ), dim=-1)


        sim_preds = self.pipeline.infer(
            prompt,
            obs,
            self.chunk,
            guide_scale=guide_scale,
            sampling_steps=num_inference_steps,
            shift=shift,
            return_video=True,
            return_reward=True,
            actions=actions.to(self.device, dtype=self.dtype),
            decode = True,
            mem_img = mem,
            sample_solver=sample_solver,
        )

        ### v, c, t, h, w
        video = torch.stack(sim_preds[0], dim=0).cpu()        
        pred_final_frame = video[:,:,-1]
        pred_reward = sim_preds[-1]

        if save_sim_video:
            sim_root = os.path.dirname(sim_path)
            os.makedirs(sim_root, exist_ok=True)
            video2save = torch.cat(sim_preds[0], dim=-1).cpu()
            if self.video_cache is None:
                self.video_cache = video2save
            else:
                self.video_cache = torch.cat((self.video_cache, video2save), dim=1)
            save_video(self.video_cache, sim_path, fps=int(30*(self.chunk-1)/(self.action_chunk-1)))

        return pred_final_frame.data.cpu().numpy(), pred_reward.squeeze().data.cpu().float().numpy()


    def reset(self, *args, **kwargs):
        if self.add_mem:
            self.memory = dict({
                "obs": [],
            })

        self.video_cache = None

