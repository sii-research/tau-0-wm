import argparse
import logging
import os
import pdb
import sys
import time
from datetime import datetime
import cv2
from yaml import Dumper, Loader, dump, load

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)

root_dir = os.path.dirname(os.path.dirname(root_dir))
sys.path.insert(0, root_dir)

sys.path.insert(0, current_dir)

import os
import pdb
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from einops import rearrange

from models.wan_2_2_models.transformers.attention import set_attention_backend

from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model

from utils import init_logging, import_custom_class, save_video

from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video

from utils.action_space_utils import rela_eef_to_abs, quaternion_to_rotation_6d


from typing import Any, Dict, List

import json

import torch

    

logger = logging.getLogger(__name__)

class TauPolicy:
    def __init__(
        self,
        config_file,
        device = torch.device("cuda:0"),
        rank=0,
        compile_model=False,
        compile_mode="reduce-overhead",
        compile_dynamic="auto",
        compile_target="full",
        enable_action_cross_attn_kv_cache=False,
        enable_self_attn_fused_qkv=True,
        enable_context_null_cache=True,
        attention_impl="auto",
        sdpa_backend="auto",
        flash_attn_version="auto",
        enable_action_rope_cache=True,
    ):
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        
        self.device = device
        self.rank=rank
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.compile_dynamic = compile_dynamic
        self.compile_target = compile_target
        self.enable_action_cross_attn_kv_cache = enable_action_cross_attn_kv_cache
        self.enable_self_attn_fused_qkv = enable_self_attn_fused_qkv
        self.enable_context_null_cache = enable_context_null_cache
        self.attention_impl = attention_impl
        self.sdpa_backend = sdpa_backend
        self.flash_attn_version = flash_attn_version
        self.enable_action_rope_cache = enable_action_rope_cache
        
        self.dtype = torch.bfloat16
        self.action_dim = args.action_dim
        self.gripper_dim = args.gripper_dim
        
        self.chunk = args.chunk
        self.action_chunk = args.action_chunk
        self.img_size = args.img_size

        self.action_type = args.action_type
        self.action_space = args.action_space
        
        self.load_weights = getattr(args, "load_weights", True)
        if not self.load_weights:
            print("You are not loading the pretrained weights of transformer.")

        print(self.action_type, self.action_space)
        print("Add state")

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
            self.sta_mean = np.array(self.StatisticInfo["state"]["mean"])
            self.sta_std = np.array(self.StatisticInfo["state"]["std"])+1e-6
        else:
            raise NotImplementedError
        
        self.context = None
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

        # TODO: hard code here
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
        set_attention_backend(
            attention_impl=self.attention_impl,
            sdpa_backend=self.sdpa_backend,
            flash_attn_version=self.flash_attn_version,
        )
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        diffusion_model_config = dict(self.args.diffusion_model['config'])
        diffusion_model_config["fused_self_attn_qkv"] = self.enable_self_attn_fused_qkv
        diffusion_model_config["enable_action_cross_attn_kv_cache"] = self.enable_action_cross_attn_kv_cache
        diffusion_model_config["enable_action_rope_cache"] = self.enable_action_rope_cache
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=True,
            **diffusion_model_config
        ).to(device, dtype=dtype).eval()

        if self.compile_model:
            if not hasattr(torch, "compile"):
                logger.warning("torch.compile is not available in this PyTorch build; skipping compilation.")
            else:
                try:
                    compile_dynamic = None
                    if self.compile_dynamic == "true":
                        compile_dynamic = True
                    elif self.compile_dynamic == "false":
                        compile_dynamic = False

                    compile_kwargs = dict(
                        mode=self.compile_mode,
                        fullgraph=False,
                    )
                    if compile_dynamic is not None:
                        compile_kwargs["dynamic"] = compile_dynamic

                    logger.info(
                        "Compiling diffusion model with torch.compile(mode=%s, dynamic=%s, target=%s)",
                        self.compile_mode,
                        self.compile_dynamic,
                        self.compile_target,
                    )

                    if self.compile_target in ("action_blocks", "both_blocks"):
                        if not hasattr(self.diffusion_model, "action_blocks"):
                            logger.warning(
                                "compile_target=%s requested, but diffusion model has no action_blocks; skipping action_blocks compilation.",
                                self.compile_target,
                            )
                        else:
                            compiled_action_blocks = []
                            for block in self.diffusion_model.action_blocks:
                                compiled_action_blocks.append(
                                    torch.compile(block, **compile_kwargs)
                                )
                            self.diffusion_model.action_blocks = nn.ModuleList(compiled_action_blocks)

                    if self.compile_target in ("video_blocks", "both_blocks"):
                        if not hasattr(self.diffusion_model, "blocks"):
                            logger.warning(
                                "compile_target=%s requested, but diffusion model has no blocks; skipping video_blocks compilation.",
                                self.compile_target,
                            )
                        else:
                            compiled_video_blocks = []
                            for block in self.diffusion_model.blocks:
                                compiled_video_blocks.append(
                                    torch.compile(block, **compile_kwargs)
                                )
                            self.diffusion_model.blocks = nn.ModuleList(compiled_video_blocks)

                    if self.compile_target in ("action_blocks", "video_blocks", "both_blocks"):
                        pass
                    else:
                        self.diffusion_model = torch.compile(
                            self.diffusion_model,
                            **compile_kwargs,
                        )
                except Exception:
                    logger.exception("torch.compile failed; continuing with the eager diffusion model.")

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
            enable_context_null_cache=self.enable_context_null_cache,
        )


    @torch.no_grad()
    def play(
        self, obs, prompt="Action", state=None,
        num_inference_steps=5, execution_step=99, shift=1.0, sample_solver='unipc',
        gripper_states=None, video_save_folder=None, video_sampling_steps=1, joint_denoising=False,
        reset_context=True, *args, **kwargs,
    ):
        """

        Input:
            obs: One of the followings
                1. torch.tensor of shape: {v, 3, h, w}, ranging from -1 to 1
                2. np.array of shape: {v, h, w, 3}, ranging from 0 to 255 (np.uin8)
            prompt: task description
            execution_step: excution step of the past play
            state:
                shape: np.array of shape: {C};
                action space: the pose of end effector, 2 * {xyz+quaternion(with order xyzw)}, the coordinate origin of each eef pose is the current Arm Base link.
            gripper_states: np.array of shape: {2}

        Output:
            action:
                shape: np.array of shape: {T, C}
                action space: the pose of end effector, 2 * {xyz+quaternion(with order xyzw)}, the coordinate origin of each eef pose is the current Arm Base link.
        """

        if reset_context:
            self.context = None

        state = torch.tensor(state).unsqueeze(0) # C->1,C
        gripper_states = torch.tensor(gripper_states).unsqueeze(0) # C->1,C

        if self.action_space == "eef6d":

            ### from eef to eef6d
            state_rot_l_6d = quaternion_to_rotation_6d(state[:, 3:7])
            state_rot_r_6d = quaternion_to_rotation_6d(state[:, 10:14])

            state = torch.cat((
                state[:,:3], state_rot_l_6d,
                gripper_states[:,:gripper_states.shape[-1]//2],
                state[:,7:10], state_rot_r_6d,
                gripper_states[:,gripper_states.shape[-1]//2:]
            ),dim=-1)

        else:
            raise NotImplementedError

        assert execution_step >= 1 and execution_step <= 100, "execution_step should be in [1, 100]"

        if obs.dtype == np.uint8:
            ### obs / 255 * 2 - 1
            obs = obs.astype(np.float32) / 127.5 - 1
            obs = np.transpose(obs, (0,3,1,2))

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)

        v, c, h, w = obs.shape
        obs = obs.to(self.device, dtype=self.dtype).unsqueeze(2).transpose(0,1)  # v,c,h,w -> c v t h w
            
        if self.norm_type == "meanstd":
            ### C -> 1,1,C
            sta_mean = self.sta_mean[None, :]
            sta_std = self.sta_std[None, :]
            normed_state = np.expand_dims((state-sta_mean)/sta_std, axis=0)
        else:
            raise NotImplementedError

        history_action_state = torch.from_numpy(normed_state).to(self.device, dtype=self.dtype)
        assert(len(history_action_state.shape) == 3 and history_action_state.shape[-1]==self.action_dim)

        negative_prompt = ''        

        if not joint_denoising: # For action-predcition only
            pred_all = self.pipeline.infer(
                prompt,
                obs,
                self.chunk,
                sampling_steps=num_inference_steps,
                return_video=False,
                return_action=True,
                current_state=history_action_state,
                action_chunk=self.action_chunk,
                action_dim=self.action_dim,
                shift=shift,
                sample_solver=sample_solver,
                context=self.context,
                return_dict=True,
            )
            self.context = pred_all['context']
            ### 1,t,c
            actions_pred = pred_all['action'].detach().cpu()

        else: # For simultaneously video-generation and action-prediction
            pred_all = self.pipeline.infer_cotrain(
                prompt,
                obs,
                self.chunk,
                sampling_steps=num_inference_steps,
                return_video=True,
                return_action=True,
                current_state=history_action_state,
                action_chunk=self.action_chunk,
                action_dim=self.action_dim,
                shift=shift,
                sample_solver=sample_solver,
                video_sampling_steps=video_sampling_steps,
            )

            video_pred = pred_all['video']

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(video_save_folder, exist_ok=True)
            save_video(
                torch.cat(video_pred, dim=-1).cpu(),
                os.path.join(video_save_folder, f"{timestamp}.mp4"), fps=int(30/((self.action_chunk-1)/(self.chunk-1)))
            )

            actions_pred = pred_all['action'].detach().cpu()


        ### original state: 1,1,C
        state = state.unsqueeze(0)


        ### for dual-arm
        gripper_dim = self.gripper_dim
        arm_dim = (self.action_dim - 2*self.gripper_dim)//2


        ### prediction post-processing
        if self.action_type == "absolute":
            final_actions_pred = actions_pred[:, :execution_step, :] * self.act_std + self.act_mean
        elif self.action_type == "relative":
            final_actions_pred = actions_pred[:, :execution_step, :]
            final_actions_pred = final_actions_pred * self.act_std + self.act_mean
            if self.action_space == "eef6d":
                action_ = torch.cat((
                    final_actions_pred[:, :, :arm_dim],
                    final_actions_pred[:, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim]
                ), dim=-1)[0]
                state_ = torch.cat((state[:, :, :arm_dim], state[:, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim]), dim=-1)[0]
                abs_action = rela_eef_to_abs(action_, state_)
                final_actions_pred[0, :, :arm_dim] = abs_action[:, :arm_dim]
                final_actions_pred[0, :, arm_dim+gripper_dim:2*arm_dim+gripper_dim] = abs_action[:, arm_dim:]
        else:
            raise NotImplementedError

        final_actions_pred = final_actions_pred[0].data.cpu().numpy()

        return final_actions_pred


    def reset(self):
        self.context = None
        pass

