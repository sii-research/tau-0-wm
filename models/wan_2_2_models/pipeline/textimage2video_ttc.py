# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial
from typing import Any
import copy

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models.wan_2_2_models.transformers.model import WanModel
from models.wan_2_2_models.text_encoder.t5 import T5EncoderModel
from models.wan_2_2_models.vae.vae2_2 import Wan2_2_VAE
from models.wan_2_2_models.scheduler.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from models.wan_2_2_models.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from einops import rearrange



def mark_compile_step_begin():
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "cudagraph_mark_step_begin"):
        compiler.cudagraph_mark_step_begin()

def sp_attn_forward():
    raise NotImplementedError

def sp_dit_forward():
    raise NotImplementedError

def shard_model():
    raise NotImplementedError

def get_world_size():
    raise NotImplementedError

def masks_like(tensor, zero=False, generator=None, p=0.2):
    # shape c, v, t, h w
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, :, 0] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, :, 0]).exp()
                    v[:, :, 0] = torch.zeros_like(v[:, :, 0])
                else:
                    u[:, :, 0] = u[:, :, 0]
                    v[:, :, 0] = v[:, :, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, :, 0] = torch.zeros_like(u[:, :, 0])
                v[:, :, 0] = torch.zeros_like(v[:, :, 0])

    return out1, out2



def masks_like_raw(tensor, zero=False, generator=None, p=0.2):
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, 0] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, 0]).exp()
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, 0] = torch.zeros_like(u[:, 0])
                v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


def sync_current_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class WanTI2V:

    def __init__(
        self,
        text_encoder,
        vae,
        diffusion_model,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        enable_context_null_cache=True,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.enable_context_null_cache = enable_context_null_cache

        self.num_train_timesteps = 1000
        self.param_dtype = torch.bfloat16

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = text_encoder

        self.vae_stride = [4,16,16]
        self.patch_size = [1,2,2]
        self.vae = vae

        self.model = diffusion_model
        # self.model = self._configure_model(
        #     model=self.model,
        #     use_sp=use_sp,
        #     dit_fsdp=dit_fsdp,
        #     shard_fn=shard_fn,
        #     convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = ""

        self._text_context_cache = {}

    def _get_text_context_cache_key(self, text):
        device_index = self.device.index if self.device.index is not None else -1
        return (text, self.device.type, device_index, self.t5_cpu)

    def _encode_single_text(self, text, offload_model=False, use_cache=False):
        cache_key = self._get_text_context_cache_key(text)
        use_cache = use_cache and self.enable_context_null_cache
        if use_cache and cache_key in self._text_context_cache:
            return self._text_context_cache[cache_key]

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([text], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([text], torch.device('cpu'))
            context = [t.to(self.device) for t in context]

        if use_cache:
            self._text_context_cache[cache_key] = context
        return context

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        # TODO: disable this for training inference
        # if dist.is_initialized():
        #     dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    @torch.no_grad()
    def infer(self,
            input_prompt,
            img,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=1.0,
            n_prompt="",
            seed=-1,
            offload_model=False,

            return_video=False,
            return_action=True,

            current_state=None,
            action_chunk=None,
            action_dim=None,

            N_proposals=1,
            
            get_RCS=False,
            RCS_timestep_ratios=[0.5],

            latent=None,

            fix_video_t=None,
            
            context=None,
            return_dict=False,

        ):
        r"""
        
        Generate N action predictions with test-time computation
        
        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            return_video (`bool`, *optional*, defaults to False):
                If True, generate video
            return_action (`bool`, *optional*, defaults to False):
                If True, generate action
            current_state (`torch.tensor`):
                Current robot state. Shape: [1, 1, C]
            action_chunk (`int`):
                The number of actions in a chunk
            action_dim (`int`):
                The number of action dimension
                
            N_proposals (`int`):
                The number of action proposals
            get_RCS (`bool`, *optional*, defaults to True):
                If True, return Re-denoising Consistency Score
            RCS_timestep_ratios (`list`, *optional*, defaults to [0.5]):
                Timesteps used for rcs calculation
                
            latent (`torch.tensor`, *optional*, defaults to None):
                Video latent after specific denoising step. If it is not None, `fix_video_t` should be given.
            fix_video_t (`int`, *optional*, defaults to None):
                Related denoising step of the pre-denoised video latent.

        """
        
        assert (not return_video) and return_action
        
        if latent is not None:
            assert(fix_video_t is not None)

        # preprocess
        C, V, _, H, W = img.shape
        img = list(img.unbind(dim=1))
        z = self.vae.encode(img)
        z = torch.stack(z, dim=1)    # C, V, T, H, W
        z = [rearrange(z, "c v t h w -> c t h (v w)")]

        F = frame_num

        F_latent = (F - 1) // self.vae_stride[0] + 1

        seq_len = F_latent * (
            H // self.vae_stride[1]) * V * (W // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        noise = torch.randn(
            self.vae.model.z_dim,
            F_latent,
            z[0].shape[-2],
            z[0].shape[-1],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if return_action:
            noise_action = torch.randn(N_proposals, action_chunk, action_dim, 
                                    dtype=self.param_dtype,generator=seed_g,device=self.device)
            if current_state is not None:
                current_state = current_state.repeat(N_proposals,1,1).to(self.device, dtype=self.param_dtype)
        else:
            noise_action = None

        if context is None:
            context = self._encode_single_text(
                input_prompt,
                offload_model=offload_model,
                use_cache=False,
            )
        should_encode_context_null = return_video or not self.enable_context_null_cache
        context_null = None
        if should_encode_context_null:
            context_null = self._encode_single_text(
                n_prompt,
                offload_model=offload_model,
                use_cache=True,
            )


        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=shift,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=shift,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            elif sample_solver == 'euler':
                sample_scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=shift,
                    use_dynamic_shifting=False
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            if get_RCS:
                action_scheduler_sigmas = sample_scheduler.sigmas.clone().to(device=self.device, dtype=self.param_dtype)

            if fix_video_t is not None:
                sigmas = sample_scheduler.sigmas.clone().to(device=self.device, dtype=self.param_dtype)
                print("sigmas[fix_video_t]: ", sigmas[fix_video_t])


            mask1, mask2 = masks_like_raw([noise], zero=True)

            if latent is None:
                latent = (1. - mask2[0]) * z[0] + mask2[0] * noise

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }
            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            action_states = noise_action.clone()

            video_states_buffer=None
            for i, t in enumerate(tqdm(timesteps)):
                compute_video = i==0 or return_video
                store_buffer = i==0 and not return_video


                if return_action:

                    if video_states_buffer is None:

                        latent_model_input = [latent.to(self.device)] # * N_proposals

                        if fix_video_t is None:
                            vid_t = torch.tensor(1000)
                        else:
                            vid_t = timesteps[fix_video_t]

                        timestep = [vid_t]
                        timestep = torch.stack(timestep).to(self.device)
                        temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                        temp_ts = torch.cat([
                            temp_ts,
                            temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                        ])
                        timestep = temp_ts.unsqueeze(0) #.repeat(N_proposals,1)
                                                
                        noise_pred_basic = self.model(
                            latent_model_input,
                            timestep,
                            return_video=compute_video,
                            store_buffer=store_buffer,
                            video_states_buffer=None,
                            return_action=False,
                            **arg_c,
                        )
                        video_states_buffer = noise_pred_basic['video_states_buffer']
                        video_states_buffer = [_.unsqueeze(0).repeat(N_proposals,1,1,1).flatten(0,1) for _ in video_states_buffer]

                    else:
                        latent_model_input = None
                        timestep = None


                    action_timestep = [t]
                    action_timestep = torch.stack(action_timestep).to(self.device).unsqueeze(1).repeat(N_proposals, action_chunk)

                    noise_pred = self.model(
                        None,
                        None,
                        return_video=False,
                        store_buffer=False,
                        video_states_buffer=video_states_buffer,

                        action_states=action_states,
                        action_timestep=action_timestep,
                        history_action_state=current_state,
                        return_action=True,

                        **arg_c,
                    )
                    if offload_model:
                        torch.cuda.empty_cache()

                    action_states = sample_scheduler.step(
                        noise_pred['action'],
                        t,
                        action_states,
                        return_dict=False,
                        generator=seed_g
                    )[0]

            if get_RCS:

                action_timestep_index = [ int(len(timesteps)*_) for _ in RCS_timestep_ratios ]
                n_t = len(action_timestep_index)

                action_timestep = [ timesteps[_] for _ in action_timestep_index ]
                action_timestep = torch.stack(action_timestep).to(self.device).unsqueeze(1).unsqueeze(1).repeat(1, N_proposals, action_chunk).flatten(0,1) # n_t*n, t
                
                action_sigma = torch.stack([action_scheduler_sigmas[_] for _ in action_timestep_index], dim=0).view(n_t,1,1,1).to(self.param_dtype) # n_t,1,1,1
                
                new_noise_action = torch.randn_like(noise_action[:1]).repeat(N_proposals,1,1)
                noisy_actions = (1 - action_sigma) * action_states.unsqueeze(0) + action_sigma * new_noise_action.unsqueeze(0)
                noisy_actions = noisy_actions.flatten(0,1) # n_t*n, t, c
                
                arg_c['context'] = [context[0]]*n_t*N_proposals

                noise_pred_sid = self.model(
                    latent_model_input,
                    timestep,
                    return_video=False,
                    store_buffer=False,
                    video_states_buffer=[_.unsqueeze(0).repeat(n_t,1,1,1).flatten(0,1) for _ in video_states_buffer],
                    action_states=noisy_actions,
                    action_timestep=action_timestep,
                    history_action_state=current_state.unsqueeze(0).repeat(n_t,1,1,1).flatten(0,1),
                    return_action=True,
                    **arg_c,
                )
                action_target = new_noise_action - action_states
                rcs = (noise_pred_sid['action'].view(n_t, N_proposals, action_chunk, -1)-action_target.unsqueeze(0))**2
                rcs = rcs.mean(-2).mean(-1).mean(0)
                rcs *= -1
                                

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if return_video:
                if self.rank == 0:
                    x0 = list(torch.cat([rearrange(x, "c t h (v w) -> v c t h w", v=V) for x in x0], dim=0).unbind(dim=0))
                    videos = self.vae.decode(x0)

        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        # if dist.is_initialized():
        #     dist.barrier()


        if return_dict:
            results = {'context': context}
            results.update({"action": action_states})
            if get_RCS:
                results.update({"rcs": rcs})
            return results if self.rank==0 else None

        if get_RCS:
            outputs = [action_states, rcs]
        else:
            outputs = [action_states, ]
        return outputs if self.rank == 0 else None
