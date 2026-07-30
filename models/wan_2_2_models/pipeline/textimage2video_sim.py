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

import time



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
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=False,

            return_video=True,
            return_reward=True,
            
            actions=None,
            
            reward_dim=1,       
            
            mem_img=None,

            decode=True,

            return_latent_timestep=None,
        ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

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
            return_video (`bool`, *optional*, defaults to True):
                If True, generate video
            return_reward (`bool`, *optional*, defaults to True):
                If True, generate reward
            actions (`torch.tensor`):
                Action condition. Shape: [B, T, C]
            reward_dim (`int`, *optional*, defaults to 1):      
                The number of reward dimension
            mem_img (`torch.tensor`, *optional*, defaults to None):
                Input Memory Image tensor. Shape: [3, V, T, H, W]
            decode (`bool`, *optional*, defaults to True):
                If True, return decoded video frames
                If False, return latents
            return_latent_timestep (`int`, *optional*, defaults to None):
                When it is not None, return latents at specific timestep (i.e., i==return_latent_timestep)
        """
        
        assert(return_video)

        # preprocess
        C, V, _, H, W = img.shape
        img = list(img.unbind(dim=1))
        z = self.vae.encode(img)
        z = torch.stack(z, dim=1)    # C, V, T, H, W
        z = [rearrange(z, "c v t h w -> c t h (v w)")]

        N_proposals = actions.shape[0]

        reward_chunk = actions.shape[1]

        if mem_img is not None:
            mem_img = rearrange(mem_img, "c v t h w -> c (v t) h w").unsqueeze(2)
            mem_img = list(mem_img.unbind(dim=1))
            mem_z = self.vae.encode(mem_img)
            mem_z = torch.stack(mem_z, dim=1).squeeze(2)    # C, VT, H, W
            mem_z = rearrange(mem_z, "c (v t) h w -> c t h (v w)", v=V)
            # print("mem_z.shape", mem_z.shape)
            n_mem = mem_z.shape[1]
        else:
            n_mem = 0

        # print("#Memory : ", n_mem)

        F_latent = (frame_num - 1) // self.vae_stride[0] + 1 + n_mem

        seq_len = F_latent * (H // self.vae_stride[1]) * V * (W // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size
        

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        noise = torch.randn(
            N_proposals,
            self.vae.model.z_dim,
            F_latent,
            z[0].shape[-2],
            z[0].shape[-1],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device
        )
        if mem_img is not None:
            assert(z[0].shape[1]==1)
            z[0] = z[0].repeat(1, F_latent, 1, 1)
            z[0][:,:n_mem] = mem_z

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if return_reward:
            noise_reward = torch.randn(N_proposals, reward_chunk, reward_dim, 
                    dtype=self.param_dtype,generator=seed_g,device=self.device)
        else:
            noise_reward = None

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]


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


            if return_reward:
                sampling_sigmas_reward = get_sampling_sigmas(sampling_steps, 1.0)
                sample_scheduler_reward = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1.0,
                    use_dynamic_shifting=False
                )
                reward_timesteps, _ = retrieve_timesteps(
                    sample_scheduler_reward,
                    device=self.device,
                    sigmas=sampling_sigmas_reward)


            # sample videos
            ### b,c,t,h,w
            _, mask2 = masks_like_raw([noise[0]], zero=False)
            mask2 = torch.stack(mask2, dim=0)
            if mem_z is not None:
                mask2[:, :, :n_mem+1] = 0
            else:
                mask2[:, :, :1] = 0

            latent = (1. - mask2) * z[0].unsqueeze(0) + mask2 * noise

            arg_c = {
                'context': [context[0]]*N_proposals,
                'seq_len': seq_len,
                'action_states': actions,
            }
            arg_null = {
                'context': [context[0]]*N_proposals,
                'seq_len': seq_len,
                'action_states': actions*0.0,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            if return_reward:
                reward_states = noise_reward

            with tqdm(total=sampling_steps) as pbar:

                for i, t in enumerate(timesteps):

                    pbar.set_description(f"Processing {i}/{sampling_steps} | Current timestep: {t}")

                    latent_model_input = list(latent.to(self.device).unbind(0))

                    timestep = [t]
                    timestep = torch.stack(timestep).to(self.device)

                    temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                    ])
                    timestep = temp_ts.unsqueeze(0).repeat(N_proposals,1)

                    if return_reward:
                        reward_timestep = [reward_timesteps[i]]
                        reward_timestep = torch.stack(reward_timestep).to(self.device).unsqueeze(1).repeat(N_proposals, reward_chunk)

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, 
                        return_reward=return_reward,
                        reward_states=reward_states if return_reward else None,
                        reward_timestep=reward_timestep if return_reward else None,
                        n_mem=n_mem,
                        **arg_c
                    )
                    noise_pred_cond_vid = torch.stack(noise_pred_cond['video'], dim=0)

                    if offload_model:
                        torch.cuda.empty_cache()

                    if guide_scale != 1:
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep,
                            return_reward=return_reward,
                            reward_states=reward_states if return_reward else None,
                            reward_timestep=reward_timestep if return_reward else None,
                            n_mem=n_mem,
                            **arg_null
                        )
                        noise_pred_uncond_vid = torch.stack(noise_pred_uncond['video'], dim=0)
                        if offload_model:
                            torch.cuda.empty_cache()

                        noise_pred = noise_pred_uncond_vid + guide_scale * (
                            noise_pred_cond_vid - noise_pred_uncond_vid)
                    else:
                        
                        noise_pred = noise_pred_cond_vid


                    latent = sample_scheduler.step(
                        noise_pred,
                        t,
                        latent,
                        return_dict=False,
                        generator=seed_g
                    )[0]
                    latent = (1. - mask2) * z[0].unsqueeze(0) + mask2 * latent # replace memories and current obs with origin inputs

                    if return_reward:
                        reward_states = sample_scheduler_reward.step(
                            noise_pred_cond['reward'],
                            reward_timesteps[i],
                            reward_states,
                            return_dict=False,
                            generator=seed_g
                        )[0]


                    if i == len(timesteps)-1:
                        x0 = latent[:,:,n_mem:] ### drop memories

                    if return_latent_timestep is not None:
                        if i == return_latent_timestep:
                            x0_return = latent[:,:,n_mem:].clone() ### drop memories
                            print("SIM: x0_return timestep: ", t, i)
                            print("SIM: x0_return sigmas  : ", sample_scheduler.sigmas)
                            scheduler_return = copy.deepcopy(sample_scheduler)

                    del latent_model_input, timestep

                    pbar.update(1)

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            x0 = rearrange(x0, "b c t h (v w) -> (b v) c t h w", v=V)
            if decode:
                print("x0.shape: ", x0.shape)
                videos = self.vae.decode(list(x0.unbind(dim=0)))
                videos = [torch.clip(_, min=-1, max=1) for _ in videos]
            else:
                videos = x0
                

        outputs = [videos, ]

        if return_reward:
            outputs.append(reward_states)

        if return_latent_timestep is not None:
            outputs.append(x0_return)
            outputs.append(scheduler_return)


        del sample_scheduler
        del noise, latent
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        return outputs
