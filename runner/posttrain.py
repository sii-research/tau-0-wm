import os, random, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from pathlib import Path
from typing import Any, Dict, List

from datetime import datetime, timedelta
import argparse
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib

from yaml import load, dump, Loader, Dumper, safe_load
import numpy as np
from tqdm import tqdm
import torch
import time
# torch.set_num_threads(96)
# torch.set_num_interop_threads(96)

import torch.nn.functional as F

from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
import glob

# ----------------------------------------------------
from utils.model_utils import load_diffusion_model, count_model_parameters, unwrap_model, load_checkpoints
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video
from utils.config_utils import expand_env_vars

# ----------------------------------------------------
from utils.data_utils import apply_color_jitter_to_video

# ----------------------------------------------------
from utils.extra_utils import save_two_tensors_by_channel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import gc
from models.wan_2_2_models.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
logger = get_logger("wm_runner")
logger.setLevel(LOG_LEVEL)


def sample_timestep_indices(
    batch_size: int,
    num_train_timesteps: int,
    sample_mode: str,
    logit_mean: float,
    logit_std: float,
    device: torch.device,
) -> torch.Tensor:
    if sample_mode == "uniform":
        return torch.randint(0, num_train_timesteps, (batch_size,), device=device)

    # logit-normal: sample z~N(mu,sigma), u=sigmoid(z)
    z = torch.randn(batch_size, device=device) * logit_std + logit_mean
    u = torch.sigmoid(z)
    idx = (u * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
    return idx


def latent_seq_len(latent: torch.Tensor, patch_size: tuple, sp_size: int = 1) -> int:
    # latent: [B, C, F, H, W]
    seq = (latent.shape[2] * latent.shape[3] * latent.shape[4]) // (patch_size[1] * patch_size[2])
    return int(math.ceil(seq / sp_size) * sp_size)


def build_timestep_map(cond_mask: torch.Tensor, t_scalar: torch.Tensor, seq_len: int, patch_size: tuple) -> torch.Tensor:
    # cond_mask: [B, C, F, H, W], t_scalar: scalar tensor
    ts = (cond_mask[:,:1,::patch_size[0], ::patch_size[1], ::patch_size[2]] * t_scalar).flatten(1)
    return ts


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None



class Trainer:

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        
        self.config_file = config_file
        
        cd = expand_env_vars(safe_load(open(config_file, "r")))
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args


        ### train_mode: "vlm+video+action"
        self.args.train_mode = [_train_mode.strip().lower() for _train_mode in self.args.train_mode.split("+")]


        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')

        self.state = State()

        self.tokenizer = None
        self.text_encoder = None
        self.diffusion_model = None
        self.unet = None
        self.vae = None
        self.scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        
        self.state.accelerator.wait_for_everyone()
        
        if self.state.accelerator.is_main_process:

            self.save_folder = os.path.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):
                self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
            os.makedirs(self.save_folder, exist_ok=True)

            args_dict = vars(deepcopy(self.args))
            for k, v in args_dict.items():
                args_dict[k] = str(v)
            with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
                json.dump(args_dict, file, indent=4, sort_keys=False)
            
            if to_log:
                self.writer = SummaryWriter(log_dir=self.save_folder)
            else:
                self.writer = None

            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor([len(save_folder_bytes)], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
        else:
            folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.empty(folder_len_tensor.item(), dtype=torch.uint8, device=self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
            self.save_folder = bytes(folder_tensor.tolist()).decode()

        self.state.accelerator.wait_for_everyone()
        init_logging(self.save_folder, rank=self.state.accelerator.process_index)
        
        self.global_rank = self.state.accelerator.process_index
        self.world_size = self.state.accelerator.num_processes


    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_config = {
            "find_unused_parameters": False,
            "bucket_cap_mb": 100,
            "gradient_as_bucket_view": True,
        }
        ddp_config.update(getattr(self.args, "ddp_kwargs", {}) or {})
        self.args.ddp_kwargs = ddp_config
        ddp_kwargs = DistributedDataParallelKwargs(**ddp_config)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            grad_accum = self.args.gradient_accumulation_steps

            train_batch_size = per_device_bs * world_size * grad_accum
            self.args.deepspeed["train_batch_size"] = train_batch_size
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed,
                gradient_accumulation_steps=grad_accum
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.state.weight_dtype = weight_dtype


    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(self.state.accelerator.state, main_process_only=False)
        

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir


    def prepare_dataset_new(self, dataset_configs, public_args):
        """
        A list of yaml showing the config of dataset.
        """
        datasets = []
        for dataset_config in dataset_configs:
            with open(dataset_config) as f:
                config = expand_env_vars(safe_load(f))
            data_class = config['data_class']
            data_class_path = config['data_class_path']
            
            dataset_class = import_custom_class(data_class, data_class_path)
            
            data_args = config['data']
            data_args.update(public_args)
            _dataset = dataset_class(**data_args)
            datasets.append(_dataset)
        
        return torch.utils.data.ConcatDataset(datasets)
    
    
    def prepare_dataset(self) -> None:
        public_args = self.args.data.get("public_args", {})
        train_data_configs = None
        if 'train' in self.args.data:
            if isinstance(self.args.data['train'], str) and os.path.exists(self.args.data['train']):
                with open(self.args.data['train'], 'r', encoding="utf-8") as f:
                    train_data_configs = [line.strip() for line in f if line.strip()]
            elif isinstance(self.args.data['train'], list):
                train_data_configs = self.args.data['train']
            
        if train_data_configs is not None:
            self.train_dataset = self.prepare_dataset_new(train_data_configs, public_args)
        else:
            raise NotImplementedError
        
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            **self.get_loader_args(),
        )
        print(f">>>>>>>>>>>>>Total Train Frames: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")


        if 'val' in self.args.data:
            val_data_configs = None
            if isinstance(self.args.data['val'], str) and os.path.exists(self.args.data['val']):
                with open(self.args.data['val'], 'r', encoding="utf-8") as f:
                    val_data_configs = [line.strip() for line in f if line.strip()]
            elif isinstance(self.args.data['val'], list):
                val_data_configs = self.args.data['val']
            if val_data_configs is not None:
                self.val_dataset = self.prepare_dataset_new(val_data_configs, public_args)
                self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, 
                batch_size=self.args.batch_size, 
                shuffle=getattr(self.args, "val_shuffle", False),
                # collate_fn=wm_train_collate,
                )
                print(f">>>>>>>>>>>>>Total Validatoin Frames: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")
            else:
                raise NotImplementedError


    def get_loader_args(self):
        args = {"batch_size": self.args.batch_size,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.pin_memory,
                "persistent_workers": False,
                "prefetch_factor": getattr(self.args, "prefetch_factor", None),
                "shuffle": True,
                "drop_last": True}
        
        return args
    
    
    def prepare_models(self):
        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype
        
        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        
        if getattr(self.args, 'vae_path', False):
            vae_path = self.args.vae_path
        else:
            vae_path = os.path.join(
                self.args.pretrained_model_name_or_path,
                getattr(self.args, "vae_checkpoint", "Wan2.2_VAE.pth"),
            )
        self.vae = vae_class(vae_pth=vae_path, device=device, dtype=dtype)

        # Wan-2.2 VAE down-sampling factors.
        self.SPATIAL_DOWN_RATIO = 16
        self.TEMPORAL_DOWN_RATIO = 4
        
        logger.info(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        logger.info(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')

        self.vae.vae_encoding_mode = getattr(self.args, "vae_encoding_mode", "3dvae")
        
        
        
        if getattr(self.args, "latest_log_dir", None) is not None:

            # optimizer_dir = os.path.join(self.args.output_dir, "latest_accelerator_state")
            # self.state.accelerator.load_state(
            #     optimizer_dir, load_kwargs={"weights_only": False}
            # )
            
            checkpoint_path = glob.glob(
                os.path.join(self.args.latest_log_dir, "diffusion_pytorch_model.safetensors")
            )
            checkpoint_path += glob.glob(
                os.path.join(self.args.latest_log_dir, "diffusion_pytorch_model.bin")
            )
            checkpoint_path += glob.glob(
                self.args.latest_log_dir
            )
            assert(len(checkpoint_path) >= 1)
            checkpoint_path = checkpoint_path[0]
        else:
            checkpoint_path = self.args.diffusion_model['model_path']

        ### Load Tokenizer
        
        if getattr(self.args, "textenc_class_path", "transformers").lower() != "none":
            textenc_class = import_custom_class(
                self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
            )
            self.text_encoder = textenc_class(
                dtype=dtype,
                device=device,
                **self.args.text_encoder
            )
        else:
            print("Got None Tokenizer")
            self.tokenizer = None
            self.text_encoder = None
            self.text_uncond = None
            self.uncond_prompt_embeds = None
            self.uncond_prompt_attention_mask = None


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=checkpoint_path,
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model:{total_params}')
        self.diffusion_patch_size = self.diffusion_model.patch_size
        
        if hasattr(self.diffusion_model, "vlm_interface"):
            self.vlm_interface = self.diffusion_model.vlm_interface
        else:
            self.vlm_interface = None

        ### Load Diffuser Scheduler
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=5.0,
            use_dynamic_shifting=False,
        )
        self.action_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=False
        )

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )

        self.state.accelerator.wait_for_everyone()

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        components_to_disable_grads = []
            
        for component in components_to_disable_grads:
            if component is not None:
                component.requires_grad_(False)

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True


    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        train_mode = self.args.train_mode

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
            cast_training_params([self.diffusion_model], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        diffusion_model_trainable_params = []

        ### train_mode: "vlm+video+action"

        for name, param in self.diffusion_model.named_parameters():
            
            if name.find("action_")>=0:
                if "action" in train_mode:
                    diffusion_model_trainable_params.append(param)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            elif name.find("vlm_interface")>=0:
                if "vlm" in train_mode:
                    diffusion_model_trainable_params.append(param)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            else:
                if "video" in train_mode:
                    diffusion_model_trainable_params.append(param)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        num_trainable_params = sum(p.numel() for p in diffusion_model_trainable_params)
        print(f'Total trainable parameters: {num_trainable_params}')

        diffusion_model_parameters_with_lr = {
            "params": diffusion_model_trainable_params,
            "lr": self.state.learning_rate,
        }
        params_to_optimize = [diffusion_model_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in diffusion_model_trainable_params)

        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit = self.args.optimizer_8bit,
            use_torchao = self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        

    def prepare_for_training(self):
        self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        self.load_checkpoint()

    def load_checkpoint(self):
        if getattr(self.args, "optimizer_path", None) is not None:
            
            self.state.accelerator.load_state(self.args.optimizer_path)
            
            print("Successfully load optimizer state------------")


    def prepare_trackers(self):
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "model_train"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)


    def train(self):
        logger.info("Starting training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.train_dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        
        first_epoch = getattr(self.args, "latest_epoch", 0)
        global_step = getattr(self.args, "latest_global_step", 0)
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )


        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        action_scheduler_sigmas = self.action_scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator
        
        if getattr(self.args, "gc_disable", False):
            print("Your are disabling gc collect, pay attention to the memory usage.")
            gc.disable()

        if getattr(self.args, "repeat_in_batch", 1) > 1:
            repeat_in_batch = self.args.repeat_in_batch
        else:
            repeat_in_batch = 1

        # loss spikes
        anomalies = []
        
        local_steps = len(self.train_dataloader)
        steps_tensor = torch.tensor([local_steps], device=accelerator.device, dtype=torch.int64)
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.MIN)
        min_steps_per_epoch = int(steps_tensor.item())

        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            self.diffusion_model.train()

            running_loss = 0.0
            
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            data_iter = iter(self.train_dataloader)
            
            for step in range(min_steps_per_epoch):
                logger.debug(f"Starting step {step + 1}")
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                        
                logs = {}
                with accelerator.accumulate([ self.diffusion_model ]):
                    
                    video = batch['video']

                    if isinstance(video, torch.Tensor):
                        video = video.to(accelerator.device, dtype=weight_dtype).contiguous()
                    else:
                        raise NotImplementedError

                    # shape: {b, c, v, t, h, w}; ranging from -1 to 1
                    batch_size, c, n_view, _, h, w = video.shape
                    video = rearrange(video, 'b c v t h w -> (b v) c t h w')

                    # here we use color jitter to the video, with different views or different batches different jitter
                    if self.args.use_color_jitter:
                        future_video = apply_color_jitter_to_video(video)
                    else:
                        future_video = video


                    # get the shape params
                    _, _, raw_frames, raw_height, raw_width = future_video.shape

                    if self.vae.vae_encoding_mode == "3dvae":
                        latent_frames = (raw_frames-1) // self.TEMPORAL_DOWN_RATIO + 1
                    else:
                        raise NotImplementedError

                    latent_height = raw_height // self.SPATIAL_DOWN_RATIO
                    latent_width = raw_width // self.SPATIAL_DOWN_RATIO

                    if self.args.return_action and self.args.noisy_video and not self.args.return_video:
                        latent_future_frame = (future_video.shape[2] - 1) // self.TEMPORAL_DOWN_RATIO + 1
                        future_video = future_video[:, :, :1]
                    
                    with torch.no_grad():
                        future_video_latents = self.vae.encode(future_video, chunk=int(getattr(self.args, "vae_encode_chunk", 3)))

                    future_video_latents = torch.stack(future_video_latents, dim=0) #.to(dtype=weight_dtype)
                    
                    future_video_latents = rearrange(future_video_latents, '(b v) c f h w -> b v c f h w', b=batch_size, v=n_view, h=latent_height, w=latent_width)

                    if self.args.return_action and self.args.noisy_video and not self.args.return_video:
                        future_video_latents = future_video_latents.repeat(
                            1, 1, 1, latent_future_frame, 1, 1
                        )
                        
                    if repeat_in_batch > 1:
                        future_video_latents = future_video_latents.repeat(repeat_in_batch, 1, 1, 1, 1, 1)
                        batch_size = batch_size * repeat_in_batch
                    
                    future_video_latents = rearrange(future_video_latents, 'b v c m h w -> b c m h (v w)')

                    latents = future_video_latents
                    cond_mask = torch.ones_like(latents)
                    cond_mask[:, :, 0] = 0.0
                    noise = torch.randn_like(latents) * cond_mask + latents * (1-cond_mask)
                    
                    timestep_indices = sample_timestep_indices(
                        batch_size=batch_size,
                        num_train_timesteps=1000,  # same as wan
                        sample_mode="uniform",
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        device=accelerator.device,
                    )
                    
                    sigma = scheduler_sigmas[timestep_indices].to(weight_dtype).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                    
                    if self.args.return_action and self.args.noisy_video and not self.args.return_video:
                        sigma = sigma * 0 + 1.0   # set to 1.0 for real action mode
                    
                    noisy_latents = (1 - sigma) * latents + sigma * noise
                    
                    target = noise - latents
                    
                    seq_len = latent_seq_len(latents, self.diffusion_patch_size, sp_size=1)
                    t_scalar = sigma * 1000
                    t_map = build_timestep_map(cond_mask, t_scalar, seq_len, self.diffusion_patch_size)


                    captions = batch['caption']
                    if repeat_in_batch > 1:
                        assert(isinstance(captions, (list, tuple)))
                        captions = captions * repeat_in_batch
                    captions = [
                        "" if random.random() < self.args.caption_dropout_p else cap
                        for cap in captions
                    ]

                    context = self.text_encoder(captions, accelerator.device)
                    
                    if self.args.return_action:
                        state = batch['state'].to(accelerator.device, dtype=weight_dtype).contiguous()  # shape b,1,c
                        actions = batch['actions'].to(accelerator.device, dtype=weight_dtype).contiguous()  # shape b,l,c

                        if repeat_in_batch > 1:
                            state = state.repeat(repeat_in_batch,1,1)
                            actions = actions.repeat(repeat_in_batch,1,1)
                            
                        noise_actions = torch.randn_like(actions)
                    
                        action_timestep_indices = sample_timestep_indices(
                            batch_size=batch_size,
                            num_train_timesteps=1000,  # same as wan
                            sample_mode="uniform",
                            logit_mean=self.args.flow_logit_mean,
                            logit_std=self.args.flow_logit_std,
                            device=accelerator.device,
                        )
                        
                        action_sigma = action_scheduler_sigmas[action_timestep_indices].to(weight_dtype).unsqueeze(1).unsqueeze(2)
                        
                        noisy_actions = (1 - action_sigma) * actions + action_sigma * noise_actions
                        
                        action_target = noise_actions - actions

                        action_t_scalar = action_sigma * 1000
                        action_timestep = action_t_scalar.repeat(1,actions.shape[1],1)
                    else:
                        noisy_actions = None
                        action_timestep = None
                        state = None
                    pred_all = self.diffusion_model(
                        list(noisy_latents.unbind(dim=0)),
                        t=t_map,
                        context=context,
                        seq_len=seq_len,
                        action_states=noisy_actions,
                        action_timestep=action_timestep,
                        return_video=True,
                        return_action=self.args.return_action,
                        history_action_state=state,
                    )

                    if self.args.return_video:
                        pred_video = torch.stack(pred_all['video'], dim=0)  # b c t h vw
                        loss_video = (pred_video - target)**2
                        loss_video = loss_video * cond_mask
                        loss_video = loss_video.mean()
                    else:
                        loss_video = 0.

                    if self.args.return_action:
                        loss_action = (pred_all['action']-action_target)**2
                        loss_action = loss_action.mean()
                    else:
                        loss_action = 0.

                    loss = loss_video * getattr(self.args, "video_loss_scale", 1.0) + loss_action * getattr(self.args, "action_loss_scale", 1.0)

                    assert torch.isnan(loss) == False, "NaN loss detected"
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.diffusion_model.zero_grad()

                running_loss += loss.item()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0], }

                if self.args.return_video:
                    logs.update({"loss_video": loss_video.item()})
                if self.args.return_action:
                    logs.update({"loss_action": loss_action.item()})

                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    logger.info(">>> max train step reached")
                    break

                if global_step % self.args.steps_to_log == 0:
                    if accelerator.is_main_process:
                        if self.writer is not None:
                            self.writer.add_scalar("Training Loss", loss.item(), global_step)
                            if self.args.return_video:
                                self.writer.add_scalar("Video loss", loss_video.item(), global_step)
                            if self.args.return_action:
                                self.writer.add_scalar("Action loss", loss_action.item(), global_step)
                    accelerator.wait_for_everyone()

                if global_step % self.args.steps_to_val == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_dir = os.path.join(self.save_folder,f'Validation_step_{global_step}')
                        self.validate(accelerator, model_save_dir, global_step, n_view=n_view, n_chunk=1)
                    accelerator.wait_for_everyone()

                if global_step % self.args.steps_to_save == 0:
                    accelerator.save_state(os.path.join(self.save_folder, "latest_accelerator_state"))
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_to_save = unwrap_model(accelerator, self.diffusion_model)
                        dtype = (
                            torch.float16
                            if self.args.mixed_precision == "fp16"
                            else torch.bfloat16
                            if self.args.mixed_precision == "bf16"
                            else torch.float32
                        )
                        model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
                        model_to_save.save_pretrained(model_save_dir, safe_serialization=False)
                        del  model_to_save
                        
                        tmp_cd = load(open(self.config_file, "r"), Loader=Loader)
                        tmp_cd["latest_log_dir"] = model_save_dir
                        tmp_cd["latest_global_step"] = global_step
                        tmp_cd["latest_epoch"] = epoch
                        tmp_cd["optimizer_path"] = os.path.join(self.save_folder, "latest_accelerator_state")
                        with open(self.config_file, "w") as f_cfg:
                            dump(tmp_cd, f_cfg, Dumper=Dumper)

                    accelerator.wait_for_everyone()
                
                if global_step % 10000 == 0:
                    accelerator.wait_for_everyone()
                    collected = gc.collect()
                    gc.disable()
                    accelerator.wait_for_everyone()
                    
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            if accelerator.is_main_process and self.writer is not None:
                avg_loss = running_loss / len(self.train_dataloader)
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)
                
            accelerator.wait_for_everyone()


        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
            dtype = (
                torch.float16
                if self.args.mixed_precision == "fp16"
                else torch.bfloat16
                if self.args.mixed_precision == "bf16"
                else torch.float32
            )
            model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
            self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=False)


        del self.diffusion_model, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()


    def validate(self, accelerator, model_save_dir, global_step, n_view=1, n_chunk=30, image=None, prompt=None, cap=None, path=None, gt_actions=None, to_log=True):

        os.makedirs(model_save_dir, exist_ok=True)

        pipe = self.pipeline_class(
            self.text_encoder,
            self.vae,
            unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model
        )

        batch = next(iter(self.val_dataloader))

        video = batch['video'][0]   # shape c,v,t,h,w
        
        chunk = video.shape[2]
        
        image = video[:,:,:1].clone()  # shape c,v,t,h,w
        image = image.to(accelerator.device, dtype=self.state.weight_dtype).contiguous()

        prompt = batch['caption'][0]

        c, v, t, h, w = image.shape

        negative_prompt = ''

        num_denois_steps = self.args.num_inference_step
        
        action_chunk = chunk
        if self.args.return_action:
            history_action_state = batch['state'][:1].contiguous()
            gt_actions = batch['actions'][:1]
            action_dim = gt_actions.shape[-1]
            action_chunk = gt_actions.shape[1]

            args = {
                "sample_solver": "euler",
                "shift": 1.0,
            }
            
            preds = pipe.infer(
                prompt,
                image,
                chunk,
                guide_scale=3.0,
                sampling_steps=num_denois_steps,
                return_video=False,
                return_action=self.args.return_action,
                current_state=history_action_state,
                action_chunk=action_chunk,
                action_dim=action_dim,
                **args,
            )
            
            save_two_tensors_by_channel(gt_actions, preds, os.path.join(model_save_dir, "action_pred.png"), "GT", "Pred", ncols=2)

        if self.args.return_video:
            
            preds = pipe.infer(
                prompt,
                image,
                chunk,
                guide_scale=3.0,
                sampling_steps=num_denois_steps,
                return_video=self.args.return_video,
                return_action=False,
            )
            
            cap = 'Validation'
            fps = int(int(getattr(self.args, "basic_fps", 30)) / ((action_chunk-1)/(chunk-1)))

            device_id = str(accelerator.device).lower().replace("cuda:", "")

            save_video(rearrange(video.data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{cap}_gt_{device_id}.mp4'), fps=fps)

            video = torch.cat(preds, dim=-1)

            save_video(video.cpu(), os.path.join(model_save_dir, f'{cap}_{device_id}.mp4'), fps=fps)
            
        if to_log:
            self.writer.add_text(f'step_{global_step}/{cap} prompt:', prompt, global_step)
