from typing import Dict
import json
import os

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import load_file


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def count_model_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def load_index_file(index_filename, mode="safetensors"):
    checkpoint_folder = os.path.split(index_filename)[0]
    with open(index_filename) as f:
        index = json.loads(f.read())

    if "weight_map" in index:
        index = index["weight_map"]
    checkpoint_files = sorted(list(set(index.values())))
    checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]
    state_dict = {}
    for checkpoint_file in checkpoint_files:
        if mode == "safetensors":
            state_dict.update(load_file(checkpoint_file))
        elif mode == "bin":
            state_dict.update(torch.load(checkpoint_file, map_location="cpu"))
    return state_dict


def _find_mismatched_keys(
    state_dict,
    model_state_dict,
    loaded_keys,
):
    mismatched_keys = []
    for checkpoint_key in loaded_keys:
        model_key = checkpoint_key

        if (
            model_key in model_state_dict
            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
        ):
            mismatched_keys.append(
                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
            )
            del state_dict[checkpoint_key]

    return mismatched_keys


def load_checkpoints(model, pretrained_ckpt, strict=False, ignore_mismatched_sizes=True):
    """
    Load safetensors model state dict file.
    """

    # In this case we have many shards to load
    if os.path.isdir(pretrained_ckpt):
        if os.path.exists(os.path.join(pretrained_ckpt, "diffusion_pytorch_model.safetensors.index.json")):
            state_dict = load_index_file(os.path.join(pretrained_ckpt, "diffusion_pytorch_model.safetensors.index.json"), mode="safetensors")
        else:
            state_dict = load_index_file(os.path.join(pretrained_ckpt, "diffusion_pytorch_model.bin.index.json"), mode="bin")
    # in this case we need give the file path
    elif pretrained_ckpt.endswith("safetensors"):
        state_dict = load_file(pretrained_ckpt)
    else:
        state_dict = torch.load(pretrained_ckpt, map_location="cpu")

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        if ignore_mismatched_sizes:
            model_state_dict = model.state_dict()
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                list(state_dict.keys()),
            )
        else:
            mismatched_keys = []
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(">>> mismatched_keys: %s" % mismatched_keys)
        print(">>> missing: %s" % missing)
        print(">>> unexpected: %s" % unexpected)
    print(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)



def load_diffusion_model(model_cls, model_dir, load_weights=True, **kwargs):
    model = model_cls(**kwargs)
    if load_weights:
        print("Load pretrained model weights: ", model_dir)
        load_checkpoints(model, pretrained_ckpt=model_dir)
    return model
