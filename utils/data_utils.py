from PIL import Image
import torch
from einops import rearrange

import torchvision.transforms as transforms

def apply_color_jitter_to_video(tensor, jitter=None):
    """
    inputs:
        tensor (torch.Tensor): {b,c,t,h,w}, range [-1, 1]
        jitter (ColorJitter) : torchvision.transforms.ColorJitter
    output:
        augmented video tensor
    """
    B, C, T, H, W = tensor.shape
    assert C == 3
    if jitter is None:
        # jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        jitter = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1)
    tensor = (tensor + 1.0) / 2.0
    tensor = rearrange(tensor, 'b c t h w -> (b t) c h w').contiguous()
    
    tensor = jitter(tensor)
    tensor = rearrange(tensor, '(b t) c h w -> b c t h w', b=B).contiguous()
    tensor = tensor * 2.0 - 1.0
    return tensor
