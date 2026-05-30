
import asyncio
import http
import logging
import time
import traceback

from web_infer_utils.openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from web_infer_utils.WanPolicy import WanPolicy
import numpy as np
import cv2
import json
import argparse
import socket

import os
import torch
import torch.distributed as dist
from PIL import Image

logger = logging.getLogger(__name__)


import os
import sys


def init_distributed_and_get_device(backend: str = "nccl"):
    """
    return:
        device: torch.device
        is_distributed: bool
        rank: int
        local_rank: int
        world_size: int
    """
    is_distributed = False
    rank = 0
    local_rank = 0
    world_size = 1

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        is_distributed = int(os.environ["WORLD_SIZE"]) > 1
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist_backend = backend
        else:
            device = torch.device("cpu")
            dist_backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=dist_backend, init_method="env://")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

    return device, is_distributed, rank, local_rank, world_size


VIEW_KEYS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]
def build_hil_serl_obs(results: dict, image_size=(192, 256)) -> dict:
    """
    Resize the three observation image views and stack them into an obs tensor
    with shape [V, C, H, W].

    Args:
        results: dict containing the three image views and other fields.
        image_size: target image size as (H, W), default (192, 256).

    Returns:
        A new dict with all original fields preserved, plus:
            results["obs"]: np.ndarray of shape [3, 3, H, W]
    """
    target_h, target_w = image_size

    resized_views = []
    for key in VIEW_KEYS:
        if key not in results:
            raise KeyError(f"Missing key in results: {key}")

        img = results[key]
        if not isinstance(img, np.ndarray):
            raise TypeError(f"{key} must be a numpy array, but got {type(img)}")

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"{key} must have shape [H, W, 3], but got {img.shape}"
            )

        # cv2.resize uses (width, height)
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # HWC -> CHW
        img_chw = np.transpose(img_resized, (2, 0, 1)).astype(np.float32)

        resized_views.append(img_chw)

    # [V, C, H, W]
    obs = np.stack(resized_views, axis=0) * 2 - 1

    new_results = {}
    new_results["obs"] = obs
    new_results["prompt"] = results["prompt"]
    new_results["arm_joint_states"] = results["observation.state"][:14]
    new_results["gripper_states"] = results["observation.state"][14:16]


    # TODO: hard code here
    new_results["shift"] = 1.0
    new_results["sample_solver"] = "euler"
    return new_results


def get_mock_output(obs):
    arm_joint_state = obs['arm_joint_states']
    gripper_state = obs['gripper_states'] / 120
    return np.concatenate((arm_joint_state,gripper_state),axis=-1)[None,:].repeat(33,axis=0)

class WanPolicyServer(WanPolicy):
    
    def __init__(self, host, port, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if obs["prompt"].find("<reset>")>=0:
                    self.reset()
                obs["prompt"] = obs["prompt"].replace("<reset>", "")

                infer_time = time.monotonic()

                action = self.play(**obs)

                action = dict(actions=action,)
    

                infer_time = time.monotonic() - infer_time
                print(f"Infer time cost:{infer_time}")

                await websocket.send(packer.pack(action))
                
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break

            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def get_args():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )

    parser.add_argument('-c', '--config', type=str, required=True, help='Path for the model config')

    parser.add_argument('--host', type=str, default="127.0.0.1")
    
    parser.add_argument('-p', '--port', type=int, default=8001)

    parser.add_argument(
        '--compile',
        action='store_true',
        help='Enable torch.compile for the diffusion model.',
    )
    parser.add_argument(
        '--compile-mode',
        type=str,
        default='reduce-overhead',
        choices=['default', 'reduce-overhead', 'max-autotune'],
        help='Compilation mode passed to torch.compile.',
    )
    parser.add_argument(
        '--compile-dynamic',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false'],
        help='dynamic argument passed to torch.compile. Use false to force static-shape compilation.',
    )
    parser.add_argument(
        '--compile-target',
        type=str,
        default='full',
        choices=['full', 'action_blocks', 'video_blocks', 'both_blocks'],
        help='Compile the whole diffusion model or only selected action/video blocks.',
    )
    parser.add_argument(
        '--disable-self-attn-fused-qkv',
        action='store_true',
        help='Disable fused QKV projection for self-attention blocks.',
    )
    parser.add_argument(
        '--disable-context-null-cache',
        action='store_true',
        help='Disable context_null caching and action-only negative-prompt skip optimization.',
    )
    parser.add_argument(
        '--enable-action-cross-attn-kv-cache',
        action='store_true',
        help='',
    )
    parser.add_argument(
        '--attention-impl',
        type=str,
        default='sdpa',
        choices=['auto', 'flash_attn', 'sdpa'],
        help='Attention implementation to use. "auto" prefers flash_attn when available.',
    )
    parser.add_argument(
        '--sdpa-backend',
        type=str,
        default='auto',
        choices=['auto', 'flash', 'efficient', 'math', 'cudnn'],
        help='SDPA backend used when attention_impl=sdpa or flash_attn is unavailable.',
    )
    parser.add_argument(
        '--flash-attn-version',
        type=str,
        default='auto',
        choices=['auto', '2', '3'],
        help='Preferred flash_attn version when attention_impl is auto or flash_attn.',
    )
    parser.add_argument(
        '--disable-action-rope-cache',
        action='store_true',
        help='Disable action-branch 1D RoPE precompute path.',
    )
    parser.add_argument(
        '--use-hil-serl',
        action='store_true',
        help='Using hil serl to run inference.',
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    policy_metadata = dict(test_meta="Wan Policy Meta Data")
    
    device, is_distributed, rank, local_rank, world_size = init_distributed_and_get_device()
    
    actor = WanPolicyServer(
        args.host, args.port, policy_metadata,
        config_file=args.config,
        device=device,
        rank=rank,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        compile_dynamic=args.compile_dynamic,
        compile_target=args.compile_target,
        enable_self_attn_fused_qkv=not args.disable_self_attn_fused_qkv,
        enable_context_null_cache=not args.disable_context_null_cache,
        attention_impl=args.attention_impl,
        sdpa_backend=args.sdpa_backend,
        flash_attn_version=args.flash_attn_version,
        enable_action_rope_cache=not args.disable_action_rope_cache,
        enable_action_cross_attn_kv_cache=args.enable_action_cross_attn_kv_cache
    )
    
    ### init server
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("Creating server (host: %s, ip: %s)", hostname, local_ip)

    print("Waiting...")

    ### start server and waiting for response
    actor.serve_forever()

