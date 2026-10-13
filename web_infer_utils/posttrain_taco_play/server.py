import argparse
import http
import logging
import os
import socket
import time
import traceback
import asyncio

import numpy as np
import torch
import torch.distributed as dist
import websockets.asyncio.server as _server
import websockets.frames

from web_infer_utils.openpi_client import msgpack_numpy
from web_infer_utils.posttrain_taco_play.TauPolicy import TauPolicy

logger = logging.getLogger(__name__)


def init_distributed_and_get_device(backend: str = "nccl", device_type: str = None):
    """
    Args:
        backend: distributed backend to use for CUDA (default 'nccl').
        device_type: override device type ('xpu', 'cuda', 'cpu').
                     Auto-detects the best available when None.
    """
    if device_type is None:
        if torch.xpu.is_available():
            device_type = "xpu"
        elif torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

    is_distributed = False
    rank = 0
    local_rank = 0
    world_size = 1

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        is_distributed = int(os.environ["WORLD_SIZE"]) > 1
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device_type == "xpu":
            torch.xpu.set_device(local_rank)
            device = torch.device("xpu", local_rank)
            dist_backend = "ccl"
        elif device_type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist_backend = backend
        else:
            device = torch.device("cpu")
            dist_backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=dist_backend, init_method="env://")
    else:
        if device_type == "xpu":
            device = torch.device("xpu", 0)
            torch.xpu.set_device(device)
        elif device_type == "cuda":
            device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

    return device, is_distributed, rank, local_rank, world_size


class TauPolicyServer(TauPolicy):
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

        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if obs["prompt"].find("<reset>") >= 0:
                    self.reset()
                obs["prompt"] = obs["prompt"].replace("<reset>", "")

                infer_time = time.monotonic()
                action = self.play(**obs)
                action = dict(actions=action)
                infer_time = time.monotonic() - infer_time
                print(f"Infer time cost:{infer_time}")
                await websocket.send(packer.pack(action))

                _ = time.monotonic() - start_time
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
    return None


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for the posttrain inference server.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path for the model config")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=8001)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile-dynamic", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--compile-target", type=str, default="full", choices=["full", "action_blocks", "video_blocks", "both_blocks"])
    parser.add_argument("--disable-self-attn-fused-qkv", action="store_true")
    parser.add_argument("--disable-context-null-cache", action="store_true")
    parser.add_argument("--enable-action-cross-attn-kv-cache", action="store_true")
    parser.add_argument("--attention-impl", type=str, default="sdpa", choices=["auto", "flash_attn", "sdpa"])
    parser.add_argument("--sdpa-backend", type=str, default="auto", choices=["auto", "flash", "efficient", "math", "cudnn"])
    parser.add_argument("--flash-attn-version", type=str, default="auto", choices=["auto", "2", "3"])
    parser.add_argument("--disable-action-rope-cache", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device type: 'xpu', 'cuda', or 'cpu'. Auto-detects if not specified.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    policy_metadata = dict(test_meta="Tau Posttrain Policy Meta Data")
    device, is_distributed, rank, local_rank, world_size = init_distributed_and_get_device(device_type=args.device)
    actor = TauPolicyServer(
        args.host,
        args.port,
        policy_metadata,
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
        enable_action_cross_attn_kv_cache=args.enable_action_cross_attn_kv_cache,
    )
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("Creating server (host: %s, ip: %s)", hostname, local_ip)
    print("Waiting...")
    actor.serve_forever()
