
import asyncio
import http
import logging
import time
import traceback

from web_infer_utils.openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from web_infer_utils.simulator.TauSimulator import TauSimulator


import numpy as np
import cv2
import json
import argparse
import socket

import os
import torch
import torch.distributed as dist

from utils import import_custom_class


logger = logging.getLogger(__name__)


def init_distributed_and_get_device(backend: str = "nccl"):
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

class TauSimulatorServer(TauSimulator):
    
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
                    self.reset(**obs)

                obs["prompt"] = obs["prompt"].replace("<reset>", "")

                infer_time = time.monotonic()

                results = self.play(**obs)

                results = dict(results=results,)

                infer_time = time.monotonic() - infer_time

                await websocket.send(packer.pack(results))
                
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
    
    parser.add_argument('-p', '--port', type=int, default=8002)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    policy_metadata = dict(test_meta="Tau Simulator Policy Meta Data")
    
    device, is_distributed, rank, local_rank, world_size = init_distributed_and_get_device()

    
    # ### init server
    # hostname = socket.gethostname()
    # local_ip = socket.gethostbyname(hostname)
    # print("Creating server (host: %s, ip: %s)", hostname, local_ip)


    actor = TauSimulatorServer(
        args.host, args.port, policy_metadata,
        config_file=args.config, device=device, rank=rank
    )


    print("Waiting...")

    ### start server and waiting for response
    actor.serve_forever()