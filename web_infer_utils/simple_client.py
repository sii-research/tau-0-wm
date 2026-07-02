import dataclasses

from web_infer_utils.openpi_client import websocket_client_policy as _websocket_client_policy
import numpy as np
import logging
import tyro
import time
import os
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import random

from einops import rearrange

    
@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "localhost"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8001
    
    
def main(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")
    
    print("A mock payload is provide for example")
    payload = {
        "obs": np.random.rand(3, 3, 192, 256),    # range -1 to 1, {v,c,h,w}
        "prompt": "task or step caption",
        "state": np.random.rand(14),
        "gripper_states": np.random.rand(2)*120,
        "num_inference_steps": 5,
        "execution_step": 30,
        "sample_solver": "euler",
        "shift": 1.0
    }
    print(payload)
    
    action = policy.infer(obs=payload)['actions']


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
    