import dataclasses
import logging

import numpy as np
import tyro

from web_infer_utils.openpi_client import websocket_client_policy as _websocket_client_policy


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int | None = 8001


def main(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    payload = {
        "obs": np.random.rand(2, 3, 192, 256).astype(np.float32)*2-1,    # range -1 to 1, {v,c,h,w}
        "prompt": "task or step caption", 
        # single arm: state index + grippe index; 
        # dual arm: left state index + left gripper index + right state index + right gripper index 
        "state": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32), # raw value
        "num_inference_steps": 5,
        "execution_step": 30,
        "sample_solver": "euler",
        "shift": 1.0,
    }
    print(payload)
    
    # single arm: action index + grippe index ; 
    # dual arm: left action index + left gripper index + right action index + right action index 
    action = policy.infer(obs=payload)["actions"] # raw value
    print(action.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
