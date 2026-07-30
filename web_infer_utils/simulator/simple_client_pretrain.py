import dataclasses
import logging

import numpy as np
import tyro

from web_infer_utils.openpi_client import websocket_client_policy as _websocket_client_policy


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int | None = 8002


def main(args: Args) -> None:
    simulator = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Server metadata: {simulator.get_server_metadata()}")

    payload_posttrain_taco_play = {
        "obs": np.random.rand(2, 3, 192, 256).astype(np.float32)*2-1,    # range -1 to 1, {v,c,h,w}
        "prompt": "<reset>task or step caption",  # add <reset> to reset memory if necessary
        
        ### In the pre-training stage of our video simulator, its action space of dual arms is the same as that of the video action model, i.e., the relative eef with xyz and 6d-rotation. The coordinate origin of each eef pose is its corresponding Arm Base link.
        ### left/right arm: [xyz (3) + rot6d (9) + gripper (1)]
        ### gripper:  0 for opening and 1 for close
        "actions": np.zeros([33, 20]), # {t,c}

        "num_inference_steps": 25,
        "execution_step": 30,
        "sample_solver": "unipc",
        "shift": 5.0,
        "guide_scale": 1,
        "save_sim_video": True,
        "n_mem": 3,
        "sim_path": "path/to/save/videos",
    }

    print(payload_posttrain_taco_play)
    videos = simulator.infer(obs=payload_posttrain_taco_play)["results"]
    print(videos.shape)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
