
# $\tau_0$-World Model

<div id="top" align="center">

![Overview](figures/VAM-teaser-img.jpg)

</a> &nbsp; <a href='https://finch.agibot.com/research/tau0-wm'><img src='https://img.shields.io/badge/Project_Website-tau0_WM-blue' height='25'></a> &nbsp; <a href='https://finch-static.agibot.com/VAM/blog/tau_0_wm.pdf'><img src='https://img.shields.io/badge/Paper-tau_0_WM-red' height='25'></a> &nbsp; <a href='https://huggingface.co/sii-research/tau-0-wm'><img src='https://img.shields.io/badge/Weight-huggingface-orange' height='25'></a> &nbsp;

</div>

This repo is the official implementation of **$\tau_0$-World Model: A Unified Video-Action World Model for Robotic Manipulation**.


## News
- [2026.06.26] 🚀 We release the post-training training and inference code for $\tau_0$-World Model.
- [2026.05.31] 🚀 We release $\tau_0$-World Model [Paper](https://finch-static.agibot.com/VAM/blog/tau_0_wm.pdf), [Project Website](https://finch.agibot.com/research/tau0-wm), [Huggingface](https://huggingface.co/sii-research/tau-0-wm).


## Pretrained Model

* The pretrained weights of VAM can be found on [Hugging Face](https://huggingface.co/sii-research/tau-0-wm).

* The pretrained weights of Simulator will be released soon.

* The codes of Test-Time Computation will be further released with The pretrained weights of Simulator.





## Real-World Deployment

### Setup

```
pip install -r requirements.txt
```

### Preparation

1. Download the pretrained weight of $\tau_0$-World Model.

2. Download the weight of [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

3. Replace `diffusion_model.model_path` in `configs/deployment/tau_pretrain_rela_eef6d.yaml` with your local path to $\tau_0$-WM's weight.

4. Replace `vae_path` in the config with your local path to VAE's weight.

5. Replace `text_encoder.checkpoint_path` and `text_encoder.tokenizer_path` in the config with your local path to text encoder and tokenizer.


### Action Space

*state* sent to the server should be the **absolute** poses of two end-of-effectors, including 14 channels (xyz and quaternion with order *xyzw*). The coordinate origin of each eef pose is its corresponding **Arm Base link**.

*gripper state* should include 2 channels, ranging from 0 to 120 (0 for opening and 120 for close).

*action* obtained from the server will be the **absolute** poses of end-of-effectors with shape {T, 16}. 

The order of output actions:
- left end-effector (xyz + quaternion with order *xyzw*)
- left gripper openness (ranging from 0 to 1, 0 for opening and 1 for close)
- right end-effector
- right gripper openness.


In the pretraining stage, $\tau_0$-WM is optimized to predict the relative pose of end-effectors, including 20 channels (xyz and 6d-rotation for each arm). **The conversion between quaternion and 6d-rotation will be executed automatically.**



## Post-Training

The repo also ships the post-training pipeline used to adapt $\tau_0$-WM to a downstream manipulation task. The entry point is `main.py`, driven by a YAML config that names the trainer, model classes, and dataset.

### Data Format

Each downstream task needs three pieces:

1. **A LeRobot-format dataset.** A LeRobot dataset directory is expected (>= 0.4.0 is recommended; legacy 0.3.x is also supported). For one example task, the data can be downloaded from [taco_play](https://huggingface.co/datasets/lerobot/taco_play).

2. **A data YAML under `configs/data/<task>/`.** This YAML defines the dataset class, dataset roots, statistics file, and action/state layout. Use `configs/data/example_task/taco_abs_joint.yaml` as a sample. See [`data/README.md`](data/README.md) for the dataset contract and how to implement your own dataset.

3. **A training YAML under `configs/tau_model/`.** This YAML references the data YAML and model/training settings. Use `configs/tau_model/posttrain_taco_play_abs.yaml` as a sample.

### Launching training

```
bash scripts/train.sh main.py \
    configs/tau_model/posttrain_taco_play_abs.yaml \
    runner/posttrain.py
```

Before launching, edit `configs/tau_model/posttrain_taco_play_abs.yaml` and fill in the local paths for:

- `output_dir`, `diffusion_model.model_path`, `text_encoder.checkpoint_path`, `text_encoder.tokenizer_path` and `vae_path`
- the dataset root referenced by `configs/data/example_task/taco_abs_joint.yaml`

### Running
We provide two inference examples:

```
# Policy Server
bash run_infer_server.sh $HOST $PORT

# A simple client that send random observations
python web_infer_utils/simple_client.py
```

For one downstream post-training example based on `taco_play`, use:

```
# Posttrain Policy Server
bash scripts/run_posttrain_infer.sh $HOST $PORT

# Posttrain client
bash scripts/run_posttrain_client.sh
```

The pretrain example keeps using `configs/deployment/tau_pretrain_rela_eef6d.yaml`.
The posttrain example uses `configs/deployment/tau_posttrain_taco_play_abs.yaml`.

For both examples, fill in these parameters in the YAML before launching:

- `diffusion_model.model_path`,`vae_path`,`text_encoder.checkpoint_path` and `text_encoder.tokenizer_path`
- `statistics_file`: mean/std JSON matching the action/state layout
- `action_space` and `action_type`: must match the dataset/task contract


## Acknowledgment
- The video model of $\tau_0$-WM is built on [Wan-2.2](https://github.com/Wan-Video/Wan2.2).
- Some codes in this repo are modified from [GE-Act](https://github.com/AgibotTech/Genie-Envisioner.git).
- The dataset examples are adapted from [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/).
- The web-socket based policy server is built on [openpi](https://github.com/Physical-Intelligence/openpi).


### License
Data and codes within this repo are under [Apache License 2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE).
