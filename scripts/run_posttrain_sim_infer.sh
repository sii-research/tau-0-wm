#!/usr/bin/env bash

IP_ADDRESS_OF_SERVER=${1-"127.0.0.1"}
PORT=${2-"8002"}

python -m web_infer_utils.simulator.server_sim \
    --config configs/deployment/tau_simulator_posttrain_taco_play_abs.yaml \
    --host $IP_ADDRESS_OF_SERVER \
    --port $PORT
