#!/usr/bin/env bash

IP_ADDRESS_OF_SERVER=${1-"127.0.0.1"}
PORT=${2-8001}

python -m web_infer_utils.posttrain_taco_play_withTTC.server \
    --config configs/deployment/tau_posttrain_taco_play_abs_ttc.yaml \
    --host $IP_ADDRESS_OF_SERVER \
    --port $PORT
