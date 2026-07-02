#!/usr/bin/env bash

IP_ADDRESS_OF_SERVER=$1
PORT=$2

python -m web_infer_utils.posttrain_taco_play.server \
    --config configs/deployment/tau_posttrain_taco_play_abs.yaml \
    --host $IP_ADDRESS_OF_SERVER \
    --port $PORT
