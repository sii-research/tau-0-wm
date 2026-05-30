
IP_ADDRESS_OF_SERVER=$1
PORT=$2

python -m web_infer_utils.server \
    --config configs/deployment/wan_pretrain_rela_eef6d.yaml \
    --host $IP_ADDRESS_OF_SERVER \
    --port $PORT
