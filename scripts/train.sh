# Tune NCCL / threading for your cluster as needed.
# Keep these unset by default so the script is portable across machines.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export TOKENIZERS_PARALLELISM=false

if [ -n "${PUBLIC_TAU_CONDA_ENV:-}" ]; then
    source "${PUBLIC_TAU_CONDA_ENV}/bin/activate"
fi


script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

runner_class_path=${3:-"runner/posttrain.py"}
echo $runner_class_path

runner_class=${4:-"Trainer"}
echo $runner_class


PET_NNODES="${PET_NNODES:-1}"

if [ $PET_NNODES -eq 1 ]; then
NGPU=`nvidia-smi --list-gpus | wc -l`
echo "Training on 1 Nodes, $NGPU GPUs"
torchrun --nnodes=1 \
    --nproc_per_node=$NGPU \
    --node_rank=0 \
    $script_path \
    --config_file $config_path \
    --runner_class_path $runner_class_path \
    --runner_class $runner_class
else
echo "Training on $PET_NNODES Nodes, 8 GPU per Node"
NGPU=`nvidia-smi --list-gpus | wc -l`
torchrun --nnodes=$PET_NNODES \
    --nproc_per_node=$NGPU \
    --node_rank=$PET_NODE_RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $script_path \
    --config_file $config_path \
    --runner_class_path $runner_class_path \
    --runner_class $runner_class
fi
