# Data Format

This repo expects a dataset class that returns a dict with at least:

- `video`: tensor shaped `C, V, T, H, W` or a compatible video tensor
- `actions`: tensor shaped `T, C`
- `state`: tensor shaped `1, C`
- `caption`: task text

For `taco_play`, the `action` / `state` layout used by the example config is:

- per-arm joint block
- the last dimension of each arm block is `gripper`

The example implementation lives in [`data/example_dataset.py`](./example_dataset.py). Copy it when you want to build your own dataset class.

## Example data config

The sample YAML is [`configs/data/example_task/taco_abs_joint.yaml`](../configs/data/example_task/taco_abs_joint.yaml).
Its `data:` block is passed into `CustomLeRobotDataset.__init__`, so those values override the defaults in [`data/example_dataset.py`](./example_dataset.py).
It shows how to wire:

- `data_class`: the dataset class name imported from `data_class_path`.
- `data_class_path`: the Python module that defines the dataset class.
- `data.data_roots`: one or more dataset root paths passed into the dataset constructor.
- `data.statistic_files`: one or more statistics JSON files used for normalization.
- `data.action_space` / `data.action_type`: the action layout expected by the dataset and the model.

## Statistics

Use this command to compute action/state statistics for a dataset config:

```bash
python -m scripts.compute_stat \
  --config_file [data_config] \
  --action_space [action_space] \
  --action_type [action_type] \
  --output [target_stat_file_path] \
  --num [num_to_compute_stat] \
  --batch_size [bsz] \
  --num_workers [n_worker]
```

The resulting JSON should match the structure expected by the training and inference configs:

- `action.mean`
- `action.std`
- `state.mean`
- `state.std`

## Custom dataset skeleton

Create a new module next to `example_dataset.py` and follow the same constructor contract used by `CustomLeRobotDataset`.

Minimum pattern:

```python
class MyDataset(Dataset):
    def __init__(self, data_roots, statistic_files, action_space, action_type, **kwargs):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        return {
            "video": video,
            "actions": actions,
            "state": state,
            "caption": caption,
        }
```

## Notes

- For `taco_play`, see: https://huggingface.co/datasets/lerobot/taco_play
