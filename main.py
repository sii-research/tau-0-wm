import os
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



import argparse
from utils import import_custom_class
from utils.config_utils import expand_env_vars



def main():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--runner_class_path', type=str, default="runner/posttrain.py")
    parser.add_argument('--runner_class', type=str, default="Trainer")
    parser.add_argument('--mode', type=str, default="train")
    
    args = parser.parse_args()
    args.config_file = expand_env_vars(args.config_file)

    Runner = import_custom_class(
        args.runner_class, args.runner_class_path, 
    )

    if args.mode == "train":
                
        ### Trainer
        runner = Runner(args.config_file)
        runner.prepare_dataset()
        
        if not hasattr(sys.stdout, 'isatty'):
            sys.stdout.isatty = lambda: False
        if not hasattr(sys.stderr, 'isatty'):
            sys.stderr.isatty = lambda: False
        runner.prepare_models()
        
        runner.prepare_trainable_parameters()
        runner.prepare_optimizer()
        runner.prepare_for_training()
        runner.prepare_trackers()

        # logical_cpus = os.cpu_count()
        # os.environ["OMP_NUM_THREADS"] = str(logical_cpus//8)

        runner.train()

    else:
        raise NotImplementedError



if __name__ == "__main__":
  
    main()
    
