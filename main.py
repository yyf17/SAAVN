#!/usr/bin/env python3
import os
import git
import sys
repo = git.Repo(".", search_parent_directories=True)
# print(repo.working_tree_dir)
if f"{repo.working_tree_dir}" not in sys.path:
    sys.path.append(f"{repo.working_tree_dir}")
    print("add")

import argparse
import logging

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import tensorflow as tf
import torch


from ss_baselines.common.baseline_registry import baseline_registry
from habitat.core.registry import registry

from contextlib import redirect_stdout

from simulator import get_simulator_class
from envs import get_env_class
from trainer import get_trainer_class


from copy import deepcopy
import sys
def get_default_config_by_arg(arg_name="--default"):

    cmd_params = deepcopy(sys.argv)

    print("sys.argv:",sys.argv)

    default_config_str = None
    for _i, _v in enumerate(cmd_params):
        if _v.split(" ")[0] == arg_name:
            default_config_str = _v.split(" ")[1]
            del cmd_params[_i]
            break

    assert default_config_str is not None ,"default config  is not specify"
    return default_config_str



DEFAULT_CONFIG_DIR = "configs/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default",
        choices="ma",
        # required=True,
        default="ma",
        help="default config of the experiment to get_config function",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        # required=True,
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        # required=True,
        default='av_nav/config/pointgoal_rgb.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--tag-config",
        type=str,
        # required=True,
        default='',
        help="path to config yaml containing info about experiment with tag",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--eval-best",
        default=False,
        help="Modify config options from command line"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    ckpt_msg = ""
    if args.eval_best:
        best_ckpt_idx = find_best_ckpt_idx(os.path.join(args.model_dir, 'tb'))
        best_ckpt_path = os.path.join(args.model_dir, 'data', f'ckpt.{best_ckpt_idx}.pth')
        ckpt_msg = f"best spl ckpt:{best_ckpt_path}"
        args.opts += ['EVAL_CKPT_PATH_DIR', best_ckpt_path]

    root = getattr(args,'default')

    delattr(args,'default')
    delattr(args,'eval_best')


    config = get_config(get_config_dict[root]["_C"],get_config_dict[root]["_TC"],args.tag_config,args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.TYPE = config.SIM_NAME
    config.freeze()


    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)

    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info(ckpt_msg)
    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, config.USE_LAST_CKPT)


if __name__ == "__main__":
    main()
