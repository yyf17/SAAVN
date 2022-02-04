#!/usr/bin/env python3

import os
import time
from typing import ClassVar, Dict, List
import glob

import torch

from habitat import Config, logger
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import poll_checkpoint_folder


class BaseTrainer:
    r"""Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """

    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device
    config: Config
    video_option: List[str]
    _flush_secs: int

    def __init__(self, config: Config):
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def eval(self, eval_interval=1, prev_ckpt_ind=-1, use_last_ckpt=False) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            # eval last checkpoint in the folder
            if use_last_ckpt:
                models_paths = list(
                    filter(os.path.isfile, glob.glob(self.config.EVAL_CKPT_PATH_DIR + "/*"))
                )
                models_paths.sort(key=os.path.getmtime)
                self.config.defrost()
                self.config.EVAL_CKPT_PATH_DIR = models_paths[-1]
                self.config.freeze()

            # add for continue eval ckpt from break step
            if self.config.EVAL_prev_ckpt_index_enable:
                prev_ckpt_ind = int(self.config.EVAL_prev_ckpt_index)

            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                result = self._eval_checkpoint(self.config.EVAL_CKPT_PATH_DIR, writer)
                return result
            else:
                # evaluate multiple checkpoints in order
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind, eval_interval
                        )
                        # time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += eval_interval
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind
                    )

                    if prev_ckpt_ind == 799:
                            break
                logger.info(f"=======congratulations,you have finished check all the ckpt,the last ckpt is: {current_ckpt}=======")

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)

        config.TASK_CONFIG.SIMULATOR.AGENT_0.defrost()
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )

    @staticmethod
    def _pause_envs_agent_attack(
        attacker_actions_desc,
        envs_to_pause,
        envs,
        test_recurrent_hidden_states_agent,
        test_recurrent_hidden_states_attack,
        not_done_masks_agent,
        not_done_masks_attack,
        current_episode_reward_agent,
        current_episode_reward_attack,
        prev_actions_agent,
        prev_actions_attack,
        batch,
        rgb_frames,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states_agent = test_recurrent_hidden_states_agent[
                :, state_index
            ]
            test_recurrent_hidden_states_attack = test_recurrent_hidden_states_attack[
                :, state_index
            ]

            not_done_masks_agent = not_done_masks_agent[state_index]
            not_done_masks_attack = not_done_masks_attack[state_index]

            current_episode_reward_agent = current_episode_reward_agent[state_index]
            current_episode_reward_attack = current_episode_reward_attack[state_index]

            prev_actions_agent = prev_actions_agent[state_index]
            prev_actions_attack_new = {}
            if attacker_actions_desc.position.is_action:
                prev_actions_attack_new["position"] = prev_actions_attack["position"][state_index]
            if attacker_actions_desc.alpha.is_action:
                prev_actions_attack_new["alpha"] = prev_actions_attack["alpha"][state_index]
            if attacker_actions_desc.category.is_action:
                prev_actions_attack_new["category"] = prev_actions_attack["category"][state_index]

            prev_actions_attack = None
            prev_actions_attack = prev_actions_attack_new

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states_agent,
            test_recurrent_hidden_states_attack,
            not_done_masks_agent,
            not_done_masks_attack,
            current_episode_reward_agent,
            current_episode_reward_attack,
            prev_actions_agent,
            prev_actions_attack,
            batch,
            rgb_frames,
        )
