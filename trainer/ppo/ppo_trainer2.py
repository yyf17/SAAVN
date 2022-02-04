#!/usr/bin/env python3


import os

import os
import time
import datetime
import logging
from collections import deque
from typing import Dict, List
import json
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from habitat import Config, logger
from ss_baselines.common.base_trainer import BaseRLTrainer

from ss_baselines.common.env_utils import construct_envs


from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    exponential_decay,
    plot_top_down_map,
    resize_observation
)

from soundspaces.visualizations.utils import observations_to_image



from habitat.config import Config as CN

from storage.rollout_storage import RolloutStorage, RolloutStorageHybrid, RolloutStorageMA
from trainer.ppo.policy import PointNavBaselinePolicy, PointNavBaselinePolicyHybrid
from trainer.ppo.ppo import PPOMA


from envs import get_env_class
import time

from ss_baselines.common.baseline_registry import baseline_registry
@baseline_registry.register_trainer(name="PPOTrainer")
class PPOTrainerMa(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)

        self.config = config

        self.actor_critic_attack = None

        self.actor_critic_agent = None

        self.ppo = None

        self.envs = None

    

        assert self.config.RL.PPO_attack.attack_enable, "make sure the PPO_attack.attack_enable is True"


        self._attacker_actions_desc = CN()
        self._attacker_actions_desc.position = CN()
        self._attacker_actions_desc.position.dim = self.config.RL.PPO_attack.position.dim
        self._attacker_actions_desc.position.is_action = self.config.RL.PPO_attack.position.is_action
        self._attacker_actions_desc.position.random_policy = self.config.RL.PPO_attack.position.random_policy

        self._attacker_actions_desc.alpha = CN()
        self._attacker_actions_desc.alpha.dim = self.config.RL.PPO_attack.alpha.dim
        self._attacker_actions_desc.alpha.is_action = self.config.RL.PPO_attack.alpha.is_action
        self._attacker_actions_desc.alpha.random_policy = self.config.RL.PPO_attack.alpha.random_policy
        self._attacker_actions_desc.alpha.enable_curriculum_learning= self.config.RL.PPO_attack.alpha.enable_curriculum_learning    

        self._attacker_actions_desc.category = CN()
        self._attacker_actions_desc.category.dim = self.config.RL.PPO_attack.category.dim
        self._attacker_actions_desc.category.is_action = self.config.RL.PPO_attack.category.is_action
        self._attacker_actions_desc.category.random_policy = self.config.RL.PPO_attack.category.random_policy

    def _setup_actor_critic_two_agents(self, ppo_cfg: Config, observation_space=None):

        logger.add_filehandler(self.config.LOG_FILE)

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]
        
        #-----------------------------------


        
        self.actor_critic_attack = PointNavBaselinePolicyHybrid(
            observation_space=observation_space,
            attacker_actions_desc = self._attacker_actions_desc,
            hidden_size=ppo_cfg.PPO_attack.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            extra_rgb=self.config.EXTRA_RGB
        )
        self.actor_critic_attack.to(self.device)


        # policy for agent, training    or fixed
        self.actor_critic_agent = PointNavBaselinePolicy(
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.PPO_agent.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            extra_rgb=self.config.EXTRA_RGB
        )
        self.actor_critic_agent.to(self.device)

        value_loss_coef_dict = {
            "agent":ppo_cfg.PPO_agent.value_loss_coef,
            "attack":ppo_cfg.PPO_attack.value_loss_coef,
        }

        entropy_coef_dict  = {
            "agent":ppo_cfg.PPO_agent.entropy_coef,
            "attack":ppo_cfg.PPO_attack.entropy_coef,
        }

        self.ppo = PPOMA(             
            self.actor_critic_agent,     
            self.actor_critic_attack,   
            self._attacker_actions_desc, 
            ppo_cfg.PPO_agent.clip_param,
            ppo_cfg.PPO_agent.ppo_epoch,
            ppo_cfg.PPO_agent.num_mini_batch,
            value_loss_coef_dict,
            entropy_coef_dict,
            lr=ppo_cfg.PPO_agent.lr,
            eps=ppo_cfg.PPO_agent.eps,
            max_grad_norm=ppo_cfg.PPO_agent.max_grad_norm,
        )
   
        self.ppo.to(self.device)



    def save_checkpoint(self, file_name: str):
        checkpoint = {
            "state_dict": self.ppo.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs):
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(
        self, rollouts, current_episode_reward_agent, current_episode_reward_attack, current_episode_step, episode_rewards_agent,episode_rewards_attack,
            episode_spls, episode_counts, episode_steps, episode_distances,tau_agent,tau_attack_dict
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        with torch.no_grad():

            step_observation_attack = {
                k: v[rollouts.attack.step] for k, v in rollouts.attack.observations.items()
            }
 
            prev_actions_step = {}
            if self._attacker_actions_desc.position.is_action:
                prev_actions_step["position"] = rollouts.attack.prev_actions["position"][rollouts.attack.step]
            
            if self._attacker_actions_desc.alpha.is_action:
                prev_actions_step["alpha"] = rollouts.attack.prev_actions["alpha"][rollouts.attack.step]
            if self._attacker_actions_desc.category.is_action:
                prev_actions_step["category"] =  rollouts.attack.prev_actions["category"][rollouts.attack.step]

            (
                values_attack,
                actions_attack,                                                
                actions_log_probs_attack,                                       
                recurrent_hidden_states_attack,
            ) = self.actor_critic_attack.act(                                  
                step_observation_attack,
                rollouts.attack.recurrent_hidden_states[rollouts.attack.step], 
                prev_actions_step,
                rollouts.attack.masks[rollouts.attack.step],
                tau_dict=tau_attack_dict,
            )
            
            # for agent
            step_observation_agent = {
                k: v[rollouts.agent.step] for k, v in rollouts.agent.observations.items()
            }

            (
                values_agent,
                actions_agent,                                                    
                actions_log_probs_agent,
                recurrent_hidden_states_agent,
            ) = self.actor_critic_agent.act(                                      
                step_observation_agent,
                rollouts.agent.recurrent_hidden_states[rollouts.agent.step],
                rollouts.agent.prev_actions[rollouts.agent.step],                 
                rollouts.agent.masks[rollouts.agent.step],
                tau=tau_agent,
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()


        action_list = []
        for i in range(self.config.NUM_PROCESSES):
            a_agent = actions_agent[i].item()
            a_attck_dict = {}
            if "position" in actions_attack:
                a_attack = actions_attack["position"][i].item()
                a_attck_dict["position"] = a_attack
            if "alpha" in actions_attack:
                a_attck_dict["alpha"] = actions_attack["alpha"][i].item()

                #--------------------------------------------------------------------
            if "category" in actions_attack:
                a_attck_dict["category"] =  actions_attack["category"][i].item()
            a_tuple = {
                    "action_attack": a_attck_dict,     
                    "action" : a_agent,      
            }
            action_list.append(a_tuple)


        outputs = self.envs.step(action_list) 

        observations, rewards_two, dones, infos = [list(x) for x in zip(*outputs)]

        rewards_attack, rewards_agent =[list(x) for x in zip(*rewards_two)]


        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations)


        # for attack
        rewards_attack = torch.tensor(rewards_attack, dtype=torch.float , device=self.device)
        rewards_attack = rewards_attack.unsqueeze(1)

        masks_attack = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float , device=self.device)
        # for agent
        rewards_agent = torch.tensor(rewards_agent, dtype=torch.float , device=self.device)
        rewards_agent = rewards_agent.unsqueeze(1)


        masks_agent = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float , device=self.device
        )


        spls = torch.tensor(
            [[info['spl']] for info in infos], device=self.device
        )

        distances = torch.tensor(
            [[info['distance_to_goal']] for info in infos], device=self.device
        )



        current_episode_reward_agent += rewards_agent
        current_episode_reward_attack += rewards_attack
        current_episode_step += 1
        episode_rewards_agent += (1 - masks_agent) * current_episode_reward_agent
        episode_rewards_attack += (1 - masks_attack) * current_episode_reward_attack
        episode_spls += (1 - masks_agent) * spls
        episode_steps += (1 - masks_agent) * current_episode_step
        episode_counts += 1 - masks_agent
        current_episode_reward_agent *= masks_agent
        current_episode_reward_attack *= masks_attack
        current_episode_step *= masks_agent
        episode_distances += (1 - masks_agent) * distances


        rollouts.attack.insert(
            batch,
            recurrent_hidden_states_attack,
            actions_attack,
            actions_log_probs_attack,
            values_attack,
            rewards_attack,
            masks_attack,
        )

        rollouts.agent.insert(
            batch,
            recurrent_hidden_states_agent,
            actions_agent,
            actions_log_probs_agent,
            values_agent,
            rewards_agent,
            masks_agent,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            # for attack
            last_observation_attack = {
                k: v[-1] for k, v in rollouts.attack.observations.items()
            }
            prev_actions_step = dict()
            assert self._attacker_actions_desc.position.is_action or self._attacker_actions_desc.alpha.is_action or self._attacker_actions_desc.category.is_action,"position,alpha,category at least one is True"
            
            if self._attacker_actions_desc.position.is_action:
                prev_actions_step["position"] = rollouts.attack.prev_actions["position"][-1]
            
            if self._attacker_actions_desc.alpha.is_action:
                prev_actions_step["alpha"] = rollouts.attack.prev_actions["alpha"][-1]
            if self._attacker_actions_desc.category.is_action:
                prev_actions_step["category"] =  rollouts.attack.prev_actions["category"][-1]

            next_value_attack = self.actor_critic_attack.get_value(
                last_observation_attack,
                rollouts.attack.recurrent_hidden_states[-1],
                prev_actions_step,
                rollouts.attack.masks[-1],
            ).detach()

            last_observation_agent = {
                k: v[-1] for k, v in rollouts.agent.observations.items()
            }
            next_value_agent = self.actor_critic_agent.get_value(
                last_observation_agent,
                rollouts.agent.recurrent_hidden_states[-1],  #T+1, N, 1
                rollouts.agent.prev_actions[-1],
                rollouts.agent.masks[-1],
            ).detach()


        if ppo_cfg.PPO_agent.is_trainable:  
            rollouts.agent.compute_returns(next_value_agent, ppo_cfg.PPO_agent.use_gae, ppo_cfg.PPO_agent.gamma, ppo_cfg.PPO_agent.tau)
            rollouts.attack.compute_returns(next_value_attack, ppo_cfg.PPO_agent.use_gae, ppo_cfg.PPO_agent.gamma, ppo_cfg.PPO_agent.tau)
            epoch_info = self.ppo.update(rollouts, use_grad=True)
            rollouts.agent.after_update()
            rollouts.attack.after_update()
        else:
            with torch.no_grad():        
                rollouts.agent.compute_returns(next_value_agent, ppo_cfg.PPO_agent.use_gae, ppo_cfg.PPO_agent.gamma, ppo_cfg.PPO_agent.tau)
                rollouts.attack.compute_returns(next_value_attack, ppo_cfg.PPO_agent.use_gae, ppo_cfg.PPO_agent.gamma, ppo_cfg.PPO_agent.tau)
                epoch_info = self.ppo.update(rollouts, use_grad=False)
                rollouts.agent.after_update()
                rollouts.attack.after_update()
        value_loss = {
            "attack":epoch_info.value_loss_epoch.attack,
            "agent":epoch_info.value_loss_epoch.agent,
        }

        action_loss = {
            "attack":epoch_info.action_loss_epoch.attack,
            "agent":epoch_info.action_loss_epoch.agent,
        }

        dist_entropy = {
            "attack":epoch_info.dist_entropy_epoch.attack,
            "agent":epoch_info.dist_entropy_epoch.agent,

        }
        loss = {
            "attack":epoch_info.loss_epoch.attack,
            "agent":epoch_info.loss_epoch.agent,
            "all":epoch_info.loss_epoch.agent,
        }


        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            loss,
        )
    def get_tau(self,tau,step):
        tau = np.random.rand(1)[0]*0.1+0.1+0.8*np.power(step*1.0/(float(self.config.NUM_UPDATES))-1,2)
        return tau

    def train(self):
        global lr_lambda
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_two_agents(ppo_cfg)

        logger.info(
            "ppo number of parameters: {}".format(
                sum(param.numel() for param in self.ppo.parameters())
            )
        )
 
        rollouts_attack = RolloutStorageHybrid(
            ppo_cfg.PPO_attack.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self._attacker_actions_desc,
            ppo_cfg.PPO_attack.hidden_size,
            self.device,
        )
        rollouts_agent = RolloutStorage(
            ppo_cfg.PPO_agent.num_steps+ppo_cfg.PPO_attack.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.PPO_agent.hidden_size,
            self.device,
        )

        rollouts = RolloutStorageMA(attack=rollouts_attack, agent=rollouts_agent)



        observations = self.envs.reset()
        batch = batch_obs(observations)


        for sensor in rollouts.attack.observations:
            rollouts.attack.observations[sensor][0].copy_(batch[sensor])

        for sensor in rollouts.agent.observations:
            rollouts.agent.observations[sensor][0].copy_(batch[sensor])

        batch = None
        observations = None

        episode_rewards_agent = torch.zeros(self.envs.num_envs, 1, dtype=torch.float, device=self.device)                 
        episode_rewards_attack = torch.zeros(self.envs.num_envs, 1, dtype=torch.float, device=self.device)                
        episode_spls = torch.zeros(self.envs.num_envs, 1, dtype=torch.float, device=self.device)                    
        episode_steps = torch.zeros(self.envs.num_envs, 1, device=self.device)                  
        episode_counts = torch.zeros(self.envs.num_envs, 1, device=self.device)                
        episode_distances = torch.zeros(self.envs.num_envs, 1, device=self.device)   
        current_episode_reward_agent = torch.zeros(self.envs.num_envs, 1, dtype=torch.float, device=self.device)      
        current_episode_reward_attack = torch.zeros(self.envs.num_envs, 1, dtype=torch.float, device=self.device)         
        current_episode_step = torch.zeros(self.envs.num_envs, 1, device=self.device)            

        window_episode_reward_agent = deque(maxlen=ppo_cfg.PPO_agent.reward_window_size)     
        window_episode_reward_attack = deque(maxlen=ppo_cfg.PPO_attack.reward_window_size)     
        window_episode_spl = deque(maxlen=ppo_cfg.PPO_agent.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.PPO_agent.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.PPO_agent.reward_window_size)
        window_episode_distances = deque(maxlen=ppo_cfg.PPO_agent.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        if ppo_cfg.PPO.use_linear_lr_decay:
            def lr_lambda(x):  
                return linear_decay(x,self.config.NUM_UPDATES)
        elif ppo_cfg.PPO.use_exponential_lr_decay:
            def lr_lambda(x):  
                return exponential_decay(x,self.config.NUM_UPDATES,ppo_cfg.PPO.exp_decay_lambda)
        else:
            def lr_lambda(x):
                return 1
        lr_scheduler = LambdaLR(
            optimizer=self.ppo.optimizer,
            lr_lambda=lr_lambda,
        )


        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:

            tau_agent = 1.0
            tau_attack_dict = {
                "position":1.0,
                "alpha":1.0,
                "category":1.0,
            }

            for update in range(self.config.NUM_UPDATES):
                tau_agent = self.get_tau(tau_agent,update)
                tau_attack_dict["position"] = self.get_tau(tau_attack_dict["position"],update)
                tau_attack_dict["alpha"] = self.get_tau(tau_attack_dict["alpha"],update)
                tau_attack_dict["category"] = self.get_tau(tau_attack_dict["category"],update)

                if ppo_cfg.PPO.use_linear_lr_decay or ppo_cfg.PPO.use_exponential_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.PPO.use_linear_clip_decay:
                    self.ppo.clip_param = ppo_cfg.PPO.clip_param * linear_decay(update, self.config.NUM_UPDATES)

                for step in range(ppo_cfg.PPO.num_steps):    
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward_agent,
                        current_episode_reward_attack,
                        current_episode_step,
                        episode_rewards_agent,
                        episode_rewards_attack,
                        episode_spls,
                        episode_counts,
                        episode_steps,
                        episode_distances,
                        tau_agent,
                        tau_attack_dict,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps



                delta_pth_time, value_loss, action_loss, dist_entropy,total_loss = self._update(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                window_episode_reward_agent.append(episode_rewards_agent.clone())
                window_episode_reward_attack.append(episode_rewards_attack.clone())
                window_episode_spl.append(episode_spls.clone())
                window_episode_step.append(episode_steps.clone())
                window_episode_counts.append(episode_counts.clone())
                window_episode_distances.append(episode_distances.clone())

                stats = zip(
                    ["count", "reward_agent",  "reward_attack","step", 'spl','distance'],
                    [window_episode_counts, window_episode_reward_agent, window_episode_reward_attack, window_episode_step, window_episode_spl,window_episode_distances],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                if update % 10 == 0:
                    print("spl:",deltas["spl"] / deltas["count"])
                    writer.add_scalar("Environment/SPL", deltas["spl"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Episode_length", deltas["step"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Reward_sum", ( deltas["reward_agent"] + deltas["reward_attack"] ) / deltas["count"], count_steps)
                    writer.add_scalar('Policy/Learning_Rate', lr_scheduler.get_lr()[0], count_steps)
                    writer.add_scalar('Policy/Loss', total_loss["all"], count_steps)

                    writer.add_scalar("Environment/Distance_to_goal", deltas["distance"] / deltas["count"], count_steps)
                    writer.add_scalar("Environment/Reward_agent", deltas["reward_agent"] / deltas["count"], count_steps)
                    writer.add_scalar('PolicyAgent/Value_Loss', value_loss['agent'], count_steps)
                    writer.add_scalar('PolicyAgent/Action_Loss', action_loss['agent'], count_steps)
                    writer.add_scalar('PolicyAgent/Entropy', dist_entropy['agent'], count_steps)
                    writer.add_scalar('PolicyAgent/TotalLoss', total_loss['agent'], count_steps)
                    
                    writer.add_scalar("Environment/Reward_attack", deltas["reward_attack"] / deltas["count"], count_steps)
                    writer.add_scalar('PolicyAttack/Value_Loss', value_loss['attack'], count_steps)
                    writer.add_scalar('PolicyAttack/Action_Loss', action_loss['attack'], count_steps)
                    writer.add_scalar('PolicyAttack/Entropy', dist_entropy['attack'], count_steps)
                    writer.add_scalar('PolicyAttack/TotalLoss', total_loss['attack'], count_steps)


                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards_agent = (
                        window_episode_reward_agent[-1] - window_episode_reward_agent[0]
                    ).sum()

                    window_rewards_attack = (
                        window_episode_reward_attack[-1] - window_episode_reward_attack[0]
                    ).sum()

                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {}  reward_agent: {:3f}".format(
                                len(window_episode_reward_agent),
                                (window_rewards_agent / window_counts).item(),
                            )
                        )
                        logger.info(
                            "Average window size {} reward_attack: {:3f}".format(
                                len(window_episode_reward_attack),
                                (window_rewards_attack / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()


    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ):

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL


        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
            observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 1)
        else:
            observation_space = self.envs.observation_spaces[0]

        self._setup_actor_critic_two_agents(ppo_cfg, observation_space)

        self.ppo.load_state_dict(ckpt_dict["state_dict"])
        self.ppo.to(self.device)
        self.actor_critic_agent = self.ppo.actor_critic_agent
        self.actor_critic_attack = self.ppo.actor_critic_attack


        self.metric_uuids = []
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                metric_cfg.TYPE
            )
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)
        batch = batch_obs(observations, self.device)

        current_episode_reward_agent = torch.zeros(self.envs.num_envs, 1, dtype=torch.float, device=self.device)
        current_episode_reward_attack = torch.zeros(self.envs.num_envs, 1, dtype=torch.float, device=self.device)
        test_recurrent_hidden_states_attack = torch.zeros(
            self.actor_critic_attack.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.PPO_attack.hidden_size,
            device=self.device,
        )
        prev_actions_attack = {}
        assert self._attacker_actions_desc.position.is_action or self._attacker_actions_desc.alpha.is_action or self._attacker_actions_desc.category.is_action ,"position,alpha,category at least one is True"
        if self._attacker_actions_desc.position.is_action:
            prev_actions_attack["position"] = torch.zeros(
                self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
            )
        if self._attacker_actions_desc.alpha.is_action:
            prev_actions_attack["alpha"] = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long)
        if self._attacker_actions_desc.category.is_action:
            prev_actions_attack["category"] = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long)

        not_done_masks_attack = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        test_recurrent_hidden_states_agent = torch.zeros(
            self.actor_critic_agent.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.PPO_agent.hidden_size,
            device=self.device,
        )
        prev_actions_agent = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks_agent = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )


        stats_episodes = dict()  

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  
        audios = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions_attack, _, test_recurrent_hidden_states_attack = self.actor_critic_attack.act(
                    batch,
                    test_recurrent_hidden_states_attack,
                    prev_actions_attack,
                    not_done_masks_attack,
                    deterministic=False
                )
                if self._attacker_actions_desc.position.is_action:
                    prev_actions_attack["position"].copy_(actions_attack["position"])
                if self._attacker_actions_desc.alpha.is_action:
                    prev_actions_attack["alpha"].copy_(actions_attack["alpha"])
                if self._attacker_actions_desc.category.is_action:
                    prev_actions_attack["category"].copy_(actions_attack["category"])

                _, actions_agent, _, test_recurrent_hidden_states_agent = self.actor_critic_agent.act(
                    batch,
                    test_recurrent_hidden_states_agent,
                    prev_actions_agent,
                    not_done_masks_agent,
                    deterministic=False
                )

                prev_actions_agent.copy_(actions_agent)


            action_list = []
            for i in range(self.config.NUM_PROCESSES):
                a_agent = actions_agent[i].item()
                a_attck_dict = {}
                if "position" in actions_attack:
                    a_attck_dict["position"] = actions_attack["position"][i].item()
                if "alpha" in actions_attack:
                    a_attck_dict["alpha"] = actions_attack["alpha"][i].item()
                if "category" in actions_attack:
                    a_attck_dict["category"] =  actions_attack["category"][i].item()
                a_tuple = {
                        "action_attack": a_attck_dict,     
                        "action" : a_agent,      
                }
                action_list.append(a_tuple)


            outputs = self.envs.step(action_list)

            observations, rewards_twos, dones, infos = [list(x) for x in zip(*outputs)]
            rewards_attack, rewards_agent =[list(x) for x in zip(*rewards_twos)]

            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observations[i]:
                        for observation in observations[i]['intermediate']:
                            frame = observations_to_image(observation, infos[i])
                            rgb_frames[i].append(frame)
                        del observations[i]['intermediate']

                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                           self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
                    audios[i].append(observations[i]['audiogoal'])   

            if config.DISPLAY_RESOLUTION != model_resolution:
                resize_observation(observations, model_resolution)
            batch = batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards_attack = torch.tensor(rewards_attack, dtype=torch.float, device=self.device)
            rewards_attack = rewards_attack.unsqueeze(1)
            rewards_agent = torch.tensor(rewards_agent, dtype=torch.float, device=self.device)
            rewards_agent = rewards_agent.unsqueeze(1)

            current_episode_reward_agent += rewards_agent
            current_episode_reward_attack += rewards_attack
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["reward_agent"] = current_episode_reward_agent[i].item()
                    episode_stats["reward_attack"] = current_episode_reward_attack[i].item()
                    episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                    episode_stats['euclidean_distance'] = norm(np.array(current_episodes[i].goals[0].position) -
                                                               np.array(current_episodes[i].start_position))
                    
                    current_episode_reward_agent[i] = 0
                    current_episode_reward_attack[i] = 0
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    t.update()

                    if len(self.config.VIDEO_OPTION) > 0:
                        fps = self.config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS if self.config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i][:-1],
                            scene_name=current_episodes[i].scene_id.split('/')[3],
                            sound=current_episodes[i].info['sound'],
                            sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=audios[i][:-1],
                            fps=fps
                        )

                        rgb_frames[i] = []
                        audios[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        top_down_map = plot_top_down_map(infos[i],
                                                         dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET)
                        scene = current_episodes[i].scene_id.split('/')[3]
                        writer.add_image('{}_{}_{}/{}'.format(config.EVAL.SPLIT, scene, current_episodes[i].episode_id,
                                                              config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5]),
                                         top_down_map,
                                         dataformats='WHC')

            (
                self.envs,
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
            ) = self._pause_envs_agent_attack(
                self._attacker_actions_desc,
                envs_to_pause,
                self.envs,
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


        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        ckpt_index = checkpoint_path.split("/")[-1].replace("ckpt.", '').replace(".pth", '')
        ct =datetime.datetime.now()

        stats_file = os.path.join(config.TENSORBOARD_DIR, '{}_stats_{}_ckpt{}_{}.json'.format(config.EVAL.SPLIT, config.SEED,ckpt_index,ct))
        new_stats_episodes = {','.join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        episode_reward_mean_agent = aggregated_stats["reward_agent"] / num_episodes
        episode_reward_mean_attack = aggregated_stats["reward_attack"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        
        logger.info(f"Average episode reward_agent: {episode_reward_mean_agent:.6f}")
        logger.info(f"Average episode reward_attack: {episode_reward_mean_attack:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(
                f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}"
            )

        if not config.EVAL.SPLIT.startswith('test'):
            writer.add_scalar("{}/reward_sum".format(config.EVAL.SPLIT), episode_reward_mean_agent+episode_reward_mean_attack, checkpoint_index)
            writer.add_scalar("{}/reward_agent".format(config.EVAL.SPLIT), episode_reward_mean_agent, checkpoint_index)
            writer.add_scalar("{}/reward_attack".format(config.EVAL.SPLIT), episode_reward_mean_attack, checkpoint_index)
            for metric_uuid in self.metric_uuids:
                writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid],
                                  checkpoint_index)

        self.envs.close()

        result = {
            'episode_reward_mean_agent': episode_reward_mean_agent,
            'episode_reward_mean_attack': episode_reward_mean_attack,
        }
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        return result
