import argparse
import os
import random
from threading import Lock
from typing import Union, Optional, Dict

import gymnasium as gym
import numpy as np
from ray import air, logger, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms import Algorithm, PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.train import CheckpointConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.registry import POLICIES

from src.marl.centralised_critic import CentralizedCriticModel, CentralizedCritic
from src.marl.corridor_env import CorridorEnv

eps = 0
algo: Optional[Algorithm] = None
file_lock = Lock()


def parse_arguments(llm_version: bool):
  parser = argparse.ArgumentParser(description = "Script with configurable constants")

  parser.add_argument("--episode-step-limit", type = int, default = 50,
                      help = "Step limit for each episode (default: 50)")

  parser.add_argument("--episode-total-limit", type = int, default = None,
                      help = "Total episode limit (default: None)")

  parser.add_argument("--env-change-rate-eps", type = int, default = 0,
                      help = "Change the environment after this many episodes (default: 0)")

  parser.add_argument("--timestep-total", type = int, default = None,
                      help = "Total number of timesteps (default: calculated based on episode limits)")
  parser.add_argument("--lr", type = float, default = 1e-5)
  parser.add_argument("--scenario", type = str, default = None)
  if llm_version:
    parser.add_argument("--performance-threshold", type = float, default = 0.75,
                        help = "The performance below which to use the LLM (default: 0.75)")
    parser.add_argument("--llm", type = str, default = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help = "LLM to use (default: meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--episode-threshold", type = int, default = 5)
  else:
    parser.add_argument("--num-threads", type = int, default = 3)
    parser.add_argument("--num-envs-per-thread", type = int, default = 3)
  args = parser.parse_args()

  # Calculate TIMESTEP_TOTAL based on provided arguments if not directly specified
  if args.timestep_total is None:
    if args.episode_total_limit is None:
      raise RuntimeError("You must supply either timestep_total or episode_total_limit")
    else:
      args.timestep_total = args.episode_total_limit * args.episode_step_limit

  return args


def setup():
  ModelCatalog.register_custom_model(
      "cc_model",
      CentralizedCriticModel
  )

  register_env("corridor", lambda config: CorridorEnv(config))


def get_base_config(
    is_hybrid: bool,
    num_env_runners,
    num_envs_per_runner,
    lr,
    csv_dir,
    csv_filename,
    episode_step_limit,
    env_change_rate_eps,
    scenario_name):
  config = (
    PPOConfig()
    .env_runners(
        batch_mode = "complete_episodes",
        num_env_runners = num_env_runners,  # Num threads
        enable_connectors = False,
    )
    .framework("torch")
    .resources(num_gpus = 0)
    .rollouts(
        num_envs_per_worker = num_envs_per_runner)  # Num of envs per thread -- for now using the LLMStochasticSampler this must stay at 1. TODO Fix indexing somehow to support multiple envs per worker
    .training(
        lr = lr,
        kl_coeff = 0.01,
        clip_param = 0.2,
        use_gae = True,
        use_critic = True,
        vf_loss_coeff = 0.005,
        model = {"custom_model": "cc_model", "vf_share_layers": True},
    )
    .environment("corridor", env_config = {
      "csv_path":            os.path.abspath(csv_dir),
      "csv_filename":        csv_filename,
      "episode_step_limit":  episode_step_limit,
      "env_change_rate_eps": env_change_rate_eps,  # 0 for no env change
      "scenario_name":       scenario_name
    })
    .multi_agent(
        policies = {
          "pol1": (
            None,
            gym.spaces.Box(
                low = np.array([0, 0, 0, 0]),  # Minimum values (own x, own y, own heading)
                high = np.array([CorridorEnv.GRID_WIDTH - 1, CorridorEnv.GRID_HEIGHT - 1, CorridorEnv.GRID_WIDTH - 1,
                                 CorridorEnv.GRID_HEIGHT - 1]),
                # Maximum values
                dtype = np.int8
            ),
            gym.spaces.Discrete(5),
            # `framework` would also be ok here.
            PPOConfig.overrides(framework_str = "torch"),
          )
        },
        policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "pol1"
    ).callbacks(
        callbacks_class = HybridEvaluationCallbacks if is_hybrid else MARLEvaluationCallbacks
    )
  )
  return config


def go(config, total_timesteps, wandb_proj):
  storage_path = os.path.abspath("./output")
  logger.info(f"Storage path: {storage_path}")

  tuner = tune.Tuner(
      CentralizedCritic,
      param_space = config.to_dict(),
      run_config = air.RunConfig(
          stop = {"timesteps_total": total_timesteps},
          storage_path = storage_path,
          callbacks = [
            WandbLoggerCallback(project = wandb_proj),
          ],
          checkpoint_config = CheckpointConfig(
              checkpoint_frequency = 1,
              checkpoint_at_end = True
          )
      ),
  )
  results = tuner.fit()
  return results


class HybridEvaluationCallbacks(DefaultCallbacks):
  def on_episode_end(  # Use this to checkpoint specific episodes?
      self,
      *,
      episode: Union[EpisodeType, Episode, EpisodeV2],
      env_runner: Optional["EnvRunner"] = None,
      metrics_logger: Optional[MetricsLogger] = None,
      env: Optional[gym.Env] = None,
      env_index: int,
      rl_module: Optional[RLModule] = None,
      # TODO (sven): Deprecate these args.
      worker: Optional["EnvRunner"] = None,
      base_env: Optional[BaseEnv] = None,
      policies: Optional[Dict[PolicyID, Policy]] = None,
      **kwargs,
  ) -> None:
    global eps
    logger.info(f"Episode ENDED {eps}")
    eps += 1
    if eps % 10 == 0:
      path = os.path.join(os.path.abspath(f"./models/hybrid/"), f"episode_{eps}")
      if not os.path.exists(path):
        os.mkdir(path)
      num_dirs = len(os.listdir(path))
      final_path = os.path.join(path, f"trial_{num_dirs}/")
      if not os.path.exists(final_path):
        os.mkdir(final_path)
      policies.get("pol1").export_checkpoint(final_path)


class MARLEvaluationCallbacks(DefaultCallbacks):
  def on_episode_end(  # Use this to checkpoint specific episodes?
      self,
      *,
      episode: Union[EpisodeType, Episode, EpisodeV2],
      env_runner: Optional["EnvRunner"] = None,
      metrics_logger: Optional[MetricsLogger] = None,
      env: Optional[gym.Env] = None,
      env_index: int,
      rl_module: Optional[RLModule] = None,
      # TODO (sven): Deprecate these args.
      worker: Optional["EnvRunner"] = None,
      base_env: Optional[BaseEnv] = None,
      policies: Optional[Dict[PolicyID, Policy]] = None,
      **kwargs,
  ) -> None:
    global eps
    logger.info(f"Episode ENDED {eps}")
    eps += 1
    if eps % 10 == 0:
      with file_lock:
        try:
          path = os.path.join(os.path.abspath(f"./models/marl/"), f"episode_{eps}/")
          if not os.path.exists(path):
            os.mkdir(path)
          num_dirs = len(os.listdir(path))
          final_path = os.path.join(path, f"trial_{num_dirs}/")
          if not os.path.exists(final_path):
            os.mkdir(final_path)
          policies.get("pol1").export_checkpoint(final_path)
        except FileExistsError:
          try:
            path = os.path.join(os.path.abspath(f"./models/marl/"), f"episode_{eps}/")
            if not os.path.exists(path):
              os.mkdir(path)
            num_dirs = len(os.listdir(path))
            final_path = os.path.join(path, f"trial_{num_dirs}_{random.randint(0, int(1e10))}/")
            if not os.path.exists(final_path):
              os.mkdir(final_path)
            policies.get("pol1").export_checkpoint(final_path)
          except:
            pass
