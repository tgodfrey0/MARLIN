import os
from datetime import datetime

import ray
from ray import train, logger, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from corridor_env import CorridorEnv

if __name__ == "__main__":
  ray.init()

  register_env("corridor", lambda config: CorridorEnv(config))

  # storage_path = "/home/tg/projects/p3p/tb-marl/output"
  storage_path = os.path.abspath("./output")
  logger.info(f"Storage path: {storage_path}")

  config = (
    PPOConfig().training(
        lr = 1e-5,
        kl_coeff = 0.01,
        clip_param = 0.2,
        model = {
          "vf_share_layers": True
        },
        vf_loss_coeff = 0.005)
    .environment("corridor", env_config = {
      "csv_path":           os.path.abspath("./marl_data/"),
      "csv_filename":       f"PPO_param_sharing_no_critic_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
      "episode_step_limit": 50
    })
    .env_runners(
        batch_mode = "complete_episodes",
        num_env_runners = 0,
        enable_connectors = False,
    )
    .framework("torch")
    .resources(num_gpus = 0)
    .rollouts(num_envs_per_worker = 1)
    .multi_agent(  # This block allows for parameter sharing
        policies = {"shared_policy"},
        policy_mapping_fn = (lambda agent_id, *args, **kwargs: "shared_policy"),
    )
  )

  tuner = tune.Tuner(
      PPO,
      run_config = train.RunConfig(
          storage_path = storage_path,
          callbacks = [WandbLoggerCallback(project = "hybrid_llm_marl_no_critic")],
          checkpoint_config = train.CheckpointConfig(
              checkpoint_frequency = 5,
              checkpoint_at_end = True
          )
      ),
      param_space = config,
  )

  # config = DQNConfig().training(lr=tune.grid_search([1e-9])) \
  #                     .environment("corridor") \
  #                     .framework("torch") \
  #                     .resources(num_gpus=0) \
  #                     .rollouts(num_envs_per_worker=1)

  # tuner = tune.Tuner(
  #     "DQN",
  #     run_config=train.RunConfig(
  #         stop={"timesteps_total" : 5000},
  #         storage_path=storage_path,
  #         callbacks=[WandbLoggerCallback(project="tb_marl")],
  #         checkpoint_config=train.CheckpointConfig(
  #           checkpoint_frequency=5,
  #           checkpoint_at_end=True
  #         )
  #     ),
  #     param_space=config,
  # )

  results = tuner.fit()

  ray.shutdown()
