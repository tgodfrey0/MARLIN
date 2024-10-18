from datetime import datetime
from typing import Optional

import ray

from src.marl.common_marl import parse_arguments, setup, get_base_config, go

# Parse command-line arguments
args = parse_arguments(False)

EPISODE_STEP_LIMIT: int = args.episode_step_limit
EPISODE_TOTAL_LIMIT: Optional[int] = args.episode_total_limit
TIMESTEP_TOTAL: int = args.timestep_total
ENV_CHANGE_RATE_EPS: int = args.env_change_rate_eps
LR: float = args.lr
NUM_WORKERS: int = args.num_threads
NUM_ENVS_PER_WORKER: int = args.num_envs_per_thread
SCENARIO_NAME: str = args.scenario

if __name__ == "__main__":
  ray.init()

  setup()

  config = get_base_config(
      False,
      3,  # NUM_WORKERS,
      2,  # NUM_ENVS_PER_WORKER,
      LR,
      "./marl_data/",
      f"PPO_param_sharing_critic_moves_{EPISODE_STEP_LIMIT}_lr_{LR}_{'dynamic_env_' + str(ENV_CHANGE_RATE_EPS) if ENV_CHANGE_RATE_EPS != 0 else 'static_env'}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
      EPISODE_STEP_LIMIT,
      ENV_CHANGE_RATE_EPS,
      SCENARIO_NAME
  )

  results = go(config, TIMESTEP_TOTAL, "hybrid_llm_marl")

  # restored_tuner = tune.Tuner.restore(
  #     path = "/home/toby/projects/uni/internship/Hybrid_LLM_MARL/output/CentralizedCritic_2024-07-19_15-19-29",
  #     trainable = CentralizedCritic,
  #     param_space = config.to_dict(),
  #     # Important to set this to True b/c the previous trial had failed (due to our
  #     # `CrashAfterNIters` callback).
  #     resume_errored = True,
  # )
  # # Continue the experiment exactly where we left off.
  # tuner_results = restored_tuner.fit()
