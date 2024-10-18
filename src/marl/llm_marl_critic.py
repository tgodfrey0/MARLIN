from datetime import datetime
from typing import Optional

import ray.rllib.utils.exploration.stochastic_sampling

from src.marl.common_marl import parse_arguments, setup, get_base_config, go
from src.marl.llm_action_dict import LLMActionDict

# Parse command-line arguments
args = parse_arguments(True)

EPISODE_STEP_LIMIT: int = args.episode_step_limit
EPISODE_TOTAL_LIMIT: Optional[int] = args.episode_total_limit
TIMESTEP_TOTAL: int = args.timestep_total
ENV_CHANGE_RATE_EPS: int = args.env_change_rate_eps
LR: float = args.lr
PERFORMANCE_THRESHOLD: int = args.performance_threshold
EPISODE_THRESHOLD: int = args.episode_threshold
MODEL_NAME: str = args.llm
SCENARIO_NAME: str = args.scenario

_shared_llm_action_dict = LLMActionDict()
get_shared_llm_action_dict = lambda: _shared_llm_action_dict

if __name__ == "__main__":
  ray.init()

  setup()

  config = get_base_config(
      True,
      1,
      1,
      LR,
      "./hybrid_data/",
      f"PPO_LLM_param_sharing_critic_moves_{EPISODE_STEP_LIMIT}_lr_{LR}_{MODEL_NAME}_{'dynamic_env_' + str(ENV_CHANGE_RATE_EPS) if ENV_CHANGE_RATE_EPS != 0 else 'static_env'}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
      EPISODE_STEP_LIMIT,
      ENV_CHANGE_RATE_EPS,
      SCENARIO_NAME
  )

  config.sample_timeout_s = 600
  config.exploration_config = {
    # The Exploration class to use. In the simplest case, this is the name
    # (str) of any class present in the `rllib.utils.exploration` package.
    # You can also provide the python class directly or the full location
    # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
    # EpsilonGreedy").
    "type":              "src.marl.llm_stochastic_sampling.LLMStochasticSampling",
    # Add constructor kwargs here (if any).
    "threshold":         PERFORMANCE_THRESHOLD,
    "max_moves":         EPISODE_STEP_LIMIT,
    "llm_name":          MODEL_NAME,
    "episode_threshold": EPISODE_THRESHOLD,
    "env_change_rate":   ENV_CHANGE_RATE_EPS,
    "llm_action_dict":   get_shared_llm_action_dict
  }

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
