import math
import os
import traceback
from copy import deepcopy
from datetime import datetime
from queue import Queue
from random import random, randint
from statistics import mean
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
from ray import logger
from ray.rllib.models import ModelV2, ActionDistribution
from ray.rllib.utils import override
from ray.rllib.utils.exploration import Exploration
from ray.rllib.utils.exploration.stochastic_sampling import StochasticSampling
from ray.rllib.utils.typing import TensorType

from src.llms.llm_move_gen import LLMMoveGen
from src.marl.llm_action_dict import LLMActionDict
from src.marl.meta_info import MetaInfo
from src.utils.movement import Action
from src.utils.scenarios import Scenario
from src.utils.utils import Utils


# TODO How to taper down following perfect plan as ep_num increases
# TODO Fix multithreaded llmactiondict


def get_pos_goal_list(pos, goal, ids):
  return [(pos[i], goal[i]) for i in ids]


def get_llm_perf(model: str, scenario: Scenario) -> float:
  llm_perf = {
    "gemini-1.5-flash":                      {"overall": 0.734},
    "gemini-1.5-pro":                        {"overall": 0.537},
    "gpt-4o":                                {"overall": 0.585},
    "gpt-4o-mini":                           {"overall": 0.592},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
      "Asymmetrical_Two_Slot_Corridor": 0.821,
      "Single_Slot_Corridor":           0.429,
      "Symmetrical_Two_Slot_Corridor":  0.429,
      "Two_Path_Corridor":              0.357,
      "Maze_Like_Corridor":             0.333,
      "overall":                        0.5
    }
  }

  try:
    llm_data = llm_perf[model]
  except KeyError:
    raise RuntimeError(f"Unrecognised LLM: {model}")

  if scenario.name in llm_data.keys():
    return llm_data[scenario.name]
  else:
    return llm_data["overall"]


def decay(x, base = 1000):
  return max(base ** ((-x) ** 1 / 2), 0.01)


class LLMStochasticSampling(StochasticSampling):
  def __init__(self, action_space: gym.spaces.Space, *, framework: str, model: ModelV2, **kwargs):
    self.use_llm = False
    self.performance_threshold = kwargs.pop("threshold", 0.5)
    self.max_moves = kwargs.pop("max_moves", 1)
    self.llm_name = kwargs.pop("llm_name", None)
    self.env_change_rate = kwargs.pop("env_change_rate", 0)
    self.episode_threshold = kwargs.pop("episode_threshold", 25)
    self.llm_action_dict = kwargs.pop("llm_action_dict", lambda: LLMActionDict())
    self.plan = None
    self.plan_perf = 0.
    self.plan_key = None
    self.long_term_avg_perf = Queue(maxsize = 25)
    self.is_hybrid_episode = False

    self.policy_pretrain_eps: int = self.episode_threshold
    self.episode_threshold += self.policy_pretrain_eps

    self.ids = None
    self.starting_pos = None
    self.goal_pos = None
    self.pos = None
    self.valids = None
    self.ep_num = None
    self.step_num = None
    self.scenario = None
    self.past_perf_avg = None

    assert self.llm_name is not None

    Utils.set_csv_file("llm", os.path.abspath("./llm_data/"),
                       f"LLM_{self.llm_name.split('/')[1]}_data.csv")

    super().__init__(action_space, framework = framework, model = model, **kwargs)

  @override(Exploration)
  def get_exploration_action(
      self,
      *,
      action_distribution: ActionDistribution,
      timestep: Optional[Union[int, TensorType]] = None,
      explore: bool = True
  ):
    try:
      self.last_timestep = (
        timestep if timestep is not None else self.last_timestep + 1
      )
      # logger.info(f"({self.worker_index}) random timesteps: {self.random_timesteps}")
      return self.f(action_distribution, timestep, explore)
    except Exception as e:
      logger.critical(f"Exception in fn")
      logger.critical(f"Exception occurred when generating LLM action: {type(e).__name__} - {str(e)}")
      logger.critical(traceback.format_exc())
      logger.critical(MetaInfo.get_keys(self.worker_index))
      for k in MetaInfo.get_keys(self.worker_index):
        logger.critical(f"{k}: {MetaInfo.get(self.worker_index, k)}")
      logger.critical(self.llm_action_dict())
      logger.critical(f"plan {self.plan}")
      logger.critical(f"plan_perf {self.plan_perf}")
      raise e

  def f(self, action_distribution: ActionDistribution, timestep: Optional[Union[int, TensorType]], explore: bool):
    logger.info(f"({self.worker_index}) timestep {timestep}")

    try:
      self._get_metainfo()
    except RuntimeError as e:
      logger.error(f"Exception occurred when generating LLM action: {type(e).__name__} - {str(e)}")
      logger.error(traceback.format_exc())
      return super().get_exploration_action(action_distribution = action_distribution, timestep = timestep,
                                            explore = explore)

    if timestep == 0 or self.ep_num < self.policy_pretrain_eps or self.ep_num > 1625:
      return super().get_exploration_action(action_distribution = action_distribution, timestep = timestep,
                                            explore = explore)

    if self.step_num is None:
      logger.warn("step_num is None")
      self.step_num = 0

    logger.info(f"({self.worker_index}) {datetime.now()} ss: past_perf_avg {self.past_perf_avg}")

    self.eval_use_llm(action_distribution = action_distribution, timestep = timestep, explore = explore)

    if self.use_llm:
      logger.info(f"({self.worker_index}) Using LLM to generate actions")
      return self.query_llm_plan(action_distribution = action_distribution, timestep = timestep, explore = explore)
    else:
      logger.info(f"({self.worker_index}) Using Action Distribution to generate actions")
      return self.query_action_distribution(action_distribution = action_distribution, timestep = timestep,
                                            explore = explore)

  def eval_use_llm(self, action_distribution, timestep, explore):
    if self.step_num == 0:
      self.is_hybrid_episode = False

      if self.past_performance >= 0:
        if self.use_llm and self.past_performance != self.plan_perf:  # E.g. if env changes
          self.llm_action_dict().update_perf(self.plan_key, self.past_performance)

      self._update_long_term_perfs()
      assert set(self.starting_pos.items()) == set(self.pos.items()) and len(self.starting_pos) == len(self.pos)
      self.plan, self.plan_perf = self.llm_action_dict().get(set(self.valids),
                                                             get_pos_goal_list(self.starting_pos, self.goal_pos,
                                                                               self.ids))
      logger.info(
          f"({self.worker_index}) Plan performance {self.plan_perf} (key {self.llm_action_dict().gen_key(set(self.valids), get_pos_goal_list(self.starting_pos, self.goal_pos, self.ids))})")

      # First n steps
      if self.ep_num < self.policy_pretrain_eps:
        logger.info(f"({self.worker_index}) Using action distribution as within spool up episode limit")
        self.use_llm = False
        self.is_hybrid_episode = True  # Lockout from changing

      # Perfect plan for one in every n episodes
      elif self.plan_perf == 1:
        logger.info(f"({self.worker_index}) Using perfect plan")
        self.use_llm = True

      # First 2n steps
      elif self.ep_num < self.episode_threshold:
        logger.info(f"({self.worker_index}) Using LLM as within mandatory usage limit")
        self.use_llm = True
        self.is_hybrid_episode = True  # Lockout from changing

      # Performance is below average LLM performance
      elif self.past_perf_avg < get_llm_perf(self.llm_name, self.scenario):
        logger.info(f"({self.worker_index}) Using LLM as performance is below average LLM performance")
        self.use_llm = True

      # Performance has plateaued
      elif self.perf_plateau():
        logger.info(f"({self.worker_index}) Performance has plateaued, using LLM")
        self.use_llm = True

      # Performance must be above average LLM performance
      else:
        logger.info(f"({self.worker_index}) Using action distribution as performance is above LLM average performance.")
        self.use_llm = False

    if not self.is_hybrid_episode and self.step_num == (self.max_moves / 2) and random() < 0.1:
      if self.use_llm:
        logger.info(f"({self.worker_index}) Hybrid Episode -- now using action distribution")
        self.use_llm = not self.use_llm
        self.is_hybrid_episode = True
      elif not self.use_llm:
        logger.info(f"({self.worker_index}) Hybrid Episode -- now using LLMs")
        self.use_llm = not self.use_llm
        self.is_hybrid_episode = True

        hybrid_ep_starting_pos = self.pos
        self._gen_plan(
            self.scenario,
            self.valids,
            self.ids,
            hybrid_ep_starting_pos,
            self.goal_pos,
            hybrid_ep_starting_pos,
            self.ep_num,
            self.step_num,
            (self.max_moves - self.step_num)
        )
        self.plan_perf = 1  # Force using LLM for rest of episode

  def query_action_distribution(self, action_distribution, timestep, explore):
    return super().get_exploration_action(action_distribution = action_distribution, timestep = timestep,
                                          explore = explore)

  def get_action_from_plan(self, action_distribution, timestep, explore):
    if self.plan is not None:
      acts = torch.tensor(np.array([Action.WAIT.value, Action.WAIT.value]))
      if self.step_num < len(self.plan):
        acts = torch.tensor(np.array([self.plan[self.step_num]["alice"], self.plan[self.step_num]["bob"]]))

      if explore:
        logps = action_distribution.logp(acts)
        # if self.plan_perf == 1:
        #   logps = pow(math.e, logps) * 2
        #   logps = logps.clamp(0, 1).log()
      else:
        logps = torch.zeros_like(acts)

      return acts, logps
    else:
      logger.info(f"({self.worker_index}) Cannot get action from None plan")
      return super().get_exploration_action(action_distribution = action_distribution, timestep = timestep,
                                            explore = explore)

  def query_llm_plan(self, action_distribution, timestep, explore):
    if self.plan is not None and (
        (self.plan_perf > self.past_perf_avg) or
        (self.plan_perf == 1)
    ):  # Plan must exist and either be better than current performance or be a perfect plan. If plan == current performance then we make a new plan and try to get better (if the plan is worse it will not overwrite the better one)
      logger.info(f"({self.worker_index}) Getting action from plan")
      return self.get_action_from_plan(action_distribution = action_distribution, timestep = timestep,
                                       explore = explore)
    else:
      if self.step_num == 0:
        if self.plan_perf is None:
          logger.info(f"({self.worker_index}) Creating new plan -- old plan did not exist")
        elif self.plan_perf <= self.past_perf_avg:
          logger.info(
              f"({self.worker_index}) Creating new plan -- old plan was below or equal to average performance ({self.plan_perf} <= {self.past_perf_avg})")

        self._gen_plan(self.scenario,
                       self.valids,
                       self.ids,
                       self.starting_pos,
                       self.goal_pos,
                       self.pos,
                       self.ep_num,
                       self.step_num,
                       self.max_moves)
        return self.get_action_from_plan(action_distribution = action_distribution, timestep = timestep,
                                         explore = explore)
      else:
        return self.query_action_distribution(action_distribution = action_distribution, timestep = timestep,
                                              explore = explore)

  def _gen_plan(self, scenario: Scenario, valids, ids, starting_pos, goal_pos, pos, ep_num, step_num, n_moves):
    if scenario is None:
      raise RuntimeError("scenario is None")
    logger.info(scenario.llm_prompt)
    g = LLMMoveGen(scenario, ids, valids, starting_pos, goal_pos, self.llm_name, self.env_change_rate, write_csv = True,
                   write_conversation = True)
    moves_dict, perf = g.gen_moves(n_moves, translate_moves = True, verbose = False)
    if perf > 0:
      logger.info(f"({self.worker_index}) Generated plan has performance of {perf}")
      self.llm_action_dict().set(set(valids), get_pos_goal_list(self.starting_pos, self.goal_pos, self.ids), moves_dict,
                                 perf)
      self.plan = moves_dict
      self.plan_perf = perf
      self.plan_key = self.llm_action_dict().gen_key(set(valids),
                                                     get_pos_goal_list(self.starting_pos, self.goal_pos, self.ids))

  def _get_metainfo(self):
    sn = self.step_num
    self.ids = MetaInfo.get(self.worker_index, "agent_ids")
    self.starting_pos = MetaInfo.get(self.worker_index, "agent_starting_pos")
    self.goal_pos = MetaInfo.get(self.worker_index, "agent_goal_pos")
    self.pos = MetaInfo.get(self.worker_index, "agent_pos")
    self.valids = MetaInfo.get(self.worker_index, "valid_pos")
    self.ep_num = MetaInfo.get(self.worker_index, "ep_num")
    self.step_num = MetaInfo.get(self.worker_index, "step_num")
    self.scenario = MetaInfo.get(self.worker_index, "scenario")
    self.past_perf_avg = MetaInfo.get(self.worker_index, "past_performance_avg")
    self.past_performance = MetaInfo.get(self.worker_index, "past_performance")

    # logger.info(f"({self.worker_index}) {datetime.now()} ss: MetaInfo pulled")
    # logger.info(f"({self.worker_index}) {datetime.now()} ss: step_num {self.step_num} (was {sn})")
    # logger.info(f"({self.worker_index}) {datetime.now()} ss: pos {self.pos}")
    # logger.info(f"({self.worker_index}) {datetime.now()} ss: starting_pos {self.starting_pos}")
    # logger.info(f"({self.worker_index}) {datetime.now()} ss: goal_pos {self.goal_pos}")

    if self.ids is None:
      raise RuntimeError("ids is None")
    # if starting_pos is None:
    #   raise RuntimeError("starting_pos is None")
    if self.goal_pos is None:
      raise RuntimeError("goal_pos is None")
    if self.pos is None:
      raise RuntimeError("pos is None")
    if self.valids is None:
      raise RuntimeError("valids is None")
    if self.ep_num is None:
      raise RuntimeError("ep_num is None")
    if self.past_perf_avg is None:
      self.past_perf_avg = -1.0
    if self.past_performance is None:
      self.past_performance = -1.0

  def _update_long_term_perfs(self):
    if self.long_term_avg_perf.full():
      _ = self.long_term_avg_perf.get()

    self.long_term_avg_perf.put(self.past_perf_avg)

  def perf_plateau(self, tolerance: float = 0.075) -> bool:
    perf_list = list(deepcopy(self.long_term_avg_perf.queue))

    if len(perf_list) != self.long_term_avg_perf.maxsize:
      return False

    if mean(perf_list) == 1:
      return False

    rel_changes = np.diff(perf_list) / perf_list[:-1]
    return np.all(np.abs(rel_changes) <= tolerance)
