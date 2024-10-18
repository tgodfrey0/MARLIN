import math
import os.path
from copy import deepcopy
from datetime import datetime
from random import choices
from statistics import mean, median
from typing import Any, Dict, Tuple, List
from ray import train
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray import logger
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.train import Checkpoint

from src.marl.meta_info import MetaInfo
from src.utils.grid import Grid
from src.utils.movement import Action
from src.utils.scenarios import *
from src.utils.utils import Utils

PERFORMANCE_AVG_NUM_EPS = 5


# ~~T O D O~~ All envs should have same scenario at same time? -- No, this will cause issues with certain envs reaching the threshold at different times
# TODO Don't randomise anything? Can this then improve generality?

class CorridorEnv(MultiAgentEnv):
  N_AGENTS = 2
  GRID_WIDTH = 3
  GRID_HEIGHT = 8

  def __init__(self, config):
    super().__init__()

    # print(config.worker_index)

    self.csv_path_name = "marl"

    Utils.set_csv_file(self.csv_path_name, config["csv_path"], config["csv_filename"])
    try:
      self.worker_index = config.worker_index
    except AttributeError:
      try:
        self.worker_index = config["worker_index"]
      except KeyError:
        raise RuntimeError("No worker index")

    self.env_change_rate_eps = config["env_change_rate_eps"]

    self._agent_ids = ["alice", "bob"]
    self.meta_info: Dict[str, Any] = {"past_performance_avg": 0.0}
    self.past_performances: List[float] = [0.0 for _ in range(PERFORMANCE_AVG_NUM_EPS)]

    self.action_space = spaces.Discrete(5)  # * Forward, Backwards, Clockwise, Anticlockwise, Wait
    self._ep_step_limit = config["episode_step_limit"]
    self.total_steps = 0
    self.ep_num: int = 0

    self.agent_info = {i: {} for i in self._agent_ids}

    self._scenario_list = get_all_instantiated_scenarios()
    self.scenario = None
    if config["scenario_name"] is None:
      self.scenario = choices(self._scenario_list, k = 1)[0]
    else:
      for s in self._scenario_list:
        if s.name == config["scenario_name"]:
          self.scenario = s

    if self.scenario is None:
      self.scenario = choices(self._scenario_list, k = 1)[0]

    logger.info(self.scenario.name)
    MetaInfo.put(self.worker_index, "scenario", self.scenario)
    self.valid_pos = self.scenario.valid_pos

    logger.info(f"({self.worker_index}) Environment")
    logger.info(
        "\n".join(
            Grid.render_grid(
                Grid.unflatten_grid(self.valid_pos, CorridorEnv.GRID_WIDTH, CorridorEnv.GRID_HEIGHT))) + "\n")

    self.COLLISION_PENALTY = 0.5

    self.observation_space = gym.spaces.Box(
        low = np.array(np.array([0, 0, 0, 0])),  # Minimum values (own x, own y, own heading, goal x, goal y)
        high = np.array([CorridorEnv.GRID_WIDTH - 1, CorridorEnv.GRID_HEIGHT - 1, CorridorEnv.GRID_WIDTH - 1,
                         CorridorEnv.GRID_HEIGHT - 1]),
        # Maximum values
        dtype = np.int8
    )

    # self.single_obs_space = gym.spaces.Box(
    #     low = np.array(low_arr),  # Minimum values (own x, own y, own heading)
    #     high = np.array(high_arr),  # Maximum values
    #     dtype = np.int8
    # )

    # self.observation_space = gym.spaces.Dict({
    #   "alice": self.single_obs_space,
    #   "bob":   self.single_obs_space
    # })

    self._reset_vars()

    self.agent_total_distances = {
      "alice": math.sqrt(abs(self.agent_goal_pos["alice"][0] - self.agent_pos["alice"][0]) ** 2 + abs(
          self.agent_goal_pos["alice"][1] - self.agent_pos["alice"][1]) ** 2),
      "bob":   math.sqrt(abs(self.agent_goal_pos["bob"][0] - self.agent_pos["bob"][0]) ** 2 + abs(
          self.agent_goal_pos["bob"][1] - self.agent_pos["bob"][1]) ** 2)
    }

  def reset(self, *, seed = None, options = None):
    logger.info(f"({self.worker_index}) IN RESET")
    loc_list = [(self.agent_pos[i], self.agent_goal_pos[i], self.agent_starting_pos[i]) for i in self._agent_ids]
    avg_perf = Utils.calc_multiagent_avg_perf(loc_list)
    if MetaInfo.get(self.worker_index,
                    "step_num") is not None and self.ep_num > 0:  # Ignore forced WAIT. Ignore ep = 0 as the env is created then reset causing the ep to increase but taking no steps
      Utils.write_csv(
          self.csv_path_name,
          ["episode", "alice_start", "alice_end", "alice_goal", "bob_start", "bob_end", "bob_goal", "performance",
           "scenario", "env_change_rate"],
          [self.ep_num, self.agent_starting_pos["alice"], self.agent_pos["alice"], self.agent_goal_pos["alice"],
           self.agent_starting_pos["bob"], self.agent_pos["bob"], self.agent_goal_pos["bob"],
           avg_perf, self.scenario.name, self.env_change_rate_eps])

    logger.info(f"({self.worker_index}) {datetime.now()} this perf {avg_perf}")
    index: int = self.ep_num % PERFORMANCE_AVG_NUM_EPS
    self.past_performances[index] = avg_perf
    logger.info(f"({self.worker_index}) {datetime.now()} past perfs {self.past_performances}")
    MetaInfo.put(self.worker_index, "past_performance_avg", median(self.past_performances))
    MetaInfo.put(self.worker_index, "past_performance", avg_perf)

    # logger.info(f"({self.worker_index}) MetaInfo: {self.meta_info}")
    self.ep_num += 1
    logger.info(f"({self.worker_index}) RESET")

    self._reset_vars()
    return self._get_obs(), self._get_info()  # obs & info

  def _reset_vars(self):
    self._step_counter = 0
    MetaInfo.put(self.worker_index, "step_num", self._step_counter)

    self.agent_goal_pos = {"alice": (0, 0), "bob": (0, 0)}

    self.agent_starting_pos = {"alice": (0, 0), "bob": (0, 0)}

    old_valid_pos = self.valid_pos
    if self.env_change_rate_eps != 0 and self.ep_num > 0:
      if self.ep_num % self.env_change_rate_eps == 0:
        while old_valid_pos == self.valid_pos:
          self.scenario = MazeLikeCorridor()  # choices(self._scenario_list, k = 1)[0]
          self.valid_pos = self.scenario.valid_pos

    if old_valid_pos != self.valid_pos:
      logger.info(f"({self.worker_index}) Environment has changed")
      logger.info(
          "\n".join(
              Grid.render_grid(
                  Grid.unflatten_grid(self.valid_pos, CorridorEnv.GRID_WIDTH, CorridorEnv.GRID_HEIGHT))) + "\n")

    # while (
    #     (self.agent_starting_pos["alice"] == self.agent_starting_pos["bob"]) or
    #     (self.agent_starting_pos["alice"] == self.agent_goal_pos["alice"]) or
    #     (self.agent_starting_pos["bob"] == self.agent_goal_pos["bob"]) or
    #     (self.agent_goal_pos["alice"] == self.agent_goal_pos["bob"])):
    #   self.agent_starting_pos["alice"], self.agent_starting_pos["bob"], self.agent_goal_pos["alice"], \
    #     self.agent_goal_pos["bob"] = choices(self.valid_pos, k = 4)

    while self.agent_starting_pos["alice"] == self.agent_starting_pos["bob"]:
      self.agent_starting_pos["alice"], self.agent_starting_pos["bob"] = choices(self.valid_pos, k = 2)

    # logger.info(
    #     f'Alice: {self.agent_starting_pos["alice"]} {self.agent_starting_dir["alice"]}\nBob: {self.agent_starting_pos["bob"]} {self.agent_starting_dir["bob"]}')

    self.agent_starting_pos["alice"] = (1, 0)
    self.agent_starting_pos["bob"] = (1, 7)

    # Fixed goal random start
    self.agent_pos: Dict[str, Tuple[int, int]] = {
      "alice": deepcopy(self.agent_starting_pos["alice"]),
      "bob":   deepcopy(self.agent_starting_pos["bob"])
    }

    self.agent_goal_pos = {
      "alice": (1, 7),  # deepcopy(self.agent_starting_pos["bob"]),
      "bob":   (1, 0)  # deepcopy(self.agent_starting_pos["alice"])
    }

    MetaInfo.put(self.worker_index, "scenario", self.scenario)
    MetaInfo.put(self.worker_index, "agent_ids", self._agent_ids)
    MetaInfo.put(self.worker_index, "agent_starting_pos", self.agent_starting_pos)
    MetaInfo.put(self.worker_index, "agent_goal_pos", self.agent_goal_pos)
    MetaInfo.put(self.worker_index, "agent_pos", self.agent_pos)
    MetaInfo.put(self.worker_index, "ep_num", self.ep_num)
    MetaInfo.put(self.worker_index, "valid_pos", self.valid_pos)

    assert (self.agent_pos == self.agent_starting_pos)

    self._step_counter = 0
    self.agent_reported_done = {"alice": False, "bob": False}
    self.agent_reward_penalties = {
      "alice": 0.0,
      "bob":   0.0
    }

    self.agent_wait_count = {
      "alice": 0,
      "bob":   0
    }

  def step(self, action_dict):
    s = f"{self.ep_num}/{self._step_counter}: {self.agent_pos} -- Actions: {action_dict}"
    self._update_pos(action_dict)
    MetaInfo.put(self.worker_index, "agent_pos", self.agent_pos)
    logger.info(f"({self.worker_index}) {s} -> {self.agent_pos}")

    terminateds = self._get_terminateds()

    truncateds = self._get_truncateds(terminateds.keys())

    rewards = self._get_rewards()

    info = self._get_info()

    self._step_counter += 1
    # logger.info(f"({self.worker_index}) {datetime.now()} step counter updated to {self._step_counter}")
    MetaInfo.put(self.worker_index, "step_num", self._step_counter)
    self.total_steps += 1

    assert (len(terminateds) == len(truncateds))

    return self._get_obs(), rewards, terminateds, truncateds, info  # obs reward terminateds truncateds info

  def action_space_sample(self, agent_ids: list = None) -> Dict[Any, Any]:
    actions = {}
    for agent_id in self._agent_ids:
      if self.agent_pos[agent_id] == self.agent_goal_pos[agent_id]:
        actions[agent_id] = Action.WAIT.value
      else:
        actions[agent_id] = np.random.randint(0, self.action_space.n)
    return actions

  def action_space_contains(self, action: Dict[Any, Any]) -> bool:
    for _, act in action.items():
      if not self.action_space.contains(act):
        return False
    return True

  def observation_space_contains(self, obs: Dict[Any, Any]) -> bool:
    for _, o in obs.items():
      if not self.observation_space.contains(o):
        return False
    return True

  def _get_rewards(self):
    rewards = {}

    for id in self._agent_ids:
      if not self.agent_reported_done[id]:
        rewards[id] = self._calc_reward(id)

    return rewards

  def _get_info(self):
    info = {}

    for id in self._agent_ids:
      if not self.agent_reported_done[id]:
        info[id] = deepcopy(self.agent_info[id])

    return info

  def _get_terminateds(self) -> Dict[str, bool]:
    terminateds = {}
    if self._step_counter == (self._ep_step_limit - 1):
      self.agent_reported_done = {i: True for i in self._agent_ids}
      terminateds = {i: True for i in self._agent_ids}
      # logger.info(f"({self.worker_index}) STEP LIMIT REACHED")
    # else:
    #   for id, pos in self.agent_pos.items():
    #     if (pos == self.agent_final_pos[id] and not self.agent_reported_done[id]):
    #       self.agent_reported_done[id] = True
    #       terminateds[id] = True

    if terminateds != {}:
      terminateds["__all__"] = all(self.agent_reported_done.values())
    else:
      terminateds["__all__"] = False

    return terminateds

  def _get_truncateds(self, keys) -> Dict[str, bool]:
    truncateds = {}
    for id in self._agent_ids:
      if (id not in keys):
        continue

      if (self._step_counter == self._ep_step_limit):
        truncateds[id] = True
      else:
        truncateds[id] = False

    if len(truncateds) == len(self._agent_ids):  # All agents in list
      truncateds["__all__"] = all(truncateds.values())
    else:
      truncateds["__all__"] = False

    return truncateds

  def _update_pos(self, action_dict):
    for agent_id, action in action_dict.items():
      if action == Action.FORWARD.value:  # Move forward
        dir_x = 0
        dir_y = 1
        self._update_pos_and_penalties(agent_id, dir_x, dir_y)

      elif action == Action.BACKWARD.value:  # Move backward
        dir_x = 0
        dir_y = -1
        self._update_pos_and_penalties(agent_id, dir_x, dir_y)

      elif action == Action.RIGHT.value:  # right
        dir_x = 1
        dir_y = 0
        self._update_pos_and_penalties(agent_id, dir_x, dir_y)

      elif action == Action.LEFT.value:  # left
        dir_x = -1
        dir_y = 0
        self._update_pos_and_penalties(agent_id, dir_x, dir_y)

      else:  # 4 Wait
        self.agent_wait_count[agent_id] += 1

  def _update_pos_and_penalties(self, agent_id, dir_x, dir_y):
    new_pos = (self.agent_pos[agent_id][0] + dir_x, self.agent_pos[agent_id][1] + dir_y)

    if new_pos in self.valid_pos and not self._is_occupied(agent_id, new_pos):
      self.agent_pos[agent_id] = new_pos
      self.agent_info[agent_id]["valid_move"] = True
    else:
      self.agent_reward_penalties[agent_id] -= self.COLLISION_PENALTY
      self.agent_info[agent_id]["valid_move"] = False

  def _is_done(self, id):
    return (self.agent_goal_pos[id] == self.agent_pos[id]) or (self._step_counter == self._ep_step_limit)

  def _is_truncated(self, id):
    return self._step_counter == self._ep_step_limit

  def _get_obs(self):
    observations = {}

    for agent_id in self._agent_ids:
      if self.agent_reported_done[agent_id]:
        continue

      pos = self.agent_pos[agent_id]
      g_pos = self.agent_goal_pos[agent_id]
      observations[agent_id] = np.array([int(pos[0]), int(pos[1]), int(g_pos[0]), int(g_pos[1])],
                                        dtype = np.int8)

    # assert(self.observation_space.contains(observations) or observations == {})
    # logger.info(f"({self.worker_index}) OBS: {observations}")
    # assert (self.observation_space_contains(observations))
    return observations

  def _calc_reward(self, agent_id: str):  # Manhattan distance
    progress_weight = 2
    p = Utils.calc_perf(self.agent_pos[agent_id], self.agent_goal_pos[agent_id], self.agent_starting_pos[agent_id])
    # r = (progress_weight * p) + self.agent_reward_penalties[agent_id] - (
    #     ((self.agent_wait_count[agent_id] - 3) / self._ep_step_limit) ** 2)
    # r = ((progress_weight * p) * (1 - (float(self._step_counter) / float(self._ep_step_limit)))) + \
    #     self.agent_reward_penalties[agent_id]
    r = (progress_weight * p) + self.agent_reward_penalties[agent_id]

    assert (type(r) is float)

    self.agent_reward_penalties[agent_id] = 0.0
    return r

  def _is_occupied(self, agent, new_pos):
    b = False

    for id, pos in zip(self.agent_pos.keys(), self.agent_pos.values()):
      b |= (pos == new_pos) and (agent != id)

    return b

  def _push_meta_info(self):
    # for k, v in self.meta_info.items():
    #   MetaInfo.put(self.worker_index, k, v)
    #   logger.info(f"({self.worker_index}) ce: {k} {v}")
    pass
    # logger.info(f"({self.worker_index}) {datetime.now()} real MetaInfo: {MetaInfo.get_keys(self.worker_index)}")
