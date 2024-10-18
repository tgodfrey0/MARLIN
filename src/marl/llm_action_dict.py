import threading
from typing import *


class LLMActionDict:
  """
  Plans are the same when valid_pos, (pos, goal), (pos, goal) are the same
  """

  _instance = None
  _lock = threading.RLock()
  _plans = {}
  _performances = {}

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      with cls._lock:
        # Another thread could have created the instance
        # before we acquired the lock. So check that the
        # instance is still nonexistent.
        if not cls._instance:
          cls._instance = super().__new__(cls, *args, **kwargs)
    return cls._instance

  def __copy__(self):
    return LLMActionDict._instance

  def __deepcopy__(self, memodict = {}):
    return LLMActionDict._instance

  @staticmethod
  def set(valid_pos: Set[Tuple[int, int]], agent_pos_goal: List[Tuple[Tuple[int, int], Tuple[int, int]]],
          plan: List[Dict[str, int]], perf: float):
    with LLMActionDict._lock:
      key = LLMActionDict.gen_key(valid_pos, agent_pos_goal)
      _, prev_perf = LLMActionDict.get(valid_pos, agent_pos_goal)
      if prev_perf <= perf:
        LLMActionDict._plans[key] = plan
        LLMActionDict._performances[key] = perf

  @staticmethod
  def get(valid_pos: Set[Tuple[int, int]], agent_pos_goal: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> \
      Tuple[Optional[List[Dict[str, int]]], float]:
    with LLMActionDict._lock:
      key = LLMActionDict.gen_key(valid_pos, agent_pos_goal)
      if key in LLMActionDict._plans.keys():
        return LLMActionDict._plans[key], LLMActionDict._performances[key]
      else:
        return None, 0.

  @staticmethod
  def remove(valid_pos: Set[Tuple[int, int]], agent_pos_goal: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    with LLMActionDict._lock:
      key = LLMActionDict.gen_key(valid_pos, agent_pos_goal)
      del LLMActionDict._plans[key]
      del LLMActionDict._performances[key]

  @staticmethod
  def gen_key(valid_pos: Set[Tuple[int, int]], agent_pos_goal: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    assert type(valid_pos) == set  # This affects keying
    agent_pos_goal.sort()
    tuples = [num for tup in valid_pos for num in tup] + [num for sublist in agent_pos_goal for tuple_ in sublist for
                                                          num in tuple_]
    key_s = ""
    for i in tuples:
      key_s += str(i)
    return int(key_s)

  @staticmethod
  def key_exists(valid_pos: Set[Tuple[int, int]], agent_pos_goal: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    with LLMActionDict._lock:
      return LLMActionDict.gen_key(valid_pos, agent_pos_goal) in LLMActionDict._plans.keys()

  @staticmethod
  def update_perf(key, new_perf: float):
    with LLMActionDict._lock:
      LLMActionDict._performances[key] = new_perf
