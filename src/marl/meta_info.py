from copy import deepcopy
from threading import Lock
from typing import *


# ? How to split this with parallel environments?
class MetaInfo:
  _mx = Lock()
  _info: Dict[int, Dict[str, Any]] = {}
  _instance = None

  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      cls._instance = super(MetaInfo, cls).__new__(cls, *args, **kwargs)
    return cls._instance

  @classmethod
  def get(cls, worker_index, key: str) -> Any:
    with cls._mx:
      if worker_index not in cls._info.keys():
        cls._info[worker_index] = {}
      return cls._info.get(worker_index).get(key, None)

  @classmethod
  def put(cls, worker_index, key: str, value: Any, no_copy: bool = False) -> None:
    with cls._mx:
      if worker_index not in cls._info.keys():
        cls._info[worker_index] = {}
      if no_copy:
        cls._info[worker_index][key] = value
      else:
        cls._info[worker_index][key] = deepcopy(value)

  @classmethod
  def get_keys(cls, worker_index: int):
    with cls._mx:
      print(cls._info[worker_index])
      return list(cls._info[worker_index].keys())
