import csv
import os
from statistics import mean
from threading import Lock
from typing import Tuple, List, Any


class Utils:
  csv_paths = {}
  csv_lock = Lock()

  @staticmethod
  def calc_perf(loc: Tuple[int, int], goal: Tuple[int, int], init_loc: Tuple[int, int]) -> float:
    # p = 1-(d/D)
    d_end = float(Utils.manhattan_distance(loc, goal))
    d_start = float(Utils.manhattan_distance(init_loc, goal))

    if d_start == 0 or d_end == 0:
      return 1

    p = 1. - (d_end / d_start)
    # print(f"loc {loc}\ngoal {goal}\ninitial_loc {init_loc}\np {p}")
    assert (1 >= p)
    return max(p, 0)

  @staticmethod
  def calc_multiagent_avg_perf(agent_locs: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]) -> float:
    """
    Calculate average performance of multiple agents.

    Args:
      - agent_locs [(pos, goal, start)]
    """
    return mean(map(lambda t: Utils.calc_perf(*t), agent_locs))

  @staticmethod
  def manhattan_distance(loc_0: Tuple[int, int], loc_1: Tuple[int, int]) -> int:
    x_0 = loc_0[0]
    y_0 = loc_0[1]
    x_1 = loc_1[0]
    y_1 = loc_1[1]

    return abs(x_0 - x_1) + abs(y_0 - y_1)

  @staticmethod
  def write_csv(path_name, header: List[str], vals: List[Any]):
    with Utils.csv_lock:
      if path_name not in Utils.csv_paths.keys():
        raise ValueError("CSV file path not set. Use Utils.set_csv_file(name, dir, filename) first.")

      try:
        with open(Utils.csv_paths[path_name],
                  'x' if not (exists := os.path.exists(Utils.csv_paths[path_name])) else 'a',
                  newline = '') as csvfile:
          writer = csv.writer(csvfile)
          if not exists:  # Write header only if file is new
            writer.writerow(header)
          writer.writerow(list(map(str, vals)))

      except FileNotFoundError:
        print(f"Error: CSV file '{Utils.csv_paths[path_name]}' not found.")

      except Exception as e:
        print(f"An error occurred while writing to CSV: {e}")

  @staticmethod
  def set_csv_file(name: str, directory: str, filename: str):
    if not os.path.exists(directory):
      os.makedirs(directory)

    filename = filename.replace("/", "-")

    Utils.csv_paths[name] = os.path.join(directory, filename)
