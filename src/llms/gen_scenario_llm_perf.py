import os
from random import choices, randint
from threading import Thread
import concurrent.futures
from src.llms.llm_move_gen import LLMMoveGen
from src.utils.scenarios import Scenario, get_all_instantiated_scenarios_below_difficulty, MazeLikeCorridor
from src.utils.utils import Utils

model = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def test_scenario(scenario: Scenario, run):
  Utils.set_csv_file("llm", os.path.abspath("./llm_data/"),
                     f"LLM_{model.split('/')[1]}_data.csv")
  gen = LLMMoveGen(
      scenario,
      ["alice", "bob"],
      scenario.valid_pos,
      {"alice": (1, 0), "bob": (1, 7)},
      {"alice": (1, 7), "bob": (1, 0)},
      model,
      0,
      path_name = "llm",
      write_conversation = True,
      write_csv = True
  )

  moves, perf = gen.gen_moves(50, verbose = True)

  print(f"Scenario {scenario.name}")
  print(f"Run: {run}")
  print(f"Moves: {moves}")
  print(f"Perf: {perf}")


def test(scenario, runs):
  for i in range(runs):
    test_scenario(scenario, i)


if __name__ == "__main__":
  # ss = get_all_instantiated_scenarios_below_difficulty(0)
  ss = [MazeLikeCorridor()]
  with concurrent.futures.ThreadPoolExecutor(max_workers = 12) as executor:
    futures = [executor.submit(test_scenario, s, n) for s in ss for n in range(5)]

    # Wait for all futures to complete
    for future in concurrent.futures.as_completed(futures):
      future.result()
