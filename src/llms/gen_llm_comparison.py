import os
from random import choices, randint

from src.llms.llm_move_gen import LLMMoveGen
from src.utils.utils import Utils

# valid_pos = [(2, 4), (0, 3), (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0)]
main_corridor_pos = [(1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0)]

Utils.set_csv_file("llm", os.path.abspath("./llm_data/"),
                   f"LLM_comparison.csv")

gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
gpt_models = ["gpt-4o", "gpt-4o-mini"]


def test_llms(models):
  for model in models:
    for i in range(3):

      valid_pos = main_corridor_pos + [(2, randint(1, 6)), (0, randint(1, 6))]

      agent_final_pos = {"alice": (0, 0), "bob": (0, 0)}

      agent_starting_pos = {"alice": (0, 0), "bob": (0, 0)}
      agent_starting_dir = {"alice": 0, "bob": 0}
      # alice_starting_dir = 0  # random.randint(0, CorridorEnv.MAX_HEADING)
      # bob_starting_dir = 2  # random.randint(0, CorridorEnv.MAX_HEADING)

      while (
          (agent_starting_pos["alice"] == agent_starting_pos["bob"]) or
          (agent_starting_pos["alice"] == agent_final_pos["alice"]) or
          (agent_starting_pos["bob"] == agent_final_pos["bob"]) or
          (agent_final_pos["alice"] == agent_final_pos["bob"])):
        agent_starting_pos["alice"], agent_starting_pos["bob"], agent_final_pos["alice"], \
          agent_final_pos["bob"] = choices(valid_pos, k = 4)

      gen = LLMMoveGen(["alice", "bob"], valid_pos, agent_starting_pos, agent_final_pos, model, write_csv = True)

      print(f"\n\n{model}: {i}\n")

      try:
        gen.gen_moves(50)
      except KeyboardInterrupt as e:
        raise e
      except Exception as e:
        print(e)


if __name__ == "__main__":
  # gemini_thread = Thread(target = test_llms, args = [gemini_models])
  # gpt_thread = Thread(target = test_llms, args = [gpt_models])
  #
  # gemini_thread.start()
  # gemini_thread.join()
  #
  # gpt_thread.start()
  # gpt_thread.join()
  test_llms(["gpt-4o-mini"])
