from abc import ABC, abstractmethod
from random import choice, choices
from typing import List, Type

from src.utils.grid import Grid


class Scenario(ABC):
  def __init__(self, name: str, llm_prompt: str):
    self.name = name.replace(" ", "_")
    self.llm_prompt = llm_prompt
    self.difficulty = 0
    self.valid_pos = self._gen_pos()

  def regen_positions(self):
    self.valid_pos = self._gen_pos()

  @abstractmethod
  def _gen_pos(self):
    raise NotImplementedError()


class AsymmetricalTwoSlotCorridor(Scenario):
  def __init__(self):
    super().__init__(
        "Asymmetrical_Two_Slot_Corridor",
        "The exclaves at (0, 3) and (2, 4) provide areas for agents to move out of the main corridor and wait for the other agent to pass, this prevents the agents from colliding in the main corridor.")

  def _gen_pos(self):
    positions = [(1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (2, 4)]

    return positions


class SymmetricalTwoSlotCorridor(Scenario):
  def __init__(self):
    super().__init__(
        "Symmetrical_Two_Slot_Corridor",
        "The exclaves at (0, 3) and (2, 3) provide areas for agents to move out of the main corridor and wait for the other agent to pass, this prevents the agents from colliding in the main corridor."
    )

  def _gen_pos(self):
    positions = [(1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0), (0, 3), (2, 3)]

    return positions


class SingleSlotCorridor(Scenario):
  def __init__(self):
    super().__init__(
        "Single_Slot_Corridor",
        "The exclave at (0, 3) provides an area for an agent to move out of the main corridor and wait for the other agent to pass, this prevents the agents from colliding in the main corridor."
    )

  def _gen_pos(self):
    positions = [(1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0), (0, 3)]

    return positions


class TwoPathCorridor(Scenario):
  def __init__(self):
    super().__init__(
        "Two_Path_Corridor",
        "There are two paths, one is in column 0 (from y=1 to y=6) and the other is in column 2 (from y=1 to y=6), and they provide different routes for agents to take so that they can pass without colliding, before rejoining the main corridor. One agent should use the path in column 0 and the other should use the path in column 2."
    )

  def _gen_pos(self):
    positions = [(1, 7), (1, 0)]
    positions = positions + [(0, y) for y in range(1, 7)] + [(2, y) for y in range(1, 7)] + [(1, 1), (1, 6)]

    return positions


# class FigureOfEightCorridor(Scenario):
#   def __init__(self):
#     super().__init__(
#         name = "Figure_Of_Eight_Corridor",
#         llm_prompt = "There are two parallel corridors, one is in column 0 and the other is in column 2 (both from y=0 to y=7), and they provide different routes for agents to take so that they can pass without colliding. The two corridors are joined at both ends (y=0 and y=7) and have an additional join in the centre at (1, 3). One agent should use the path in column 0 and the other should use the path in column 2."
#     )
#
#   def _gen_pos(self):
#     positions = [(x, y) for y in range(0, 8) for x in [0, 2]] + [(1, 0), (1, 7), (1, 3)]
#
#     return positions


class MazeLikeCorridor(Scenario):
  def __init__(self):
    super().__init__(
        "Maze_Like_Corridor",
        "There are several obstructions in the environment. The main path is in the x=2 two column but this only has space for one agent at a time. Obstructions are shown as blank areas on the grid above. Any other free space from the main route can be used to move out the way of the other agent and let them use the only clear path through the space. One agent will have to use the area at (1, 2) to temporarily move out of the way of the other agent then you can move back into the x=2 column once the other agent has moved past you. If you do not move into this area, you will collide at some point and neither of you will be able to complete your tasks."
    )
    self.difficulty = 1

  def _gen_pos(self):
    positions = [(1, 7), (1, 0), (2, 0), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (2, 3), (0, 4), (2, 4), (2, 5), (2, 6),
                 (1, 6), (1, 7)]

    return positions


# class TriangularObstructionCorridor(Scenario):
#   def __init__(self):
#     super().__init__(
#         "Triangular_Obstruction_Corridor",
#         "There is a triangular shaped obstruction in columns 0 and 1. The route around it is in column 2. The obstruction is from y=2 to y=5 in the x=0 column, and from y=3 to y=4 in the x=1 column. Any other free space from the main route can be used to move out the way of the other agent and let them use the only clear path through the space."
#     )
#     self.difficulty = 1
#
#   def _gen_pos(self):
#     positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (2, 3), (2, 4), (1, 5), (2, 5), (0, 6),
#                  (1, 6), (2, 6), (0, 7), (1, 7), (2, 7)]
#
#     return positions


def _get_all_scenarios() -> List[Type[Scenario]]:
  return Scenario.__subclasses__()


def get_all_instantiated_scenarios() -> List[Scenario]:
  ts = _get_all_scenarios()
  scs = []
  for t in ts:
    scs.append(t())
  return scs


def get_all_instantiated_scenarios_below_difficulty(max_difficulty: int) -> List[Scenario]:
  clss = get_all_instantiated_scenarios()
  new_clss = []
  for cls in clss:
    if cls.difficulty <= max_difficulty:
      new_clss.append(cls)
  return new_clss


def render_all():
  for cls in get_all_instantiated_scenarios():
    print(cls.name)
    print(cls.llm_prompt)
    print("\n".join(Grid.render_grid(Grid.unflatten_grid(cls.valid_pos, 3, 8))))
    print("\n\n")


if __name__ == "__main__":
  render_all()
  print(get_all_instantiated_scenarios_below_difficulty(0))
  for _ in range(10):
    print(choices(get_all_instantiated_scenarios_below_difficulty(0), k = 1)[0])
