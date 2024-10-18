from enum import Enum


class Action(Enum):
  WAIT = 0
  FORWARD = 1
  BACKWARD = 2
  LEFT = 3
  RIGHT = 4


def action_to_string(action: Action) -> str:
  return "@" + action.name.upper()


def action_from_string(string: str) -> Action:
  a = Action.WAIT
  if string == "@FORWARD":
    a = Action.FORWARD
  elif string == "@BACKWARD":
    a = Action.BACKWARD
  elif string == "@LEFT":
    a = Action.LEFT
  elif string == "@RIGHT":
    a = Action.RIGHT

  return a
