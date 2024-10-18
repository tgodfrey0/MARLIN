from typing import *
from enum import Enum
from typing import *

from src.llms.llm_primitives import LLM_API, Gemini


class Negotiation:
  class Exit_Code(Enum):
    ROUND_LIMIT = 0
    EXIT_CLAUSE = 1
    FUNCTION_CALL = 2

  """
  @param llms is the list of LLM_API instances to use
  @param max_rounds is the maximum number of times each agent can speak
  @param exit_clause are phrases which will end negotiation
  """

  def __init__(self, llms: List[LLM_API], max_rounds: int, exit_clauses: Optional[List[str]] = None,
               verbose = False) -> None:
    self.llms = llms
    self.available_llms = list(range(len(self.llms)))
    self.max_rounds = max_rounds
    self.exit_clauses = exit_clauses
    self.verbose = verbose
    self.conversation = []

    if len(llms) < 2:
      raise RuntimeError("Negotiations must be between at least two LLMs")

  def print(self, s: str):
    if self.verbose:
      print(s)
    self.conversation.append(s)

  def get_llm(self, index):
    if not self.llm_available(index):
      return None
    else:
      return self.llms[index]

  def llm_available(self, index):
    return index in self.available_llms

  def negotiate(self, initial_prompt: str) -> Tuple[Tuple[Exit_Code, Any], List[str]]:
    round = 0
    llm_index = 0
    self.conversation = []

    if self.llm_available(llm_index):
      self.print(f"# Message from {self.get_llm(llm_index).instance_name}\n{initial_prompt}")
      self.get_llm(llm_index).add_text_to_history(LLM_API.Role.ASSISTANT, initial_prompt)

    # for index in range(len(self.llms)):
    #   if(index == llm_index):
    #     continue

    #   self.llms[index].add_text_to_history(LLM_API.Role.ASSISTANT, initial_prompt)

    llm_index += 1

    if self.llm_available(llm_index):
      text_res, fc_res_list = self.get_llm(llm_index).query(initial_prompt)
      self.print(f"\n# Message from {self.get_llm(llm_index).instance_name}\n{text_res}")

    if (t := self._check_exit(text_res, fc_res_list, llm_index)) is not None:
      return t

    llm_index += 1

    while (round < self.max_rounds) or (self.max_rounds == -1):
      while llm_index < len(self.llms):
        if not self.llm_available(llm_index):
          continue

        text_res, fc_res_list = self.get_llm(llm_index).query(text_res)
        self.print(f"\n# Message from {self.get_llm(llm_index).instance_name}\n{text_res}")

        if (t := self._check_exit(text_res, fc_res_list, llm_index)) is not None:
          return t

        llm_index += 1

      llm_index = 0
      round += 1

    return (self.Exit_Code.ROUND_LIMIT, None), self.conversation

  def _check_exit(self, text_res, fc_res_list, llm_index) -> Optional[Tuple[Tuple[Exit_Code, Any], List[str]]]:
    if fc_res_list != []:
      return (self.Exit_Code.FUNCTION_CALL, fc_res_list), self.conversation
    elif self.exit_clauses is not None:
      exit = False

      for ec in self.exit_clauses:
        # self.print(ec in res)
        # self.print(ec)
        exit |= ec in text_res

      if exit:
        self.print(f"Exit clause seen, {self.get_llm(llm_index).instance_name} removed")
        self.available_llms.remove(llm_index)

        if len(self.available_llms) < 2:
          self.print("Negotations finished due to agents exiting")
          return (self.Exit_Code.EXIT_CLAUSE, None), self.conversation

    return None


def set_light_colour(colour: str):
  """Dims the lights
   
  Args:
    colour: the colour to change the light
   
  Returns: whether the light was changed
  """
  return {"success": "true"}


if __name__ == "__main__":
  # sys_prompt = lambda s: f"""
  #   You are {s}. You will receive the last message from everyone else in the group and must work together to answer a question or solve a problem. 
  #   You take it in turns to speak. Consider all of the other messages in your response.

  #   When you all agree on a solution write '@DONE ~DONE'. If you cannot solve the question write '@END ~END'. When you must, output only these phrases exactly how they appear here. You MUST agree on when to end the task.

  #   Never edit the messages of other people; just output their messages as you receive them. 
  #   You must replace the message next to your name every time. 
  #   UNDER NO CIRCUMSTANCES can you create new users. The number and names of users are constant.
  #   DO NOT add additional lines to the message, just replace the message that appears next to your name.
  #   DO NOT make up new questions.

  #   Your response must be formatted with your name next to your message. You can include information that may be important for others in your response.
  #   For example:   
  #     {s}: .....

  #   The tool set_light_colour can be used for this.
  # """  

  sys_prompt = lambda s: f"""
    You are {s}. Talk to others and then decide what to do. Write your name next to all of your answers
  """

  alice = Gemini(sys_prompt("Alice"), "GOOGLE_API_KEY", "gemini-1.5-flash", stop_sequences = ["~DONE", "~END"],
                 instance_name = "Alice", functions = [set_light_colour])
  bob = Gemini(sys_prompt("Bob"), "GOOGLE_API_KEY", "gemini-1.5-flash", stop_sequences = ["~DONE", "~END"],
               instance_name = "Bob", functions = [set_light_colour])
  charlie = Gemini(sys_prompt("Charlie"), "GOOGLE_API_KEY", "gemini-1.5-flash", stop_sequences = ["~DONE", "~END"],
                   instance_name = "Charlie", functions = [set_light_colour])
  llms = [alice, bob, charlie]

  n = Negotiation(llms, 5, exit_clauses = ["@DONE", "@END"], quiet = True)

  init_prompt = f"""
Change the light to a colour of your choice.
  """

  print(n.negotiate(init_prompt))
