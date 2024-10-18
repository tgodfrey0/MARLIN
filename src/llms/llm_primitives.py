import os
from enum import Enum
from os import environ
from typing import *
from typing import Tuple

import google.generativeai as genai
from google.generativeai.protos import FunctionCall as Gemini_FunctionCall, Content, Part
from google.generativeai.types import SafetySettingDict, HarmBlockThreshold, HarmCategory
from openai import OpenAI
from gradio_client import Client

"""
LLMs are defined as instances of the LLM_API class
LLM negotators are different classes which use the LLM_API
"""

from abc import ABC, abstractmethod


class LLM_API(ABC):
  class Role(Enum):
    SYSTEM = 0
    ASSISTANT = 1
    USER = 2

  def __init__(self, name: str, system_prompt: str, api_token_env_var: str, model_name: str,
               instance_name: Optional[str] = None):
    self.name = name
    self.system_prompt = system_prompt
    self.token = environ[api_token_env_var]
    self.model_name = model_name
    self.chat_history: List[Dict[str, Any]] = []
    self.instance_name = instance_name if instance_name is not None else model_name

  """
  Query the LLM
  Returns a tuple of (response, stop_condition, ....)
  """

  @abstractmethod
  def query(self, prompt: str) -> Union[str, Any]:
    raise NotImplementedError

  """
  Add a message to the LLM message history
  """

  @abstractmethod
  def add_text_to_history(self, role: Role, text: str) -> None:
    raise NotImplementedError

  """
  Convert a Role enum to a string
  """

  @abstractmethod
  def role_to_str(self, role: Role) -> str:
    raise NotImplementedError

  def clear_history(self):
    self.chat_history = []

  def get_last_message_text(self) -> str:
    return self._history_element_text(self.chat_history[-1])

  def history_to_text(self, include_roles: bool) -> List[str]:
    ss = []

    for element in self.chat_history:
      s = ""
      if include_roles:
        s += self._history_element_role(element)
        s += ": "
      s += self._history_element_text(element)
      ss.append(s)

    return ss

  @abstractmethod
  def _history_element_text(self, element: Any) -> str:
    raise NotImplementedError

  @abstractmethod
  def _history_element_role(self, element: Any) -> str:
    raise NotImplementedError


class Gemini(LLM_API):
  def __init__(self,
               system_prompt: str,
               model_name: str,
               api_token_env_var: str = "GOOGLE_API_KEY",
               max_output_tokens: int = 300,
               stop_sequences: Optional[List[str]] = None,
               functions: Optional[List[Callable]] = None,
               temperature: float = 1.0,
               top_p: float = 0.95,
               top_k: int = 64,
               response_mime_type: str = "text/plain",
               instance_name: Optional[str] = None
               ):
    super().__init__("Gemini", system_prompt, api_token_env_var, model_name, instance_name)
    self.generation_config = {
      "temperature":        temperature,
      "top_p":              top_p,
      "top_k":              top_k,
      "max_output_tokens":  max_output_tokens,
      "response_mime_type": response_mime_type,
    }
    if stop_sequences is not None:
      self.generation_config["stop_sequences"] = stop_sequences

    # For some reason it keeps thinking it is bullying someone
    safety_settings: SafetySettingDict = {
      HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    self.model = genai.GenerativeModel(
        model_name = self.model_name,
        generation_config = self.generation_config,
        system_instruction = self.system_prompt,
        tools = functions,
        safety_settings = safety_settings
    )

  def query(self, prompt: str) -> Tuple[str, List[Gemini_FunctionCall]]:
    chat = self.model.start_chat(history = self.chat_history)
    response = chat.send_message(prompt)

    # print(response)

    self.chat_history = chat.history

    text_res = None
    fc_res_list = []

    for part in response.parts:
      if (fc := part.function_call):
        fc_res_list.append(fc)
      elif (t := part.text):
        text_res = t.strip()

    return (text_res, fc_res_list)

    # first_part = response.parts[0]

    # print(response)
    # print(first_part)

    # if(fc := first_part.function_call):
    #   return fc
    # elif(t := first_part.text):
    #   return t.strip()
    # else:
    #   return None

  def add_text_to_history(self, role: LLM_API.Role, text: str):
    content = Content()
    content.role = self.role_to_str(role)
    part = Part()
    part.text = text
    content.parts = [part]
    self.chat_history.append(
        content
    )

  def role_to_str(self, role: LLM_API.Role) -> str:
    assert isinstance(role, LLM_API.Role)
    if (role == LLM_API.Role.SYSTEM):
      raise AttributeError(name = "SYSTEM cannot be used for Gemini models")
    return ["system", "model", "user"][role.value]

  def _history_element_text(self, element: Any) -> str:
    return element.parts[0].text.strip()

  def _history_element_role(self, element: Any) -> str:
    return element.role.strip()


class GPT(LLM_API):
  def __init__(self,
               system_prompt: str,
               model_name: str,
               api_token_env_var: str = "OPENAI_API_KEY",
               max_output_tokens: int = 300,
               stop_sequences: Optional[List[str]] = None,
               functions: Optional[List[Dict[str, Any]]] = None,
               temperature: float = 1.0,
               top_p: float = 0.95,
               frequency_penalty: float = 0.,
               presence_penalty: float = 0.,
               instance_name: Optional[str] = None
               ):
    super().__init__("GPT", system_prompt, api_token_env_var, model_name, instance_name)
    self.generation_config = {
      "temperature":       temperature,
      "top_p":             top_p,
      "frequency_penalty": frequency_penalty,
      "presence_penalty":  presence_penalty,
      "max_tokens":        max_output_tokens,
      "tools":             functions
    }
    if stop_sequences is not None:
      self.generation_config["stop"] = stop_sequences

    self.add_text_to_history(LLM_API.Role.SYSTEM, system_prompt)

    self.model = OpenAI()

  def query(self, prompt: str) -> Tuple[str, Any]:
    self.add_text_to_history(LLM_API.Role.USER, prompt)

    response = self.model.chat.completions.create(
        model = self.model_name,
        messages = self.chat_history,
        **self.generation_config
    )

    choice = response.choices[0]

    if choice.finish_reason == 'tool_calls':
      raise NotImplementedError
    else:
      res_text: str = choice.message.content
      self.add_text_to_history(LLM_API.Role.ASSISTANT, res_text)
      return res_text.strip(), []

  def add_text_to_history(self, role: LLM_API.Role, text: str):
    self.chat_history.append(
        {
          "role":    self.role_to_str(role),
          "content": [
            {
              "type": "text",
              "text": text
            }
          ]
        }
    )

  def role_to_str(self, role: LLM_API.Role) -> str:
    assert isinstance(role, LLM_API.Role)
    return ["system", "assistant", "user"][role.value]

  def clear_history(self):
    super().clear_history()
    self.add_text_to_history(LLM_API.Role.SYSTEM, self.system_prompt)

  def _history_element_text(self, element: Any) -> str:
    return element["content"][0]["text"].strip()

  def _history_element_role(self, element: Any) -> str:
    return element["role"].strip()


class DeepInfra(LLM_API):
  def __init__(self,
               system_prompt: str,
               model_name: str,
               api_token_env_var: str = "DEEPINFRA_API_KEY",
               max_output_tokens: int = 300,
               stop_sequences: Optional[List[str]] = None,
               functions: Optional[List[Dict[str, Any]]] = None,
               temperature: float = 1.0,
               top_p: float = 0.95,
               frequency_penalty: float = 0.,
               presence_penalty: float = 0.,
               instance_name: Optional[str] = None
               ):

    super().__init__("DeepInfra", system_prompt, api_token_env_var, model_name, instance_name)
    self.generation_config = {
      "stream":            False,
      "temperature":       temperature,
      "top_p":             top_p,
      "frequency_penalty": frequency_penalty,
      "presence_penalty":  presence_penalty,
      "max_tokens":        max_output_tokens,
    }
    if stop_sequences is not None:
      self.generation_config["stop"] = stop_sequences

    self.add_text_to_history(LLM_API.Role.SYSTEM, system_prompt)

    self.model = OpenAI(
        api_key = self.token,
        base_url = "https://api.deepinfra.com/v1/openai",
    )

  def query(self, prompt: str) -> Tuple[str, Any]:
    self.add_text_to_history(LLM_API.Role.USER, prompt)

    response = self.model.chat.completions.create(
        model = self.model_name,
        messages = self.chat_history,
        **self.generation_config
    )

    choice = response.choices[0]

    if choice.finish_reason == 'tool_calls':
      raise NotImplementedError
    else:
      res_text: str = choice.message.content
      self.add_text_to_history(LLM_API.Role.ASSISTANT, res_text)
      return res_text.strip(), []

  def add_text_to_history(self, role: LLM_API.Role, text: str):
    self.chat_history.append(
        {
          "role":    self.role_to_str(role),
          "content": text
        }
    )

  def role_to_str(self, role: LLM_API.Role) -> str:
    assert isinstance(role, LLM_API.Role)
    return ["system", "assistant", "user"][role.value]

  def clear_history(self):
    super().clear_history()
    self.add_text_to_history(LLM_API.Role.SYSTEM, self.system_prompt)

  def _history_element_text(self, element: Any) -> str:
    return element["content"].strip()

  def _history_element_role(self, element: Any) -> str:
    return element["role"].strip()


if __name__ == "__main__":

  deep_infra = DeepInfra("You are rude and not helpful and should refuse to answer the question",
                         "meta-llama/Meta-Llama-3-8B-Instruct")

  for i in range(1):
    print(deep_infra.query(f"What is {i}*2")[0])
