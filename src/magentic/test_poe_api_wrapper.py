import importlib
import os
import yaml
from typing import List, Callable, Type, Dict, Union
from magentic import chatprompt, SystemMessage, UserMessage, AssistantMessage, prompt
from pydantic import BaseModel
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from magentic.chat_model.poe_api_wrapper_chat_model import PoeApiWrapperChatModel

# Define Message as a Union of message types
Message = Union[SystemMessage, UserMessage, AssistantMessage]

poe_p_lat = ""
poe_p_b = ""

############# LAZY TESTING CODE ####################
@prompt("one two three", model=PoeApiWrapperChatModel(poe_api_wrapper_model="llama31405bt", poe_api_wrapper_token_p_b=poe_p_b,poe_api_wrapper_token_p_lat=poe_p_lat))
def TEST_MAGENTIC_SIMPLE_PROMPT(diary_text: str)->str: ...
the_test = TEST_MAGENTIC_SIMPLE_PROMPT("lmao ca vai")
print(the_test)
############# END OF LAZY TESTING CODE ####################
