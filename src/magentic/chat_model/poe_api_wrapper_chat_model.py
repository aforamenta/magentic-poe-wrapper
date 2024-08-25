from collections.abc import Iterable
from typing import Any, TypeVar, cast, overload

from poe_api_wrapper import Poe

from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import AssistantMessage, Message

R = TypeVar("R")

class PoeApiWrapperChatModel(ChatModel):
    """A chat model that uses the Poe API wrapper."""

    def __init__(
        self,
        poe_api_wrapper_model: str = "claude_3_igloo",
        poe_api_wrapper_token_p_b: str | None = None,
        poe_api_wrapper_token_p_lat: str | None = None,
    ):
        self._model = poe_api_wrapper_model
        self._token_p_b = poe_api_wrapper_token_p_b
        self._token_p_lat = poe_api_wrapper_token_p_lat
        self._client = None

    def _initialize_client(self):
        if not self._client:
            tokens = {
                'p-b': self._token_p_b,
                'p-lat': self._token_p_lat,
            }
            self._client = PoeApi(tokens=tokens)

    @property
    def model(self) -> str:
        return self._model

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]: ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]: ...

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        self._initialize_client()
        combined_message = "\n".join(str(m.content) for m in messages)
        response = ""
        for chunk in self._client.send_message(self._model, combined_message):
            response += chunk["response"]
        return AssistantMessage(response)

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]: ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]: ...

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        self._initialize_client()
        combined_message = "\n".join(str(m.content) for m in messages)
        response = ""
        for chunk in self._client.send_message(self._model, combined_message):
            response += chunk["response"]
        return AssistantMessage(response)