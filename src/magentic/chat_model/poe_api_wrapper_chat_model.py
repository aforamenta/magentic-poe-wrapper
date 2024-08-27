from collections.abc import Callable, Iterable
from itertools import chain
from typing import Any, Sequence, TypeVar, cast, overload

from pydantic import ValidationError, BaseModel

import json

from magentic.chat_model.base import (
    ChatModel,
    StructuredOutputError,
    avalidate_str_content,
    validate_str_content,
)
from magentic.chat_model.function_schema import (
    FunctionCallFunctionSchema,
    async_function_schema_for_type,
    function_schema_for_type,
)
from magentic.chat_model.message import (
    AssistantMessage,
    Message,
)
from magentic.chat_model.openai_chat_model import (
    STR_OR_FUNCTIONCALL_TYPE,
    AsyncFunctionToolSchema,
    BaseFunctionToolSchema,
    FunctionToolSchema,
    aparse_streamed_tool_calls,
    discard_none_arguments,
    message_to_openai_message,
    parse_streamed_tool_calls,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    ParallelFunctionCall,
)
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    achain,
    async_iter,
)
from magentic.typing import is_any_origin_subclass, is_origin_subclass

try:
    from poe_api_wrapper import PoeApi
except ImportError as error:
    msg = "To use PoeApiWrapperChatModel you must install the `poe-api-wrapper` package."
    raise ImportError(msg) from error

R = TypeVar("R")

class PoeApiWrapperChatModel(ChatModel):
    """An LLM chat model that uses the `poe-api-wrapper` python package."""

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

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        allow_string_output: bool,
    ) -> str | None:
        """Create the tool choice argument."""
        if allow_string_output:
            return None
        if len(tool_schemas) == 1:
            # Instead of accessing .name, let's use the to_dict() method
            # and access the name from there
            tool_dict = tool_schemas[0].to_dict()
            return tool_dict.get('function', {}).get('name')
        return "auto"

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
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        """Request an LLM message and delete the chat immediately after."""
        self._initialize_client()

        if output_types is None:
            output_types = [] if functions else cast(list[type[R]], [str])

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [FunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        streamed_str_in_output_types = is_any_origin_subclass(output_types, StreamedStr)
        allow_string_output = str_in_output_types or streamed_str_in_output_types

        combined_message = "\n".join(str(m.content) for m in messages)
        if tool_schemas:
            combined_message += "\n\nAvailable tools:\n" + json.dumps([schema.to_dict() for schema in tool_schemas])
            combined_message += f"\n\nPlease use the {self._get_tool_choice(tool_schemas=tool_schemas, allow_string_output=allow_string_output)} tool.\n\n\nYou must ONLY output the json. Do not output any other text."

        response = self._client.send_message(self._model, combined_message)

        # Collect all chunks
        chunks = list(response)
        
        # Get the chatCode or chatId from the last chunk
        last_chunk = chunks[-1]
        chat_code = last_chunk.get("chatCode")
        chat_id = last_chunk.get("chatId")

        # Combine all response chunks into a single string
        full_response = "".join(chunk["response"] for chunk in chunks if chunk.get("response") is not None)

        try:
            # Try to parse the response as JSON
            parsed_json = json.loads(full_response)
            
            # If output_types is specified and the first type is a Pydantic model, use it to validate the JSON
            if output_types and issubclass(next(iter(output_types)), BaseModel):
                output_model = next(iter(output_types))
                validated_content = output_model.model_validate(parsed_json)
                result = AssistantMessage(validated_content)
            else:
                # If it's not a specific model, return the parsed JSON
                result = AssistantMessage(parsed_json)
        except json.JSONDecodeError:
            # If it's not valid JSON, process it as a string
            if allow_string_output:
                str_content = validate_str_content(
                    full_response,
                    allow_string_output=allow_string_output,
                    streamed=streamed_str_in_output_types,
                )
                result = AssistantMessage(str_content)
            else:
                msg = (
                    "Failed to parse model output as JSON. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg)

        # Delete the chat TODO: chat deletion is not working for some reason
        #if chat_code:
        #    self._client.delete_chat(self._model, chatCode=chat_code)
        #if chat_id:
        #    self._client.delete_chat(self._model, chatId=chat_id)

        return result

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
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        """Async version of `complete`."""
        # For simplicity, we're using the synchronous version here.
        # In a real implementation, you'd want to make this properly asynchronous.
        return self.complete(messages, functions, output_types, stop=stop)