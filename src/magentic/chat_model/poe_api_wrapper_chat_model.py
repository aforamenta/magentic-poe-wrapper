from collections.abc import Callable, Iterable
from itertools import chain
from typing import Any, Sequence, TypeVar, cast, overload, List, Dict, Any, get_origin, get_args, Union
from datetime import datetime

import os

import re

from pydantic import ValidationError, BaseModel, create_model
from pydantic_core import PydanticUndefined

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
        if poe_api_wrapper_model is None:
            poe_api_wrapper_model = os.getenv("MAGENTIC_POE_API_WRAPPER_MODEL")
            if poe_api_wrapper_model is None:
                raise ValueError("You must provide a model name for the PoeApiWrapperChatModel either through the constructor or the MAGENTIC_POE_API_WRAPPER_MODEL environment variable.")
        if poe_api_wrapper_token_p_b is None:
            poe_api_wrapper_token_p_b = os.getenv("MAGENTIC_POE_API_WRAPPER_TOKEN_P_B")
            if poe_api_wrapper_token_p_b is None:
                raise ValueError("You must provide a token for the PoeApiWrapperChatModel either through the constructor or the MAGENTIC_POE_API_WRAPPER_TOKEN_P_B environment variable.")
        if poe_api_wrapper_token_p_lat is None:
            poe_api_wrapper_token_p_lat = os.getenv("MAGENTIC_POE_API_WRAPPER_TOKEN_P_LAT")
            if poe_api_wrapper_token_p_lat is None:
                raise ValueError("You must provide a token for the PoeApiWrapperChatModel either through the constructor or the MAGENTIC_POE_API_WRAPPER_TOKEN_P_LAT environment variable.")
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

        # Modify schema dictionaries during serialization to exclude unwanted fields
        # HACK: the following code is a hackish way to send ONLY the parameters of the child class to poe. It shouldn't be needed in production, but we'll see
        schemas_list = []
        for schema in tool_schemas:
            schema_dict = schema.to_dict()

            # Check if 'function' key exists in schema_dict
            if 'function' in schema_dict:
                function_dict = schema_dict['function']
                function_name = function_dict.get('name')
                parameters = function_dict.get('parameters', {})
                properties = parameters.get('properties', {})
                required = parameters.get('required', [])

                unwanted_fields = [
                    'context_hook', 'prompt_hook', 'instance_hook',
                    'chat_code', 'chat_id', 'id'
                ]

                # Remove unwanted fields from 'properties'
                for field in unwanted_fields:
                    properties.pop(field, None)

                # Remove unwanted fields from 'required'
                required = [field for field in required if field not in unwanted_fields]

                # Update the parameters dict
                parameters['properties'] = properties
                parameters['required'] = required

                # Update the function dict with modified parameters
                function_dict['parameters'] = parameters

                # Update 'function' in schema_dict
                schema_dict['function'] = function_dict

            else:
                # Handle schemas without 'function' key
                function_name = schema_dict.get('name')
                parameters = schema_dict.get('parameters', {})
                properties = parameters.get('properties', {})
                required = parameters.get('required', [])

                unwanted_fields = [
                    'context_hook', 'prompt_hook', 'instance_hook',
                    'chat_code', 'chat_id', 'id'
                ]

                # Remove unwanted fields from 'properties'
                for field in unwanted_fields:
                    properties.pop(field, None)

                # Remove unwanted fields from 'required'
                required = [field for field in required if field not in unwanted_fields]

                # Update the parameters dict
                parameters['properties'] = properties
                parameters['required'] = required

                # Update the schema dict with modified parameters
                schema_dict['parameters'] = parameters

            # Append the modified schema_dict to schemas_list
            schemas_list.append(schema_dict)

        combined_message = "\n".join(str(m.content) for m in messages)
        combined_message += "\n\nAvailable tools:\n" + json.dumps(schemas_list)
        if tool_schemas:
            combined_message += (
                f"\n\nPlease use the "
                f"{self._get_tool_choice(tool_schemas=tool_schemas, allow_string_output=allow_string_output)} tool."
                "\n\nYou must ONLY output the json without rich format. Do not output any other text."
            )

        response = self._client.send_message(self._model, combined_message)

        # Collect all chunks
        chunks = list(response)
        
        # Get the chatCode or chatId from the last chunk
        last_chunk = chunks[-1]
        chat_code = last_chunk.get("chatCode")
        chat_id = last_chunk.get("chatId")
        full_response = last_chunk.get("text")

        # Combine all response chunks into a single string
        #full_response = "".join(chunk["response"] for chunk in chunks if chunk.get("response") is not None)

        try:
            # Get the full_response from the first "{" to the last "}"

            def clean_json_string(json_str):
                # Remove trailing commas in arrays
                json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                # Remove newlines outside of quotes (if needed)
                json_str = re.sub(r'\n(?=(?:[^"]*"[^"]*")*[^"]*$)', '', json_str)
                return json_str

            cleaned_json = clean_json_string(full_response) # HACK: remove the newlines thare are NOT part of the json
            full_response_json_limiter = cleaned_json[cleaned_json.find("{"):cleaned_json.rfind("}") + 1] #HACK: limit json
            # Try to parse the response as JSON
            parsed_json = json.loads(full_response_json_limiter)
            
            if output_types and issubclass(next(iter(output_types)), BaseModel):
                output_model = next(iter(output_types))
                
                # Create a dictionary with default empty values for all fields
                default_data = {}
                for field_name, field in output_model.__fields__.items():
                    field_type = field.annotation
                    
                    # Unwrap Union or Optional types, e.g., Optional[int] becomes int
                    origin_type = get_origin(field_type)
                    if origin_type is Union:
                        # Get the first non-None type in the Union (usually for Optional[Type])
                        field_type = next(t for t in get_args(field_type) if t is not type(None))

                    # Handle List and Dict types
                    if get_origin(field_type) == list or field_type == List:
                        default_data[field_name] = list()
                    elif get_origin(field_type) == dict or field_type == Dict:
                        default_data[field_name] = dict()

                    # Handle specific field names with custom logic (these are always strings or None)
                    elif field_name in ['context_hook', 'prompt_hook', 'instance_hook', 'chat_code', 'chat_id']:
                        default_data[field_name] = None
                    elif field_name == 'id':
                        default_data[field_name] = None

                    # Handle integer fields
                    elif field_type == int:
                        default_data[field_name] = None  # or -1 if required

                    # Handle float fields
                    elif field_type == float:
                        default_data[field_name] = None  # or -1.0 if required

                    # Handle string fields
                    elif field_type == str:
                        default_data[field_name] = None

                    # Handle datetime fields
                    elif field_type == datetime:
                        default_data[field_name] = None

                    # Fallback for unsupported types
                    else:
                        default_data[field_name] = None
                
                # Update default data with parsed JSON, keeping defaults for missing keys
                for key, value in parsed_json.items():
                    if key in default_data:
                        if value is not None:
                            default_data[key] = value
                        # If value is None, keep the default empty list or dict
                
                try:
                    validated_content = output_model.model_validate(default_data)
                except ValidationError as e:
                    print(f"Validation error: {e}")
                    # You might want to handle this error more gracefully
                    validated_content = default_data
                
                result = AssistantMessage(validated_content)
            else:
                # If it's not a specific model, return the parsed JSON with default empty values
                default_data = {
                    key: ([] if isinstance(value, list) else {}) if value is None else value
                    for key, value in parsed_json.items()
                }
                result = AssistantMessage(default_data)
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