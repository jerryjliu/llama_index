from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)
import typing

from google.genai import types
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ImageBlock,
    TextBlock,
)
from llama_index.core.utilities.gemini_utils import (
    ROLES_FROM_GEMINI,
    ROLES_TO_GEMINI,
    merge_neighboring_same_role_messages,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


def _error_if_finished_early(candidate: types.Candidate) -> None:
    if finish_reason := candidate.finish_reason:
        if finish_reason != types.FinishReason.STOP:
            reason = finish_reason.name

            # Safety reasons have more detail, so include that if we can.
            if finish_reason == types.FinishReason.SAFETY and candidate.safety_ratings:
                relevant_safety = list(
                    filter(
                        lambda sr: sr.probability
                        and sr.probability.value
                        > types.HarmProbability.NEGLIGIBLE.value,
                        candidate.safety_ratings,
                    )
                )
                reason += f" {relevant_safety}"

            raise RuntimeError(f"Response was terminated early: {reason}")


def completion_from_gemini_response(
    response: types.GenerateContentResponse,
) -> CompletionResponse:
    if not response.candidates:
        raise ValueError("Response has no candidates")

    top_candidate = response.candidates[0]
    _error_if_finished_early(top_candidate)

    raw = {
        **(top_candidate.model_dump()),
        **(response.prompt_feedback.model_dump() if response.prompt_feedback else {}),
    }

    if response.usage_metadata:
        raw["usage_metadata"] = response.usage_metadata.model_dump()
    return CompletionResponse(text=response.text or "", raw=raw)


def chat_from_gemini_response(
    response: types.GenerateContentResponse,
) -> ChatResponse:
    if not response.candidates:
        raise ValueError("Response has no candidates")

    top_candidate = response.candidates[0]
    _error_if_finished_early(top_candidate)

    response_feedback = (
        response.prompt_feedback.model_dump() if response.prompt_feedback else {}
    )
    raw = {
        **(top_candidate.model_dump()),
        **response_feedback,
    }
    if response.usage_metadata:
        raw["usage_metadata"] = response.usage_metadata.model_dump()

    try:
        text = response.text
    except ValueError:
        text = None

    additional_kwargs: Dict[str, Any] = {}

    if response.function_calls:
        for fn in response.function_calls:
            if "tool_calls" not in additional_kwargs:
                additional_kwargs["tool_calls"] = []
            additional_kwargs["tool_calls"].append(fn)

    role = ROLES_FROM_GEMINI[top_candidate.content.role]
    return ChatResponse(
        message=ChatMessage(
            role=role, content=text, additional_kwargs=additional_kwargs
        ),
        raw=raw,
        additional_kwargs=additional_kwargs,
    )


def chat_message_to_gemini(message: ChatMessage) -> types.Content:
    """Convert ChatMessages to Gemini-specific history, including ImageDocuments."""
    parts = []
    for block in message.blocks:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append(types.Part.from_text(text=block.text))
        elif isinstance(block, ImageBlock):
            base64_bytes = block.resolve_image(as_base64=False).read()
            if not block.image_mimetype:
                # TODO: fail ?
                block.image_mimetype = "image/png"

            parts.append(
                types.Part.from_bytes(
                    data=base64_bytes,
                    mime_type=block.image_mimetype,
                )
            )

        else:
            msg = f"Unsupported content block type: {type(block).__name__}"
            raise ValueError(msg)

    for tool_call in message.additional_kwargs.get("tool_calls", []):
        parts.append(tool_call)

    return types.Content(
        role=ROLES_TO_GEMINI[message.role],
        parts=parts,
    )


def convert_schema_to_function_declaration(tool: "BaseTool"):
    """
    Converts a tool's JSON schema into a function declaration.
    Handles $ref resolution and nested property structures.
    """

    def resolve_ref(schema_dict, ref_path):
        """
        Resolves a $ref in the schema by navigating the reference path.
        """
        if not ref_path.startswith("#/$defs/"):
            raise ValueError(f"Unsupported reference format: {ref_path}")

        ref_parts = ref_path[8:].split("/")  # Remove '#/$defs/' and split
        current = schema_dict["$defs"]
        for part in ref_parts:
            current = current[part]
        return current

    def process_property(prop_schema, schema_dict):
        """
        Processes a property schema, handling references and nested structures.
        Returns a Schema object.
        """
        if "$ref" in prop_schema:
            resolved_schema = resolve_ref(schema_dict, prop_schema["$ref"])
            return process_property(resolved_schema, schema_dict)

        prop_type = prop_schema["type"].upper()
        description = prop_schema.get("description", "")
        schema_args = {
            "type": getattr(types.Type, prop_type),
            "description": description,
        }

        if prop_type == "OBJECT":
            if "properties" in prop_schema:
                schema_args["properties"] = {
                    name: process_property(nested_prop, schema_dict)
                    for name, nested_prop in prop_schema["properties"].items()
                }
                schema_args["required"] = prop_schema.get("required", [])

        elif prop_type == "ARRAY" and "items" in prop_schema:
            schema_args["items"] = process_property(prop_schema["items"], schema_dict)

        return types.Schema(**schema_args)

    if not tool.metadata.fn_schema:
        raise ValueError("fn_schema is missing")

    # Get the JSON schema
    json_schema = tool.metadata.fn_schema.model_json_schema()

    # Create the root schema
    root_schema = types.Schema(
        type=types.Type.OBJECT,
        required=json_schema.get("required", []),
        properties={
            name: process_property(prop_schema, json_schema)
            for name, prop_schema in json_schema["properties"].items()
        },
    )

    description_parts = tool.metadata.description.split("\n", maxsplit=1)
    # Create the function declaration
    return types.FunctionDeclaration(
        description=description_parts[-1] if len(description_parts) > 1 else None,
        name=tool.metadata.name,
        parameters=root_schema,
    )


class ChatParams(typing.TypedDict):
    model: str
    history: list[types.Content]
    config: types.GenerateContentConfig


def prepare_chat_params(
    model: str, messages: Sequence[ChatMessage], **kwargs: Any
) -> tuple[types.Content, ChatParams]:
    """
    Prepare common parameters for chat creation.

    Args:
        messages: Sequence of chat messages
        **kwargs: Additional keyword arguments

    Returns:
        tuple containing:
        - next_msg: the next message to send
        - chat_kwargs: processed keyword arguments for chat creation
    """
    merged_messages = merge_neighboring_same_role_messages(messages)
    *history, next_msg = map(chat_message_to_gemini, merged_messages)

    tools: types.Tool | list[types.Tool] | None = kwargs.pop("tools", None)
    if tools and not isinstance(tools, list):
        tools = [tools]

    config: types.GenerateContentConfig = kwargs.pop("generation_config", {})
    chat_kwargs: ChatParams = {"model": model, "history": history}

    if tools:
        chat_kwargs["config"] = types.GenerateContentConfig(
            **config,
            tools=tools,  # type: ignore
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True, maximum_remote_calls=None
            ),
            tool_config=kwargs.pop("tool_config", None),
        )
    else:
        chat_kwargs["config"] = config

    return next_msg, chat_kwargs
