"""Langfuse tracing middleware for Deep Agents tool calls and turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.observability import LANGFUSE_ENABLED, langfuse_context, observe, update_current_observation

_DEFAULT_MAX_ARG_CHARS = 2000
_DEFAULT_MAX_OUTPUT_CHARS = 2000
_DEFAULT_MAX_MESSAGES = 6
_LANGFUSE_PARENT_KEY = "_langfuse_parent"


class LangfuseState(AgentState):
    """State for Langfuse tracing middleware (private parent span linkage)."""

    _langfuse_parent: Annotated[NotRequired[dict[str, str]], PrivateStateAttr]


def _truncate_repr(value: Any, limit: int) -> str:
    if limit <= 0:
        return ""
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated, {len(text) - limit} chars)"


def _summarize_tool_result(result: ToolMessage | Command | Any, max_chars: int) -> dict[str, Any]:
    if isinstance(result, ToolMessage):
        return {
            "result_type": "ToolMessage",
            "tool_call_id": result.tool_call_id,
            "content_preview": _truncate_repr(result.content, max_chars),
        }

    if isinstance(result, Command):
        summary: dict[str, Any] = {"result_type": "Command"}
        update = getattr(result, "update", None)
        if isinstance(update, dict):
            summary["update_keys"] = sorted(str(key) for key in update.keys())
            messages = update.get("messages")
            if isinstance(messages, list):
                summary["message_count"] = len(messages)
        return summary

    return {
        "result_type": type(result).__name__,
        "repr": _truncate_repr(result, max_chars),
    }


def _extract_message_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message

    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str):
                    parts.append(text_part)
            else:
                text_part = getattr(item, "text", None)
                if isinstance(text_part, str):
                    parts.append(text_part)
        return "".join(parts)

    return _truncate_repr(message, 200)


def _extract_message_type(message: Any) -> str:
    msg_type = getattr(message, "type", None)
    if isinstance(msg_type, str) and msg_type:
        return msg_type
    return type(message).__name__


def _extract_tool_call_names(message: Any, limit: int = 10) -> list[str]:
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        if len(names) >= limit:
            break
    return names


def _summarize_messages(messages: Any, max_messages: int, max_chars: int) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    tail = messages[-max_messages:] if max_messages > 0 else messages
    summaries: list[dict[str, Any]] = []
    for msg in tail:
        summary: dict[str, Any] = {
            "type": _extract_message_type(msg),
            "preview": _truncate_repr(_extract_message_text(msg), max_chars),
        }
        tool_names = _extract_tool_call_names(msg)
        if tool_names:
            summary["tool_calls"] = tool_names
        summaries.append(summary)
    return summaries


def _extract_tool_names(tools: Any, limit: int = 50) -> list[str]:
    names: list[str] = []
    if not isinstance(tools, list):
        return names
    for tool in tools:
        name = None
        if hasattr(tool, "name"):
            name = getattr(tool, "name", None)
        elif isinstance(tool, dict):
            name = tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _get_langfuse_trace_context() -> dict[str, str] | None:
    trace_getter = getattr(langfuse_context, "get_current_trace_id", None)
    observation_getter = getattr(langfuse_context, "get_current_observation_id", None)

    trace_id = trace_getter() if callable(trace_getter) else None
    observation_id = observation_getter() if callable(observation_getter) else None

    if isinstance(trace_id, str) and isinstance(observation_id, str):
        return {"trace_id": trace_id, "observation_id": observation_id}
    return None


def _get_parent_from_runtime(runtime: Any) -> dict[str, str] | None:
    state = getattr(runtime, "state", None)
    if not isinstance(state, Mapping):
        return None
    parent = state.get(_LANGFUSE_PARENT_KEY)
    if not isinstance(parent, dict):
        return None
    trace_id = parent.get("trace_id")
    observation_id = parent.get("observation_id")
    if isinstance(trace_id, str) and isinstance(observation_id, str):
        return {"trace_id": trace_id, "observation_id": observation_id}
    return None


def _build_langfuse_parent_kwargs(parent: dict[str, str] | None) -> dict[str, str]:
    if not parent:
        return {}
    return {
        "langfuse_trace_id": parent["trace_id"],
        "langfuse_parent_observation_id": parent["observation_id"],
    }


def _split_model_response(
    response: ModelResponse[Any] | ExtendedModelResponse[Any],
) -> tuple[ModelResponse[Any], Command | None]:
    model_response = response
    command = None
    if hasattr(response, "model_response") and hasattr(response, "command"):
        model_response = getattr(response, "model_response")
        command = getattr(response, "command")
    return model_response, command


def _merge_command(existing: Command | None, update: dict[str, Any]) -> Command:
    if existing is None:
        return Command(update=update)

    existing_update = getattr(existing, "update", None)
    if isinstance(existing_update, dict):
        merged_update: dict[str, Any] = {**existing_update, **update}
    elif isinstance(existing_update, list):
        try:
            merged_update = {**dict(existing_update), **update}
        except (TypeError, ValueError):
            merged_update = update
    else:
        merged_update = update

    return Command(
        update=merged_update,
        goto=getattr(existing, "goto", None),
        resume=getattr(existing, "resume", None),
        graph=getattr(existing, "graph", None),
    )


def _summarize_model_request(request: ModelRequest[Any], max_chars: int) -> dict[str, Any]:
    messages = getattr(request, "messages", None)
    tool_names = _extract_tool_names(getattr(request, "tools", None))
    system_message = getattr(request, "system_message", None)
    model = getattr(request, "model", None)
    model_name = None
    if model is not None:
        model_name = getattr(model, "model_name", None)
        if model_name is None:
            model_name = getattr(model, "model", None)
        if model_name is None:
            model_name = getattr(model, "model_id", None)

    summary: dict[str, Any] = {
        "message_count": len(messages) if isinstance(messages, list) else None,
        "tool_count": len(tool_names) if tool_names else 0,
        "tool_names": tool_names,
        "message_tail": _summarize_messages(messages, _DEFAULT_MAX_MESSAGES, max_chars),
    }
    if model_name:
        summary["model_name"] = str(model_name)
    if system_message is not None:
        system_text = _extract_message_text(system_message)
        summary["system_message_length"] = len(system_text)
        summary["system_message_preview"] = _truncate_repr(system_text, max_chars)
    return summary


def _summarize_model_response(response: ModelResponse[Any], max_chars: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"response_type": type(response).__name__}
    message = getattr(response, "message", None)
    if message is None:
        messages = getattr(response, "messages", None)
        if isinstance(messages, list) and messages:
            message = messages[-1]
    if message is not None:
        summary["content_preview"] = _truncate_repr(_extract_message_text(message), max_chars)
        summary["message_type"] = _extract_message_type(message)
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            summary["tool_call_count"] = len(tool_calls)
            summary["tool_call_names"] = _extract_tool_call_names(message)
    return summary


class LangfuseTracingMiddleware(AgentMiddleware[LangfuseState, ContextT, ResponseT]):
    """Trace tool calls and agent turns to Langfuse when configured."""

    state_schema = LangfuseState

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_arg_chars: int = _DEFAULT_MAX_ARG_CHARS,
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self._enabled = enabled
        self._max_arg_chars = max_arg_chars
        self._max_output_chars = max_output_chars

    def _should_trace(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        return LANGFUSE_ENABLED

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if not self._should_trace():
            return handler(request)

        tool_call = request.tool_call if isinstance(request.tool_call, dict) else {}
        tool_name = str(tool_call.get("name") or "unknown_tool")
        tool_call_id = tool_call.get("id")
        tool_args = tool_call.get("args")

        @observe(name=f"tool.{tool_name}")
        def _run() -> ToolMessage | Command:
            update_current_observation(
                input={
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "args_preview": _truncate_repr(tool_args, self._max_arg_chars),
                }
            )
            try:
                result = handler(request)
            except Exception as exc:
                update_current_observation(
                    output={
                        "error_type": type(exc).__name__,
                        "error_message": _truncate_repr(exc, self._max_output_chars),
                    }
                )
                raise
            update_current_observation(output=_summarize_tool_result(result, self._max_output_chars))
            return result

        parent = _get_parent_from_runtime(request.runtime)
        return _run(**_build_langfuse_parent_kwargs(parent))

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
        if not self._should_trace():
            return handler(request)

        @observe(name="agent.turn")
        def _run() -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
            update_current_observation(input=_summarize_model_request(request, self._max_arg_chars))
            try:
                response = handler(request)
            except Exception as exc:
                update_current_observation(
                    output={
                        "error_type": type(exc).__name__,
                        "error_message": _truncate_repr(exc, self._max_output_chars),
                    }
                )
                raise
            model_response, existing_command = _split_model_response(response)
            update_current_observation(output=_summarize_model_response(model_response, self._max_output_chars))
            parent = _get_langfuse_trace_context()
            if parent is None:
                return response
            merged_command = _merge_command(existing_command, {_LANGFUSE_PARENT_KEY: parent})
            return ExtendedModelResponse(
                model_response=model_response,
                command=merged_command,
            )

        return _run()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if not self._should_trace():
            return await handler(request)

        tool_call = request.tool_call if isinstance(request.tool_call, dict) else {}
        tool_name = str(tool_call.get("name") or "unknown_tool")
        tool_call_id = tool_call.get("id")
        tool_args = tool_call.get("args")

        @observe(name=f"tool.{tool_name}")
        async def _run() -> ToolMessage | Command:
            update_current_observation(
                input={
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "args_preview": _truncate_repr(tool_args, self._max_arg_chars),
                }
            )
            try:
                result = await handler(request)
            except Exception as exc:
                update_current_observation(
                    output={
                        "error_type": type(exc).__name__,
                        "error_message": _truncate_repr(exc, self._max_output_chars),
                    }
                )
                raise
            update_current_observation(output=_summarize_tool_result(result, self._max_output_chars))
            return result

        parent = _get_parent_from_runtime(request.runtime)
        return await _run(**_build_langfuse_parent_kwargs(parent))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
        if not self._should_trace():
            return await handler(request)

        @observe(name="agent.turn")
        async def _run() -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
            update_current_observation(input=_summarize_model_request(request, self._max_arg_chars))
            try:
                response = await handler(request)
            except Exception as exc:
                update_current_observation(
                    output={
                        "error_type": type(exc).__name__,
                        "error_message": _truncate_repr(exc, self._max_output_chars),
                    }
                )
                raise
            model_response, existing_command = _split_model_response(response)
            update_current_observation(output=_summarize_model_response(model_response, self._max_output_chars))
            parent = _get_langfuse_trace_context()
            if parent is None:
                return response
            merged_command = _merge_command(existing_command, {_LANGFUSE_PARENT_KEY: parent})
            return ExtendedModelResponse(
                model_response=model_response,
                command=merged_command,
            )

        return await _run()


__all__ = ["LangfuseTracingMiddleware"]
