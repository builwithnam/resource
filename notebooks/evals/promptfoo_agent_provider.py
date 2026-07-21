"""Promptfoo Python provider for the calculator tool evaluation."""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AZURE_OPENAI_API_VERSION = "2025-04-01-preview"
DEFAULT_AZURE_OPENAI_ENDPOINT = "https://bookingcare-ai-nam-resource.cognitiveservices.azure.com"
DEFAULT_AZURE_OPENAI_MODEL = "gpt-5.5"
DEFAULT_MAX_TURNS = 12

load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")

EVALUATION_PROMPT = """You are an AI assistant with access to tools.

When given a task, you MUST:
1. Use the available tools to complete the task
2. Provide summary of each step in your approach, wrapped in <summary> tags
3. Provide feedback on the tools provided, wrapped in <feedback> tags
4. Provide your final response, wrapped in <response> tags

Summary Requirements:
- In your <summary> tags, you must explain:
    - The steps you took to complete the task
    - Which tools you used, in what order, and why
    - The inputs you provided to each tool
    - The outputs you received from each tool
    - A summary for how you arrived at the response

Feedback Requirements:
- In your <feedback> tags, provide constructive feedback on the tools:
    - Comment on tool names: Are they clear and descriptive?
    - Comment on input parameters: Are they well documented? Are required vs optional parameters clear?
    - Comment on descriptions: Do they accurately describe what the tool does?
    - Comment on any errors encountered during tool usage: Did the tool fail to execute? Did the tool return too many tokens?
    - Identify specific areas for improvement and explain why they would help
    - Be specific and actionable in your suggestions

Response Requirements:
- Your response should be concise and directly address what was asked
- Always wrap your final response in <response> tags
- If you cannot solve the task return <response>NOT_FOUND</response>
- For numeric responses, provide just the number
- For names or text, provide the exact text requested
- Your response should go last
"""

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_NAMES = {
    "e": math.e,
    "pi": math.pi,
    "tau": math.tau,
}
_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "degrees": math.degrees,
    "exp": math.exp,
    "floor": math.floor,
    "len": len,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "pow": pow,
    "radians": math.radians,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "sum": sum,
    "tan": math.tan,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.List | ast.Tuple):
        return [_safe_eval(element) for element in node.elts]

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Unsupported name: {node.id}")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        operator_function = _ALLOWED_BINARY_OPERATORS.get(operator_type)
        if operator_function is None:
            raise ValueError(f"Unsupported binary operator: {operator_type.__name__}")
        return operator_function(_safe_eval(node.left), _safe_eval(node.right))

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        operator_function = _ALLOWED_UNARY_OPERATORS.get(operator_type)
        if operator_function is None:
            raise ValueError(f"Unsupported unary operator: {operator_type.__name__}")
        return operator_function(_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise ValueError("Unsupported function call")
        args = [_safe_eval(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[node.func.id](*args)

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate arithmetic expressions using a small safe expression interpreter."""
    try:
        normalized_expression = expression.replace("^", "**")
        parsed = ast.parse(normalized_expression, mode="eval")
        return str(_safe_eval(parsed))
    except Exception as exc:
        return f"Error: {exc}"


CALCULATOR_TOOL = {
    "type": "function",
    "name": "calculator",
    "description": "",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "",
            }
        },
        "additionalProperties": False,
        "required": ["expression"],
    },
}

TOOL_MAPPING = {
    "calculator": calculator,
}


def _config_value(
    config: dict[str, Any],
    key: str,
    env_name: str,
    default: str,
) -> str:
    return str(config.get(key) or os.getenv(env_name, default))


def _extract_xml_content(text: str, tag: str) -> str | None:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _tool_calls_from_response(response: Any) -> list[Any]:
    return [
        item
        for item in response.output
        if getattr(item, "type", None) == "function_call"
    ]


def _function_call_input(tool_call: Any) -> dict[str, str]:
    # Avoid passing transient response item IDs back into the API when store=False.
    return {
        "type": "function_call",
        "call_id": tool_call.call_id,
        "name": tool_call.name,
        "arguments": tool_call.arguments or "{}",
    }


def _tool_call_output(call_id: str, output: str) -> dict[str, str]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def _total_tool_calls(tool_metrics: dict[str, dict[str, Any]]) -> int:
    return sum(metrics["count"] for metrics in tool_metrics.values())


def _record_tool_call(
    tool_metrics: dict[str, dict[str, Any]],
    function_name: str,
    duration_seconds: float,
) -> None:
    metrics = tool_metrics.setdefault(function_name, {"count": 0, "durations": []})
    metrics["count"] += 1
    metrics["durations"].append(duration_seconds)


def _format_tool_error(function_name: str, exc: Exception) -> str:
    return f"Error executing tool {function_name}: {exc}\n{traceback.format_exc()}"


def _execute_tool_call(
    tool_call: Any,
    tool_metrics: dict[str, dict[str, Any]],
) -> dict[str, str]:
    function_name = tool_call.name
    started_at = time.time()

    try:
        target_tool = TOOL_MAPPING.get(function_name)
        if target_tool is None:
            raise ValueError(f"Unknown tool: {function_name}")
        parsed_arguments = json.loads(tool_call.arguments or "{}")
        tool_result_text = json.dumps(target_tool(**parsed_arguments))
    except Exception as exc:
        tool_result_text = _format_tool_error(function_name, exc)

    _record_tool_call(tool_metrics, function_name, time.time() - started_at)
    return _tool_call_output(tool_call.call_id, tool_result_text)


def _final_provider_response(
    raw_output: str,
    tool_metrics: dict[str, dict[str, Any]],
    start_time: float,
    context: dict[str, Any],
) -> dict[str, Any]:
    final_response = _extract_xml_content(raw_output, "response") or raw_output.strip()
    return {
        "output": final_response,
        "metadata": {
            "summary": _extract_xml_content(raw_output, "summary"),
            "feedback": _extract_xml_content(raw_output, "feedback"),
            "rawOutput": raw_output,
            "toolCalls": tool_metrics,
            "numToolCalls": _total_tool_calls(tool_metrics),
            "durationSeconds": time.time() - start_time,
            "expected": (context or {}).get("vars", {}).get("expected"),
        },
    }


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Promptfoo provider entrypoint."""
    start_time = time.time()
    config = (options or {}).get("config", {})
    model = _config_value(config, "model", "AZURE_OPENAI_MODEL", DEFAULT_AZURE_OPENAI_MODEL)
    api_version = _config_value(
        config,
        "api_version",
        "AZURE_OPENAI_API_VERSION",
        DEFAULT_AZURE_OPENAI_API_VERSION,
    )
    azure_endpoint = _config_value(
        config,
        "azure_endpoint",
        "AZURE_OPENAI_ENDPOINT",
        DEFAULT_AZURE_OPENAI_ENDPOINT,
    )
    max_turns = int(config.get("max_turns", DEFAULT_MAX_TURNS))

    client = AzureOpenAI(api_version=api_version, azure_endpoint=azure_endpoint)
    messages: list[Any] = [
        {"role": "system", "content": EVALUATION_PROMPT},
        {"role": "user", "content": prompt},
    ]
    tool_metrics: dict[str, dict[str, Any]] = {}

    for _ in range(max_turns):
        response = client.responses.create(
            model=model,
            input=messages,
            tools=[CALCULATOR_TOOL],
            tool_choice="auto",
            store=False,
        )

        tool_calls = _tool_calls_from_response(response)
        if not tool_calls:
            return _final_provider_response(
                raw_output=_response_text(response),
                tool_metrics=tool_metrics,
                start_time=start_time,
                context=context,
            )

        messages.extend(_function_call_input(tool_call) for tool_call in tool_calls)
        messages.extend(_execute_tool_call(tool_call, tool_metrics) for tool_call in tool_calls)

    return {
        "output": "NOT_FOUND",
        "error": f"Agent exceeded max_turns={max_turns}",
        "metadata": {
            "toolCalls": tool_metrics,
            "numToolCalls": _total_tool_calls(tool_metrics),
            "durationSeconds": time.time() - start_time,
        },
    }
