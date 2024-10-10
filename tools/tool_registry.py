"""
This code is the tool registration part. By registering the tool, the model can call the tool.
This code provides extended functionality to the model, enabling it to call and interact with a variety of utilities
through defined interfaces.
"""

from collections.abc import Callable
import copy
import inspect
import json
import traceback
from types import GenericAlias
from typing import get_origin, Annotated
import subprocess

from .interface import ToolObservation
from .milvus_search import Retriever

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = []

milvus_retriever = Retriever()


def register_tool(func: Callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = []
    for name, param in python_params.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter `{name}` missing type annotation")
        if get_origin(annotation) != Annotated:
            raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")

        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        if not isinstance(description, str):
            raise TypeError(f"Description for `{name}` must be a string")
        if not isinstance(required, bool):
            raise TypeError(f"Required for `{name}` must be a bool")

        tool_params.append(
            {
                "name": name,
                "description": description,
                "type": typ,
                "required": required,
            }
        )
    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "params": tool_params,
    }
    # print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS.append(tool_def)

    return func


def dispatch_tool(tool_name: str, code: str) -> list[ToolObservation]:
    code = code.strip().rstrip('<|observation|>').strip()

    # Dispatch custom tools
    try:
        tool_params = json.loads(code)
    except json.JSONDecodeError as e:
        err = f"Error decoding JSON: {e}"
        return [ToolObservation("system_error", err)]

    if tool_name not in _TOOL_HOOKS:
        err = f"Tool `{tool_name}` not found. Please use a provided tool."
        return [ToolObservation("system_error", err)]

    tool_hook = _TOOL_HOOKS[tool_name]
    try:
        ret: str = tool_hook(**tool_params)
        return [ToolObservation(tool_name, str(ret))]
    except:
        err = traceback.format_exc()
        return [ToolObservation("system_error", err)]


def get_tools() -> list[dict]:
    return copy.deepcopy(_TOOL_DESCRIPTIONS)


# Tool Definitions


@register_tool
def milvus_search(
        query: Annotated[str, "与政务相关的问题", True],
) -> str:
    """
    这是一个存储政务资料的数据库，可以用来搜索与政务相关的问题，如果你不确定就应该用来搜索一下。需要注意的是，搜索结果可能包含脏数据或不相关数据，你需要根据query与结果的相似度酌情使用。
    """
    return milvus_retriever.retrieve(query)


if __name__ == "__main__":
    # print(dispatch_tool("milvus_search", "{\"query\": \"身份证丢了怎么办？\"}"))
    print(get_tools())
