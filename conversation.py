import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from PIL.Image import Image

QUOTE_REGEX = re.compile(r"【(\d+)†(.+?)】")

SELFCOG_PROMPT = "你是一个名为 TeleAgent 的人工智能助手。你是基于星辰大模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。"
DATE_PROMPT = "当前日期: %Y-%m-%d"


def build_system_prompt(
    functions: list[dict],
):
    value = SELFCOG_PROMPT
    value += "\n\n" + datetime.now().strftime(DATE_PROMPT)
    value += "\n\n# 可用工具"
    contents = []
    for function in functions:
        content = f"\n\n## {function['name']}\n\n{json.dumps(function, ensure_ascii=False, indent=4)}"
        content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
        contents.append(content)
    value += "".join(contents)
    return value


def response_to_str(response: str | dict[str, str]) -> str:
    """
    Convert response to string.
    """
    if isinstance(response, dict):
        return response.get("name", "") + response.get("content", "")
    return response


class Role(Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()
    OBSERVATION = auto()

    def __str__(self):
        match self:
            case Role.SYSTEM:
                return "<|system|>"
            case Role.USER:
                return "<|user|>"
            case Role.ASSISTANT | Role.TOOL:
                return "<|assistant|>"
            case Role.OBSERVATION:
                return "<|observation|>"

    # Get the message block for the given role
    def get_message(self):
        # Compare by value here, because the enum object in the session state
        # is not the same as the enum cases here, due to streamlit's rerunning
        # behavior.
        match self.value:
            case Role.SYSTEM.value:
                return
            case Role.USER.value:
                return st.chat_message(name="user", avatar="user")
            case Role.ASSISTANT.value:
                return st.chat_message(name="assistant", avatar="assistant")
            case Role.TOOL.value:
                return st.chat_message(name="tool", avatar="assistant")
            case Role.OBSERVATION.value:
                return st.chat_message(name="observation", avatar="assistant")
            case _:
                st.error(f"Unexpected role: {self}")


@dataclass
class Conversation:
    role: Role
    content: str | dict
    # Processed content
    saved_content: str | None = None
    metadata: str | None = None
    image: str | Image | None = None

    def __str__(self) -> str:
        metadata_str = self.metadata if self.metadata else ""
        return f"{self.role}{metadata_str}\n{self.content}"

    # Human readable format
    def get_text(self) -> str:
        text = self.saved_content or self.content
        match self.role.value:
            case Role.TOOL.value:
                text = f"Calling tool `{self.metadata}`:\n\n```python\n{text}\n```"
            case Role.OBSERVATION.value:
                text = f"```python\n{text}\n```"
        return text

    # Display as a markdown block
    def show(self, placeholder: DeltaGenerator | None = None) -> str:
        if placeholder:
            message = placeholder
        else:
            message = self.role.get_message()

        if self.image:
            message.image(self.image, width=512)

        if self.role == Role.OBSERVATION:
            metadata_str = f"from {self.metadata}" if self.metadata else ""
            message = message.expander(f"Observation {metadata_str}")

        text = self.get_text()
        if self.role != Role.USER:
            show_text = text
        else:
            splitted = text.split('files uploaded.\n')
            if len(splitted) == 1:
                show_text = text
            else:
                # Show expander for document content
                doc = splitted[0]
                show_text = splitted[-1]
                expander = message.expander(f'File Content')
                expander.markdown(doc)
        message.markdown(show_text)


def postprocess_text(text: str, replace_quote: bool) -> str:
    text = text.replace("\(", "$")
    text = text.replace("\)", "$")
    text = text.replace("\[", "$$")
    text = text.replace("\]", "$$")
    text = text.replace("<|assistant|>", "")
    text = text.replace("<|observation|>", "")
    text = text.replace("<|system|>", "")
    text = text.replace("<|user|>", "")
    text = text.replace("<|endoftext|>", "")

    return text.strip()