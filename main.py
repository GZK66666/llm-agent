import os
import traceback
from enum import Enum
from io import BytesIO
from uuid import uuid4

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from PIL import Image

from client import Client, ClientType, get_client
from conversation import (
    Conversation,
    Role,
    postprocess_text,
    response_to_str,
)
from tools.tool_registry import dispatch_tool, get_tools

CHAT_MODEL_PATH = "./models/glm-4-9b-chat"


def append_conversation(
        conversation: Conversation,
        history: list[Conversation],
        placeholder: DeltaGenerator | None = None,
) -> None:
    """
    Append a conversation piece into history, meanwhile show it in a new markdown block
    """
    history.append(conversation)
    conversation.show(placeholder)


st.set_page_config(
    page_title="GLM-4 Demo",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("TeleAgent")

with st.sidebar:
    top_p = st.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    top_k = st.slider("top_k", 1, 20, 10, step=1, key="top_k")
    temperature = st.slider("temperature", 0.0, 1.5, 0.95, step=0.01)
    repetition_penalty = st.slider("repetition_penalty", 0.0, 2.0, 1.0, step=0.01)
    max_new_tokens = st.slider("max_new_tokens", 1, 4096, 2048, step=1)
    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)

if clear_history:
    client = st.session_state.client
    st.session_state.clear()
    st.session_state.client = client
    st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.client = get_client(model_path=CHAT_MODEL_PATH, typ=ClientType.VLLM)

prompt_text = st.chat_input("Chat with GLM-4!", key="chat_input")

if prompt_text == "" and retry == False:
    print("\n== Clean ==\n")
    st.session_state.history = []
    exit()

history: list[Conversation] = st.session_state.history

if retry:
    print("\n== Retry ==\n")
    last_user_conversation_idx = None
    for idx, conversation in enumerate(history):
        if conversation.role.value == Role.USER.value:
            last_user_conversation_idx = idx
    if last_user_conversation_idx is not None:
        prompt_text = history[last_user_conversation_idx].content
        print(f"New prompt: {prompt_text}, idx = {last_user_conversation_idx}")
        del history[last_user_conversation_idx:]

for conversation in history:
    conversation.show()

tools = get_tools()

client: Client = st.session_state.client


def main(prompt_text: str):
    global client
    assert client is not None

    if prompt_text:
        prompt_text = prompt_text.strip()

        role = Role.USER
        append_conversation(Conversation(role, prompt_text), history)

        placeholder = st.container()
        message_placeholder = placeholder.chat_message(
            name="assistant", avatar="assistant"
        )
        markdown_placeholder = message_placeholder.empty()

        def add_new_block():
            nonlocal message_placeholder, markdown_placeholder
            message_placeholder = placeholder.chat_message(
                name="assistant", avatar="assistant"
            )
            markdown_placeholder = message_placeholder.empty()

        def commit_conversation(
                role: Role,
                text: str,
                metadata: str | None = None,
                image: str | None = None,
                new: bool = False,
        ):
            processed_text = postprocess_text(text, role.value == Role.ASSISTANT.value)
            conversation = Conversation(role, text, processed_text, metadata, image)

            # Use different placeholder for new block
            placeholder = message_placeholder if new else markdown_placeholder

            append_conversation(
                conversation,
                history,
                placeholder,
            )

        response = ""
        for _ in range(10):  # 最多调用10次tool
            last_response = None
            history_len = None

            try:
                for response, chat_history in client.generate_stream(
                        tools=tools,
                        history=history,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        max_new_tokens=max_new_tokens,
                ):
                    if history_len is None:
                        history_len = len(chat_history)
                    elif history_len != len(chat_history):
                        commit_conversation(Role.ASSISTANT, last_response)
                        add_new_block()
                        history_len = len(chat_history)
                    last_response = response
                    replace_quote = chat_history[-1]["role"] == "assistant"
                    markdown_placeholder.markdown(
                        postprocess_text(
                            str(response) + "●", replace_quote=replace_quote
                        )
                    )
                else:
                    metadata = (
                            isinstance(response, dict)
                            and response.get("name")
                            or None
                    )
                    role = Role.TOOL if metadata else Role.ASSISTANT
                    text = (
                        response.get("content")
                        if metadata
                        else response_to_str(response)
                    )
                    commit_conversation(role, text, metadata)
                    if metadata:
                        add_new_block()
                        try:
                            with markdown_placeholder:
                                with st.spinner(f"Calling tool {metadata}..."):
                                    observations = dispatch_tool(
                                        metadata, text
                                    )
                        except Exception as e:
                            traceback.print_exc()
                            st.error(f'Uncaught exception in `"{metadata}"`: {e}')
                            break

                        for observation in observations:
                            observation.text = observation.text
                            commit_conversation(
                                Role.OBSERVATION,
                                observation.text,
                                observation.role_metadata,
                                new=True,
                            )
                            add_new_block()
                        continue
                    else:
                        break
            except Exception as e:
                traceback.print_exc()
                st.error(f"Uncaught exception: {traceback.format_exc()}")
        else:
            st.error("Too many chaining function calls!")


main(prompt_text)
