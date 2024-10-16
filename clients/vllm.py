"""
vLLM client.

Please install [vLLM](https://github.com/vllm-project/vllm) according to its
installation guide before running this client.
"""

import time
from collections.abc import Generator

from transformers import AutoTokenizer
from vllm import SamplingParams, LLMEngine, EngineArgs

from client import Client, process_input, process_response
from conversation import Conversation


class VLLMClient(Client):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.engine_args = EngineArgs(
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",  # torch.bfloat16 is needed.
            trust_remote_code=True,
            gpu_memory_utilization=0.7,
            enforce_eager=False,
            worker_use_ray=False,
        )
        self.engine = LLMEngine.from_engine_args(self.engine_args)

    def generate_stream(
        self, tools: list[dict], history: list[Conversation], **parameters
    ) -> Generator[tuple[str | dict, list[dict]]]:
        chat_history = process_input(history, tools)
        model_inputs = self.tokenizer.apply_chat_template(
            chat_history, add_generation_prompt=True, tokenize=False
        )
        print(f'model_inputs:{model_inputs}')
        parameters["max_tokens"] = parameters.pop("max_new_tokens")
        params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 20,
            "use_beam_search": False,
            "length_penalty": 1,
            "early_stopping": False,
            "ignore_eos": False,
            "max_tokens": 1024,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }
        params_dict.update(parameters)
        sampling_params = SamplingParams(**params_dict)

        self.engine.add_request(
            request_id=str(time.time()), inputs=model_inputs, params=sampling_params
        )
        while self.engine.has_unfinished_requests():
            request_outputs = self.engine.step()
            for request_output in request_outputs:
                yield process_response(request_output.outputs[0].text, chat_history)