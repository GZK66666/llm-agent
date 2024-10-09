from http import HTTPStatus
from dashscope import Generation
import dashscope

from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from vllm.utils import random_uuid, iterate_with_cancellation
from transformers import AutoTokenizer

import asyncio

dashscope.api_key = 'sk-0eeedf01bd8842a6a7e68d97f135571e'

glm4_vllm_engine, tokenizer = None, None


def qwen_turbo(question, user_stop_words=[], model='qwen-turbo'):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': question}]
    gen = Generation()
    response = gen.call(
        model,
        messages=messages,
        result_format='message',  # set the result is message format.
        stop=user_stop_words
    )

    if response.status_code == HTTPStatus.OK:
        result = response.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        result = "error: " + f"{model}调用失败" + question

    return result


def load_glm4(model_path="./models/glm-4-9b-chat", gpu_memory_utilization=0.7):
    global glm4_vllm_engine, tokenizer

    engine_args = AsyncEngineArgs(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    glm4_vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        encode_special_tokens=True
    )


async def glm4(question, user_stop_words=[]):  # todo:stop words?
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
    if len(user_stop_words) > 0:
        params_dict["stop_token_ids"] = tokenizer.encode(user_stop_words[0], add_special_tokens=False)
        print(f'stop_token_ids:{params_dict["stop_token_ids"]}')
    sampling_params = SamplingParams(**params_dict)

    model_inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], add_generation_prompt=True,
                                                 tokenize=False)
    request_id = random_uuid()
    results_generator = glm4_vllm_engine.generate(
        inputs=model_inputs,
        sampling_params=sampling_params,
        request_id=request_id
    )

    response = None
    async for request_output in results_generator:
        response = request_output.outputs[0].text

    assert response is not None
    return response


if __name__ == "__main__":
    load_glm4()
    prompt = '''
    Today is 2024-10-09. Please Answer the following questions as best you can. You have access to the following tools:
    
    milvus_search: 这是一个向量数据库，里面存储的是海口市公安局提供的资料，可以用来搜索与海口市政务相关的问题。需要注意的是，搜索结果可能包含若干问答对资料，需要根据query与搜索资料的相似度酌情使用。,args: [{"name": "query", "description": "与海口政务相关的问题", "type": "string"}]
    
    These are chat history before:
    
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [milvus_search]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: 你好    
    '''
    print(asyncio.run(glm4(prompt, user_stop_words=['Observation'])))
    print(tokenizer.encode('Observation', add_special_tokens=False))