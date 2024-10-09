from agent import agent_execute_with_retry
from llm import load_glm4
import argparse
import os
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-turbo")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
    args = parser.parse_args()

    model = args.model
    if model == 'glm4':
        load_glm4()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    my_history = []
    while True:
        query = input('query:')
        success, result, my_history = asyncio.run(agent_execute_with_retry(query, chat_history=my_history, model=model))
        my_history = my_history[-10:]
