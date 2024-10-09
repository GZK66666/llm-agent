from tools import milvus_search
from llm import qwen_turbo, glm4
import json
import datetime

# 拼接工具说明
tools = [milvus_search]
tool_names = ' or '.join([tool.name for tool in tools])
tool_descs = []
for t in tools:
    args_desc = []
    for name, info in t.args.items():
        args_desc.append(
            {'name': name, 'description': info['description'] if 'description' in info else '', 'type': info['type']})
    args_desc = json.dumps(args_desc, ensure_ascii=False)
    tool_descs.append('%s: %s,args: %s' % (t.name, t.description, args_desc))
tool_descs = '\n'.join(tool_descs)

# agent prompt template
prompt_tpl = '''
Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''


async def agent_execute(query, chat_history=[], model='qwen-turbo'):
    global tools, tool_names, tool_descs, prompt_tpl

    agent_scratchpad = ''  # agent执行过程
    while True:  # todo: max thought
        # 1）触发llm思考下一步action
        history = '\n'.join(['Question:%s\nAnswer:%s' % (his[0], his[1]) for his in chat_history])
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        prompt = prompt_tpl.format(today=today, chat_history=history, tool_descs=tool_descs, tool_names=tool_names,
                                   query=query, agent_scratchpad=agent_scratchpad)
        print('\033[32m---等待LLM返回... ...\n%s\n\033[0m' % prompt, flush=True)
        if model == 'glm4':
            response = await glm4(question=prompt, user_stop_words=['Observation'])
        else:
            response = qwen_turbo(question=prompt, user_stop_words=['Observation:'])
        print('---LLM返回---\n%s\n---' % response, flush=True)

        # 2）解析thought+action+action input+observation or thought+final answer
        thought_i = response.rfind('Thought:')
        final_answer_i = response.rfind('\nFinal Answer:')
        action_i = response.rfind('\nAction:')
        action_input_i = response.rfind('\nAction Input:')
        observation_i = response.rfind('\nObservation:')

        # 3）返回final answer，执行完成
        if final_answer_i != -1 and thought_i < final_answer_i:
            final_answer = response[final_answer_i + len('\nFinal Answer:'):].strip()
            chat_history.append((query, final_answer))
            return True, final_answer, chat_history

        # 4）解析action
        if not (thought_i < action_i < action_input_i):
            return False, 'LLM回复格式异常', chat_history
        if observation_i == -1:
            observation_i = len(response)
            response = response + 'Observation: '
        thought = response[thought_i + len('Thought:'):action_i].strip()
        action = response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i + len('\nAction Input:'):observation_i].strip()

        # 5）匹配tool
        the_tool = None
        for t in tools:
            if t.name == action:
                the_tool = t
                break
        if the_tool is None:
            observation = 'the tool not exist'
            agent_scratchpad = agent_scratchpad + response + observation + '\n'
            continue

        # 6）执行tool
        try:
            action_input = json.loads(action_input)
            tool_ret = the_tool.invoke(action_input)
        except Exception as e:
            observation = 'the tool has error:{}'.format(e)
        else:
            observation = str(tool_ret)
        agent_scratchpad = agent_scratchpad + response + observation + '\n'


async def agent_execute_with_retry(query, chat_history=[], model='qwen-turbo', retry_times=3):
    for i in range(retry_times):
        success, result, chat_history = await agent_execute(query, chat_history=chat_history, model=model)
        if success:
            return success, result, chat_history
    return success, result, chat_history
