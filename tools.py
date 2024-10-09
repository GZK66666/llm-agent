from langchain_core.tools import tool
from typing import Annotated
from milvus_search import Retriever

milvus_retriever = Retriever()


@tool
def milvus_search(
        query: Annotated[str, '与海口政务相关的问题']
):
    """这是一个存储海口市公安局政务资料的数据库，可以用来搜索与政务相关的问题，如果你不确定就应该用来搜索一下。需要注意的是，搜索结果可能包含脏数据或不相关数据，你需要根据query与结果的相似度酌情使用。"""
    return milvus_retriever.retrieve(query)
