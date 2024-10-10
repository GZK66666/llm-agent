from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import (
    connections,
    utility,
    Collection,
)
import logging


class Retriever:
    def __init__(self, collection_name: str = "hkgahz"):
        self._connect_to_milvus()
        self.collection_name = collection_name
        self.embedding_model = self._load_embedding_model()

    def _load_embedding_model(self):
        return HuggingFaceEmbeddings(model_name="./models/bge-large-zh-v1.5")

    def _connect_to_milvus(self):
        # using "default" database if not specific other name
        connections.connect("default", host="localhost", port="19530", token="root:Szgs@2024")
        logging.info("Milvus connect success!")

    def _format_retrieve_results(self, retrieve_results):
        final_result = "\n【资料】"
        for i, hit in enumerate(retrieve_results[0]):
            final_result = final_result + f"\n ########### 资料({str(i + 1)}) ###########"
            final_result = final_result + "\n提问：" + hit.entity.get('question')
            final_result = final_result + "\n解答：" + hit.entity.get('answer')
            # final_result = final_result + "\n检索相似度：" + str(hit.distance)

        return final_result

    def retrieve(self, question, k=3):
        if not utility.has_collection(self.collection_name):
            logging.info("{} is not exist", self.collection_name)
            return ""

        collection = Collection(self.collection_name)
        collection.load()  # check if collection is loaded, if loaded then not exec

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        question_embedding = self.embedding_model.embed_query(question)
        results = collection.search([question_embedding], "vectors", search_params, limit=k,
                                    output_fields=["question", "answer"])

        return self._format_retrieve_results(results)
