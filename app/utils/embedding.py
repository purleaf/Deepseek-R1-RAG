from chromadb.utils import embedding_functions
import os


class Embedding:
    def __init__(self):
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
