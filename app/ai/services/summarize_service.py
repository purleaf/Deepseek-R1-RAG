import json
from utils.openai_client import ChatGPTClient
from ai.services.rag_service import VectorDBManager
from utils.ollama import ChatOllama
from pydantic import BaseModel


class Response(BaseModel):
    summary: str


class SummarizeService:
    def __init__(self, collection_name):
        self.ollama_client = ChatOllama(host='http://localhost:11434')
        self.vector_db_manager = VectorDBManager(collection_name=collection_name)
        self.openai_client = ChatGPTClient()

    def summary(self, request):
        context = " ".join(
            self.vector_db_manager.collection.query(query_texts=request, n_results=3)["documents"][0])
        system_prompt = {
            "role": "system",
            "content": """
                        "You are an expert AI assistant specialized in analyzing textual content and generating precise, concise, and contextually accurate outputs." 
                        "Your role is to help the student understand and engage with the provided study materials." 
                        "You will be given:"
                            "1. **Document Chunks**: Relevant excerpts from the study material retrieved from a vector database."
                            "2. **Text to summarize**: The text that requires summarization."
                        "When given a context, you must:"
                            "1. Analyze the provided content carefully."
                            "2. Generate a concise and coherent summary that encapsulates the main ideas."
                            "3. Ensure your responses are precise, devoid of any additional commentary, and fully aligned with the context provided."
                        """,
        }
        human_prompt = {
            "role": "user",
            "content": f"""
                        "### Document Chunks:"
                            "{context}"
                        "### Text to summarize:"
                            "{request}"
                        """,
        }
        response_format = Response.model_json_schema()
        try:
            response = self.ollama_client.generate_response(messages=[system_prompt, human_prompt],
                                                            format=response_format)
            return json.loads(response.message.content)['summary']
        except ConnectionError as e:
            return f"ConnectionError: {e}"
        except ValueError as e:
            return f"ValueError: {e}"
        except Exception as e:
            return f"Exception: {e}"
