from utils.openai_client import ChatGPTClient
from ai.services.rag_service import VectorDBManager
import json
from utils.ollama import ChatOllama
from pydantic import BaseModel


class Response(BaseModel):
    summary: str


class TranslateService:
    def __init__(self, collection_name):
        self.gpt_model = ChatGPTClient()
        self.ollama_model = ChatOllama(host='http://localhost:11434')
        self.vector_db = VectorDBManager(collection_name=collection_name)

    def translate(self, text, language):
        system_prompt = {
            "role": "system",
            "content": """
                                "You are an expert translator with deep proficiency in accurately converting text from one language to another. 
                                "Your translations are precise, contextually appropriate, and adapted to the specific context provided." 
                                "You prioritize maintaining the original meaning, tone, and cultural nuances of the source text while ensuring clarity and fluency in the target language."  
                                "You will be given:"
                                    "1. **Document Chunks**: Relevant excerpts from the study material retrieved from a vector database."
                                    "2. **Text to translate**: A specific context for the translation (e.g., technical, legal, literary, or conversational)."
                                "When given a context, you must:"
                                    "1. Analyze the context carefully and tailor your translation to align with the provided purpose and audience."
                                    "2. Avoid overly literal translations unless explicitly requested. Focus on natural phrasing that conveys the intended meaning accurately."
                                    "3. Maintain key terminology and style consistent with the context."
                                    "4. Do not include any additional commentary or information beyond the required JSON format." 
                                    "5. Adhere strictly to the context provided, ensuring your translations meet the specific needs outlined."
                                """,
        }

        context = "".join(self.vector_db.collection.query(query_texts=text, n_results=3)["documents"][0])

        human_prompt = f"""
        Here is the context in which you have to translate: {context}
        Here is the text you have to translate to {language}: {text}"""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "reply_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ai_reply": {
                            "description": "The AI's answer to the user's question based on the provided information.",
                            "type": "string",
                        }
                    },
                    "required": ["ai_reply"],
                    "additionalProperties": False,
                },
            },
        }

        result = self.gpt_model.generate_response(
            system_prompt=system_prompt,
            request=human_prompt,
            response_format=response_format,
            max_tokens=1000,
        )
        return json.loads(result)["ai_reply"]
