import os
from utils.logger import Logger
from ai.services.rag_service import VectorDBManager
from utils.openai_client import ChatGPTClient
import json
import openai
from ollama import Client
from pydantic import BaseModel
from utils.ollama import ChatOllama

class Response(BaseModel):
    answer: str


class AskAIService:
    def __init__(self, collection_name: str):
        self.vector_db_manager = VectorDBManager(collection_name=collection_name)
        self.openai_manager = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.ollama_client = ChatOllama(host='http://localhost:11434')
        self.logger = Logger("AskAIService")

    def local_model(self, request: str, chat_history: list[dict] = None):
        context = " ".join(self.vector_db_manager.collection.query(query_texts=request, n_results=20)["documents"][0])
        system_prompt = {
            "role": "system",
            "content": """
        "You are an AI chat bot assistant in a student's study room." 
        "Your role is to help the student understand and engage with the provided study materials." 
        "You will be given:"
        "1. **Chat History**: Previous interactions between you and the student."
        "2. **Document Chunks**: Relevant excerpts from the study material retrieved from a vector database."
        "3. **User's Request**: The current question or statement from the student."

        "**Guidelines:**"
            "- **Use Only Provided Information**:" 
                "Base your responses solely on the **Chat History** and **Document Chunks**." 
                "Do not incorporate any external knowledge or make assumptions beyond the given data."

            "- **Be Clear and Concise**:" 
                "Provide explanations that are easy to understand, avoiding unnecessary jargon unless it is part" 
                "of the provided material."

            "- **Maintain Context**: "
                "Ensure continuity by considering the **Chat History**. Your responses should build upon previous" 
                "interactions when relevant."

            "- **Stay Relevant**:" 
                "Address the **User's Request** directly, ensuring that your response is pertinent and helpful."

            "- **Format**:" 
                "Respond in a clear and organized manner, using bullet points or numbered lists if it enhances clarity."

            "- **Be interactive with user and pretend to be a human**:" 
                "If the user asks you simple questions, for a small talk, you can play and interact like a users friend"
            "- **Be flexible**"
                "If the user's question is about a general knowledge that is well known to everyone provide an answer"
                 "on the user's question even if there is nothing mentioned in the provided context"
            """,
        }
        human_prompt = {
            "role": "user",
            "content": f"""
        "### Document Chunks:"
            "{context}"

        "### User's Request:"
            "{request}"
        """,
        }
        messages = [system_prompt]
        if chat_history:
            messages.extend(chat_history)
        messages.append(human_prompt)

        try:
            response = self.ollama_client.generate_response(messages=messages, format=Response.model_json_schema())
            return json.loads(response.message.content)['answer']
        except ConnectionError as e:
            self.logger.error(f"Connection error occurred: {e}")
            return f"ConnectionError: {e}"
        except ValueError as e:
            self.logger.error(f"Invalid parameter error: {e}")
            return f"ValueError: {e}"
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            return f"Exception: {e}"


    def generate_response(self, request: str, chat_history: list[dict] = None):
        context = " ".join(
            self.vector_db_manager.collection.query(query_texts=request, n_results=20)[
                "documents"
            ][0]
        )
        system_prompt = {
            "role": "system",
            "content": """
        "You are an AI chat bot assistant in a student's study room." 
        "Your role is to help the student understand and engage with the provided study materials." 
        "You will be given:"
        "1. **Chat History**: Previous interactions between you and the student."
        "2. **Document Chunks**: Relevant excerpts from the study material retrieved from a vector database."
        "3. **User's Request**: The current question or statement from the student."
        
        "**Guidelines:**"
            "- **Use Only Provided Information**:" 
                "Base your responses solely on the **Chat History** and **Document Chunks**." 
                "Do not incorporate any external knowledge or make assumptions beyond the given data."
  
            "- **Be Clear and Concise**:" 
                "Provide explanations that are easy to understand, avoiding unnecessary jargon unless it is part" 
                "of the provided material."
  
            "- **Maintain Context**: "
                "Ensure continuity by considering the **Chat History**. Your responses should build upon previous" 
                "interactions when relevant."
  
            "- **Stay Relevant**:" 
                "Address the **User's Request** directly, ensuring that your response is pertinent and helpful."
  
            "- **Format**:" 
                "Respond in a clear and organized manner, using bullet points or numbered lists if it enhances clarity."

            "- **Be interactive with user and pretend to be a human**:" 
                "If the user asks you simple questions, for a small talk, you can play and interact like a users friend"
            "- **Be flexible**"
                "If the user's question is about a general knowledge that is well known to everyone provide an answer"
                 "on the user's question even if there is nothing mentioned in the provided context"
            """,
        }
        human_prompt = {
            "role": "user",
            "content": f"""
        "### Document Chunks:"
            "{context}"

        "### User's Request:"
            "{request}"
        """,
        }
        messages = [system_prompt]
        if chat_history:
            messages.extend(chat_history)
        messages.append(human_prompt)
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
        try:
            response = self.openai_manager.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format=response_format,
                temperature=0.5,
                max_tokens=1000,
            )
            return json.loads(response.choices[0].message.content)["ai_reply"]
        except openai.APIConnectionError as e:
            self.logger.error(f"The server could not be reached: {e.__cause__}")
        except openai.RateLimitError as e:
            self.logger.error(f"A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            self.logger.error(f"Another non-200-range status code({e.status_code}) was received: {e.response}")
        return ""
