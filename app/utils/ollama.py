from ollama import Client
from pydantic import BaseModel


class ChatOllama:
    def __init__(self, model_name: str = "deepseek-r1:1.5b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = Client(host=host)

    def generate_response(self, messages: list[dict], format: dict):
        try:
            response = self.client.chat(
                messages=messages,
                model=self.model_name,
                format=format,
            )
            return response
        except ConnectionError as e:

            return f"ConnectionError: {e}"
        except ValueError as e:

            return f"ValueError: {e}"
        except Exception as e:

            return f"Exception: {e}"
