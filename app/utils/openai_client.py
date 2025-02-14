import os
import openai
from utils.logger import Logger
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


class ChatGPTClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = Logger("ChatGPTClient")

    # def generate_json(self, prompt_template: str, input_var: dict):
    #     parser = JsonOutputParser()
    #     model = ChatOpenAI(
    #         model="gpt-4o-mini",
    #         temperature=0,
    #         max_tokens=None,
    #         timeout=None,
    #         max_retries=2,
    #         api_key=os.getenv("OPENAI_API_KEY"),
    #     )
    #     prompt = PromptTemplate(
    #         template=prompt_template, input_variables=list(input_var.keys())
    #     )
    #     chain = prompt | model | parser
    #     result = chain.invoke(input_var)
    #     return result

    def generate_response(
        self,
        system_prompt: str,
        request: str,
        response_format: dict,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request},
                ],
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except openai.APIConnectionError as e:
            self.logger.error(f"The server could not be reached: {e.__cause__}")
        except openai.RateLimitError as e:
            self.logger.error("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            self.logger.error(f"Another non-200-range status code({e.status_code}) was received: {e.response}")

        return ""
