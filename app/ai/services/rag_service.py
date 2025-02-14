import json
import chromadb
import nltk
from nltk.tokenize import sent_tokenize
from utils.embedding import Embedding
from utils.logger import Logger
from utils.openai_client import ChatGPTClient
from ollama import Client
from pydantic import BaseModel
from utils.ollama import ChatOllama


class Response(BaseModel):
    answer: str

class VectorDBManager:
    def __init__(self, db_path="./chroma_db", collection_name="study_room_vectors_v2"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.openai = ChatGPTClient()
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.ollama_client = ChatOllama(host='http://localhost:11434')
        self.embedding = Embedding()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger = Logger("VectorDBManager")

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    def get_embedding(self, content):
        try:
            response = self.openai.client.embeddings.create(
                model="text-embedding-3-small", input=content
            )
            return response.data
        except Exception as e:
            self.logger.error(f"Error fetching embedding: {e}")
            return None

    def create_chunks(self, sentences, max_chunk_length=800, similarity_threshold=0.8):
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_chunk_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

            if current_length >= max_chunk_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def add_document(self, content, document_id, user_id):
        sentences = sent_tokenize(content)
        chunked_text = self.create_chunks(sentences, max_chunk_length=800)
        chunk_embeddings = self.get_embedding(chunked_text)

        for i, chunk in enumerate(chunked_text):
            metadata = {"document_id": document_id, "user_id": user_id}
            self.collection.add(
                documents=[chunk],
                embeddings=[chunk_embeddings[i].embedding],
                metadatas=metadata,
                ids=[f"doc_{document_id}_chunk_{i}"],
            )

    def get_document_content(self, request: str, num_results=20):
        results = self.collection.query(query_texts=request, n_results=num_results)
        return results  # ['documents'] if results['documents'][0] else None

    def answer_query_base(self, request: str):
        context = "".join(self.get_document_content(request=request)["documents"][0])
        system_prompt = {
            "role": "system",
            "content": """
                "You are a knowledgeable assistant for the RAG system." 
                "Your role is to help the student understand and engage with the provided study materials." 
                "You will be given:"
                "1. **Document Chunks**: Relevant excerpts from the study material retrieved from a vector database."
                "2. **User's Request**: The current question or statement from the student."

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
        response_format = Response.model_json_schema()
        try:
            response = self.ollama_client.generate_response(messages=[system_prompt, human_prompt], format=response_format)
            return json.loads(response)['answer']
        except ConnectionError as e:
            self.logger.error(f"Connection error occurred: {e}")
            return f"ConnectionError: {e}"
        except ValueError as e:
            self.logger.error(f"Invalid parameter error: {e}")
            return f"ValueError: {e}"
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            return f"Exception: {e}"
