from fastapi import FastAPI, HTTPException, Form, Query

from ai.services.ask_ai_service import AskAIService
from ai.services.quiz_gen import QuizGeneration
from ai.services.translate_service import TranslateService
from ai.services.summarize_service import SummarizeService
from ai.services.rag_service import VectorDBManager
from uuid_shortener import UUIDShortener
from utils.db_helper import init_db, add_message, get_chat_history
from utils.logger import Logger
from ollama import Client
app = FastAPI()
logger = Logger("RAG-DB")


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/api")
async def root():
    return {"message": "Hello World"}


@app.post("/api/add_document")
async def add_document(
    user_id: str = Query(...), document_id: str = Query(...), document: str = Query(...)
):
    try:
        collection_name = UUIDShortener.encode(f"{user_id}{document_id}")
        vectordb_manager = VectorDBManager(collection_name=collection_name)
        vectordb_manager.add_document(
            content=document, document_id=document_id, user_id=user_id
        )
        return {"message": "Document added successfully"}
    except ValueError as ve:
        logger.error(f"{ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"{e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ask_ai")
async def ask_ai(
    user_id: str = Query(...), document_id: str = Query(...), request: str = Query(...)
):
    try:
        collection_name = UUIDShortener.encode(f"{user_id}{document_id}")
        rag_db = VectorDBManager(collection_name=collection_name)
        completion = rag_db.answer_query_base(request)
        return completion
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/api/ai_chat")
async def chat_with_ai(
    user_id: str = Query(...), document_id: str = Query(...), request: str = Query(...)
):
    try:
        previous_messages = get_chat_history(user_id=user_id)
        add_message(user_id=user_id, role="user", content=request)
        collection_name = UUIDShortener.encode(f"{user_id}{document_id}")
        ask_ai_manager = AskAIService(collection_name=collection_name)
        completion = ask_ai_manager.local_model(request=request, chat_history=previous_messages)
        add_message(user_id=user_id, role="assistant", content=completion)
        return completion
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) or "An unexpected error occurred")


@app.get("/api/translate")
async def translate(
    user_id: str = Query(...),
    document_id: str = Query(...),
    request: str = Query(...),
    language: str = Query(...),
):
    try:
        collection_name = UUIDShortener.encode(f"{user_id}{document_id}")
        translate_manager = TranslateService(collection_name=collection_name)
        completion = translate_manager.translate(text=request, language=language)
        return completion
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/api/quizgen")
async def quiz_gen(
    user_id: str = Query(...),
    document_id: str = Query(...),
    difficulty: str = Query(...),
    n_questions: int = Query(...),
):
    try:
        collection_name = UUIDShortener.encode(f"{user_id}{document_id}")
        quiz_generation = QuizGeneration()
        rag_manager = VectorDBManager(collection_name=collection_name)
        context = " ".join(rag_manager.collection.get()["documents"])
        quiz = quiz_generation.generate_response(
            context=context, difficulty=difficulty, n_questions=n_questions
        )
        return quiz
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/api/summarize")
async def summarize(
    user_id: str = Query(...), document_id: str = Query(...), text: str = Query(...)
):
    try:
        collection_name = UUIDShortener.encode(f"{user_id}{document_id}")
        summary_manager = SummarizeService(collection_name=collection_name)
        completion = summary_manager.summary(request=text)
        return completion
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
