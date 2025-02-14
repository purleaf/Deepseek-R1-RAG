# Ollama RAG Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.70%2B-green)
![Docker](https://img.shields.io/badge/Docker-ready-brightgreen)
![Ollama](https://img.shields.io/badge/Ollama-powered-purple)
![DeepSeek](https://img.shields.io/badge/DeepSeek-Reasoning_Model-orange)

Welcome to the **Ollama RAG Project**! This repository demonstrates a Retrieval Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Ollama. The system supports intelligent document retrieval, chat history management, and advanced reasoning with the **deepseek-r1** model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Docker Compose](#docker-compose)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Ollama RAG Project is designed to assist students in engaging with their study materials by combining vector-based retrieval with AI-generated responses. It uses a combination of services including:
- **Document Ingestion & Chunking:** Automatically splits and indexes documents for efficient retrieval.
- **Chat History Management:** Maintains the context of past interactions to enhance conversational AI capabilities.
- **Advanced Reasoning:** Integrates with a reasoning model (**deepseek-r1**) for improved response accuracy.

## Features

- **Intelligent Document Retrieval:** Utilizes ChromaDB to store and retrieve document embeddings.
- **Chat History Handling:** Keeps track of previous messages to provide context-aware responses.
- **Multi-Modal AI Responses:** Uses Ollama's local LLMs and OpenAI's GPT models.
- **Special Reasoning with deepseek-r1:** Enhances retrieval and response generation through advanced reasoning capabilities.
- **API Endpoints:** Exposes endpoints for document addition, query answering, chat interaction, translation and summarization.

## Architecture

The project is organized into several services:

- **rag_service.py:** Handles document processing, chunking, and retrieval using vector databases.
- **ask_ai.py:** Manages the interaction with AI models (both local and OpenAI) for chat and query responses.
- **main.py:** FastAPI server exposing the RESTful endpoints.
- **Docker Compose:** Orchestrates the FastAPI app and Ollama container for a seamless deployment.

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/purleaf/local_rag.git
   cd ollama-rag-project
   ```

2. **Install Requirements:**
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create the `.env` File:**
   Create a `.env` file in the root directory with necessary environment variables (e.g., `OPENAI_API_KEY` and any other required configs).

4. **Docker Setup (Optional but Recommended):**
   - Make sure you have [Docker](https://www.docker.com/get-started) installed.
   - The `docker-compose.yaml` file is provided to run both the AI server and the Ollama container.

## Running the Server

You can run the FastAPI server locally using `uvicorn`:

```bash
uvicorn main:app --host 0.0.0.0 --port 5005
```

Or, run it via Docker Compose:

```bash
docker-compose up --build
```

## API Endpoints

### Add Document

- **Endpoint:** `/api/add_document`
- **Method:** `POST`
- **Query Parameters:**
  - `user_id`: User identifier
  - `document_id`: Document identifier
  - `document`: The document content
- **Description:** Adds a document to the vector database after processing and chunking.

### Ask AI

- **Endpoint:** `/api/ask_ai`
- **Method:** `GET`
- **Query Parameters:**
  - `user_id`
  - `document_id`
  - `request`: The user's query
- **Description:** Retrieves a response from the AI based on document content.

### AI Chat

- **Endpoint:** `/api/ai_chat`
- **Method:** `GET`
- **Query Parameters:**
  - `user_id`
  - `document_id`
  - `request`: The user's message
- **Description:** Engages in a conversation with the AI, maintaining chat history.

### Translate

- **Endpoint:** `/api/translate`
- **Method:** `GET`
- **Query Parameters:**
  - `user_id`
  - `document_id`
  - `request`: The text to translate
  - `language`: Target language code
- **Description:** Translates text into the specified language.


### Summarize

- **Endpoint:** `/api/summarize`
- **Method:** `GET`
- **Query Parameters:**
  - `user_id`
  - `document_id`
  - `text`: The text to summarize
- **Description:** Summarizes the provided text.

## Docker Compose

The `docker-compose.yaml` file sets up two services:

- **ai:** Builds and runs the FastAPI server.
- **ollama:** Runs the Ollama container for local LLM interactions.

```yaml
services:
  ai:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "2052:5005"
    environment:
      - PORT=5005
    env_file: ".env"
  ollama:
    image: ollama/ollama
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:11434" ]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 5s
    restart: always

volumes:
  ollama-data:
```

## Contributing

Contributions are welcome! Please open issues and submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

---

Happy Coding! 🚀
