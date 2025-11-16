# RAG-Chain-App

A simple Retrieval-Augmented Generation (RAG) application built with **LangChain** and a vector store, allowing users to ask questions about their own documents and get meaningful, context-grounded responses. Requires a GPU as Ollama is used to launch a an instance of Llama-3 on your own hardware.

---

## Features

- Ingests documents from a local source  
- Splits documents into manageable chunks for efficient retrieval  
- Uses embeddings to vectorize document chunks and store them in a vector database  
- Retrieves relevant context based on a user query  
- Passes retrieved context + user question to a large language model (LLM)  
- Generates a response grounded in the retrieved context

---

## Setup Instructions:

1. Ensure the correct packages are installed: api/requirements.txt
2. Install Ollama and install the Llama-3 Model as well as the Nomic Text Embed model.
3. Ensure Ollama is running.
4. Cd into the api directory and run ```uvicorn app:main --reload``` to start the fastAPI endpoints
5. Then, navigate into the app directory and run ```streamlit run streamlit_app.py```
