📚 FastAPI RAG Application (Retrieval-Augmented Generation)

This project is a Retrieval-Augmented Generation (RAG) API built with FastAPI, Pinecone (vector database), and a text embedding model (OpenAI, Amazon Bedrock Titan, or Llama embeddings).

It allows you to:

Upload text/PDF files

Split them into chunks

Create embeddings

Store them in Pinecone namespaces

Query your documents using semantic search

Generate AI answers grounded in your uploaded knowledge


Features

FastAPI backend with async endpoints

RAG pipeline: Chunk → Embed → Store → Retrieve

Dynamic Pinecone namespaces (per-user, per-project, per-file)

Semantic search using Pinecone vectors

LLM-powered answer generation (OpenAI, Bedrock, or other model)

Upload support: TXT, PDF, Markdown

Clean architecture for production



🏗️ Tech Stack
Component	Technology
Backend	FastAPI
Vector DB	Pinecone
Embedding Model	Llama / OpenAI / Titan
LLM for Answers	OpenAI GPT, Claude, or custom
File Parsing	PyPDF2 / plaintext
Environment	Python 3.11+


🔧 Environment Variables

CHATGPT_API_KEY = awdawd
PINECONE_CLOUD = gcp
PINECONE_REGION = europe
PINECONE_API_KEY = apwdomapwod