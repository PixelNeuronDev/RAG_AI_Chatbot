🤖 Local RAG Chatbot: PDF Intelligence System
📌 Project Overview
This project is a high-performance Retrieval-Augmented Generation (RAG) system built to provide accurate, context-aware answers from local PDF documents. Unlike standard LLMs, this system uses a dedicated "knowledge base" to eliminate hallucinations and provide specific technical details from academic or professional documents.

🛠️ Technical Stack
Language: Python 3.11+
Vector Database: FAISS (Facebook AI Similarity Search)
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional vectors)
LLM Engine: Ollama (Running Llama 3.2 locally)
Framework: LangChain (HuggingFace Integration)

🏗️ System Architecture
The system operates in two distinct phases:
1. The Ingestion Phase (Data Processing Cycle)
2. Document Parsing: Extracts raw text from PDFs.
3. Recursive Chunking: Breaks text into manageable pieces for better semantic capture.
4. Vectorization: Converts text chunks into 384-dimensional numerical embeddings using a neural network.
5. Indexing: Stores vectors in a FAISS .bin file for $O(\log n)$ search speed.

2.The Retrieval & Generation Phase
  Semantic Search: Converts the user query into a vector and calculates the Euclidean Distance (L2) against the database.
  Context Augmentation: Retrieves the top $k$ most relevant chunks and injects them into a specialized prompt.
  Local Inference: The augmented prompt is sent to Llama 3.2 via Ollama to generate a natural language response based only on the provided context.

🚀 Key Features
  100% Local & Private: No data leaves your machine; no API keys (OpenAI/Anthropic) required.
  Hallucination Guardrails: The AI is instructed to only answer based on the provided document.
  Efficient Search: Capable of searching through thousands of document chunks in milliseconds using FAISS.

📋 Prerequisites
  Ollama installed and running.
  Llama 3.2 model pulled: ollama pull llama3.2.
  Python dependencies: pip install langchain-huggingface faiss-cpu ollama pickle5 pypdf

📂 File Structure
  ingestion.py: Processes the PDF and builds the Vector DB.
  retriever.py: Handles the similarity search logic.
  chatbot.py: The main execution loop and LLM integration.
vector_index.bin: The mathematical map of the PDF.
metadata.pkl: The raw text associated with the vectors.
