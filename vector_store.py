import faiss
import pickle
import numpy as np
import os
# UPDATED IMPORT: Using the newer, non-deprecated library
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_vector_db(chunks):
    print("🧠 Step 2: Initializing Embedding Model...")
    # This will now use the updated langchain-huggingface library
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"🔢 Step 3: Converting {len(chunks)} chunks into vectors...")
    embeddings = embeddings_model.embed_documents(chunks)

    embeddings_np = np.array(embeddings).astype('float32')

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    faiss.write_index(index, "vector_index.bin")

    with open("metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Success! 'vector_index.bin' and 'metadata.pkl' created.")


if __name__ == "__main__":
    from ingestion import load_and_chunk_docs

    data_chunks = load_and_chunk_docs("data")

    if data_chunks:
        create_vector_db(data_chunks)