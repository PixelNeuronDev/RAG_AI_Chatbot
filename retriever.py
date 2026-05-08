import faiss
import pickle
from langchain_huggingface import HuggingFaceEmbeddings

# Must be the SAME model we used to create the database!
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def retrieve_relevant_chunks(query, k=3):
    # 1. Load the FAISS Index
    index = faiss.read_index("vector_index.bin")

    # 2. Load the actual text chunks
    with open("metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    # 3. Initialize the same Embedding Model
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 4. Turn the user's question into a vector
    # Note: embed_query is used for single strings
    question_vector = embeddings_model.embed_query(query)

    # 5. Search the index
    # FAISS expects a 2D array, so we wrap our vector in a list [question_vector]
    import numpy as np
    distances, indices = index.search(np.array([question_vector]).astype('float32'), k)

    # 6. Get the text from our metadata using the indices found
    results = [chunks[i] for i in indices[0]]
    return results


if __name__ == "__main__":
    user_query = input("🔍 Enter your question about the PDF: ")
    relevant_docs = retrieve_relevant_chunks(user_query)

    print("\n--- 🎯 Top Relevant Chunks Found ---")
    for i, doc in enumerate(relevant_docs):
        print(f"\n[Result {i + 1}]:\n{doc}")
        print("-" * 30)