import ollama
from retriever import retrieve_relevant_chunks


def generate_answer(question):
    # 1. Retrieve the Context (Phase 2, Step 2)
    print("🔍 Searching for relevant information in the PDF...")
    context_chunks = retrieve_relevant_chunks(question, k=3)

    # 2. Build the Context String
    # We join the 3 chunks into one big block of text
    formatted_context = "\n\n".join(context_chunks)

    # 3. Create the Prompt (Phase 2, Step 3)
    # This tells the LLM exactly how to behave.
    prompt = f"""
    You are a helpful AI assistant. Use the provided context from a PDF to answer the user's question.
    If the answer is not in the context, say that you don't know based on the provided document.

    CONTEXT:
    {formatted_context}

    USER QUESTION:
    {question}

    ANSWER:
    """

    # 4. Send to Ollama (Phase 2, Step 4)
    print("🤖 AI is thinking...")
    response = ollama.generate(model="llama3.2", prompt=prompt)

    return response['response']


if __name__ == "__main__":
    while True:
        query = input("\n💬 Ask your NumPy Chatbot (or type 'exit'): ")
        if query.lower() == 'exit':
            break

        answer = generate_answer(query)
        print(f"\n✨ AI Response:\n{answer}")