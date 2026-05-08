import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_docs(data_folder):
    """
    Reads all PDF and TXT files from a folder and splits them into chunks.
    """
    raw_text = ""

    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"❌ Error: Folder '{data_folder}' not found.")
        return []

    files = [f for f in os.listdir(data_folder) if f.endswith(('.pdf', '.txt'))]

    if not files:
        print(f"⚠️ No PDF or TXT files found in '{data_folder}'.")
        return []

    print(f"📂 Found {len(files)} file(s). Starting extraction...")

    for filename in files:
        path = os.path.join(data_folder, filename)

        try:
            if filename.endswith(".pdf"):
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    content = page.extract_text()
                    if content:
                        raw_text += content + "\n"

            elif filename.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    raw_text += f.read() + "\n"

            print(f"✅ Processed: {filename}")
        except Exception as e:
            print(f"❌ Could not read {filename}: {e}")

    # Chunking Logic
    # 500 chars is roughly 2-3 paragraphs. 50 char overlap ensures
    # context isn't lost between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks


if __name__ == "__main__":
    # Test the logic
    folder_path = "data"
    all_chunks = load_and_chunk_docs(folder_path)

    if all_chunks:
        print("\n--- 📊 Extraction Summary ---")
        print(f"Total Chunks Created: {len(all_chunks)}")
        print(f"First Chunk Preview: {all_chunks[0][:150]}...")
        print("----------------------------")