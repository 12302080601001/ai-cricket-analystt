from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the exact same embedding model we used to build the database
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to your existing local database
print("Connecting to ChromaDB...")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. The Question! (Feel free to change the city to one you know is in your recent data)
query = "Which matches were played in Mumbai recently?" 
print(f"\nSearching database for: '{query}'")

# 4. Perform the semantic search (k=2 means "give me the top 2 best results")
results = vectorstore.similarity_search(query, k=2)

# 5. Display the findings
print("\n--- Search Results ---")
if len(results) == 0:
    print("No relevant matches found.")
else:
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Match Summary: {doc.page_content}")
        print(f"Source File: {doc.metadata['source']}")