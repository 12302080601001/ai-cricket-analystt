import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 1. Load the secret API key from the .env file
load_dotenv()

# 2. Connect to your local ChromaDB (The Memory)
print("Waking up the database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Set up the database as a "retriever" (it will fetch the top 3 matches)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Connect to the AI Model (The Brain)
print("Connecting to Llama-3 AI...")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

# 4. Give the AI its personality and instructions
system_prompt = (
    "You are a hyper-niche, expert cricket sports analyst. "
    "Use the following pieces of retrieved context to answer the user's question. "
    "If you don't know the answer based on the context, just say that you don't have that data yet. "
    "Keep your answer concise but write it like a professional sports commentator.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# 5. Build the RAG Chain (Connects Retriever -> Prompt -> LLM)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Create an interactive chat loop!
print("\n" + "="*50)
print("🎙️ The AI Sports Analyst is Live! (Type 'quit' to exit)")
print("="*50)

while True:
    # This waits for you to type a custom question in the terminal
    user_question = input("\nAsk a question: ")
    
    # If you type 'quit' or 'exit', it breaks the loop and stops the script
    if user_question.lower() in ['quit', 'exit']:
        print("Thanks for tuning in! See you next time. 🏏")
        break
        
    print("Thinking...")
    
    # Run the chain with your custom input!
    response = rag_chain.invoke({"input": user_question})
    
    print("\n--- AI Analyst ---")
    print(response["answer"])