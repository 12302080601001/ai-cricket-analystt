import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- THE MAGIC WAND ---
# This looks for your hidden .env file and secretly loads your API keys into memory!
load_dotenv()
# ADD THIS LINE JUST FOR TESTING:
print("🕵️‍♂️ SECRETS TEST. Did I find Pinecone?", os.getenv("PINECONE_API_KEY"))

app = FastAPI(title="Cricket AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We use a simple cache to hold the AI in memory
ai_cache = {}

def get_ai_chain():
    if "chain" not in ai_cache:
        print("Loading AI Brain and connecting to Pinecone Cloud...")
        
        # Setup LLM - Connection to Groq
        # Groq will automatically find GROQ_API_KEY from the .env file
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.6)
        
        # --- THE NEW RAG UPGRADE ---
        # Pinecone will automatically find PINECONE_API_KEY from the .env file
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = PineconeVectorStore(index_name="cricket-db", embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Fetch top 5 relevant match chunks
        
        # Create a conversational RAG prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a hyper-niche, expert cricket sports analyst. "
                       "You have access to detailed match data in the Context below. "
                       "Use this Context to answer the user's questions accurately. "
                       "Always write like an energetic, passionate sports commentator!\n\n"
                       "Context from database:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Chain the retriever and the LLM together
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        ai_cache["chain"] = rag_chain
        print("AI loaded and connected to the cloud!")
    
    return ai_cache["chain"]

class Message(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []

@app.post("/ask")
async def ask_analyst(request: ChatRequest):
    chain = get_ai_chain()
    
    # Convert your frontend history into LangChain's memory format
    langchain_history = []
    for msg in request.history:
        if msg.sender == "user":
            langchain_history.append(HumanMessage(content=msg.text))
        else:
            langchain_history.append(AIMessage(content=msg.text))
    
    # Invoke the RAG chain!
    response = chain.invoke({
        "input": request.question,
        "chat_history": langchain_history
    })
    
    # In a retrieval chain, the final LLM response is stored under the "answer" key
    return {"answer": response["answer"]}

@app.get("/")
async def root():
    return {"status": "AI Live Agent Server is running, powered by Pinecone Cloud!"}