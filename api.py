import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 1. Initialize the FastAPI App
app = FastAPI(title="Cricket AI Analyst API")

# Allow React to communicate with this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Set up the AI
print("Starting up the AI Server...")
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

# System Prompt with Memory
system_prompt = (
    "You are a hyper-niche, expert cricket sports analyst. "
    "Use the following pieces of retrieved context AND the chat history to answer the user's question. "
    "If the user asks a follow-up question (like 'how many runs did he score?' or 'did they win?'), use the chat history to figure out who they are talking about.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "If you don't know the answer based on the data, just say that you don't have that data yet. "
    "Write it like an energetic, professional sports commentator."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 3. Define the incoming data (includes question AND history)
class Message(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []

# 4. Create the Web Endpoint
@app.post("/ask")
async def ask_analyst(request: ChatRequest):
    print(f"Received question: {request.question}")
    
    # Format the history into a string the AI can read
    history_string = ""
    for msg in request.history:
        role = "User" if msg.sender == "user" else "AI Analyst"
        history_string += f"{role}: {msg.text}\n"
    
    # Send both the new question AND the history to the AI
    response = rag_chain.invoke({
        "input": request.question,
        "chat_history": history_string
    })
    
    return {"answer": response["answer"]}

# 5. Simple health check
@app.get("/")
async def root():
    return {"status": "AI Server is running live!"}