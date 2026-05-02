import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Modern, Stable LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize App
app = FastAPI(title="Cricket AI Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

db_path = "./chroma_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

system_prompt = (
    "You are a hyper-niche, expert cricket sports analyst. "
    "Use the following pieces of retrieved context AND the chat history to answer the user's question.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "If you don't know the answer, say you don't have that data yet. "
    "Write like an energetic sports commentator."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class Message(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []

@app.post("/ask")
async def ask_analyst(request: ChatRequest):
    docs = retriever.invoke(request.question)
    context_text = format_docs(docs)
    
    history_string = ""
    for msg in request.history:
        role = "User" if msg.sender == "user" else "AI Analyst"
        history_string += f"{role}: {msg.text}\n"
    
    # The Modern Chain approach
    chain = prompt | llm | StrOutputParser()
    
    response_text = chain.invoke({
        "input": request.question,
        "chat_history": history_string,
        "context": context_text
    })
    
    return {"answer": response_text}

@app.get("/")
async def root():
    return {"status": "AI Server is running live!"}