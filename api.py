import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Stable LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# 1. Initialize the FastAPI App
app = FastAPI(title="Cricket AI Analyst API")

# Allow your Vercel frontend to talk to this Render backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Set up the AI
load_dotenv()

# Safety check for database directory
db_path = "./chroma_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

# Initialize Embeddings and Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize LLM (Ensure GROQ_API_KEY is in Render Env Variables)
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

# Expert Prompt
system_prompt = (
    "You are a hyper-niche, expert cricket sports analyst. "
    "Use the following pieces of retrieved context AND the chat history to answer the user's question. "
    "\n\nChat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "If you don't know the answer, say you don't have that data yet. "
    "Write like an energetic sports commentator."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Build the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 3. Data Models
class Message(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []

# 4. Endpoints
@app.post("/ask")
async def ask_analyst(request: ChatRequest):
    # Convert list of messages into a string for the AI
    history_string = ""
    for msg in request.history:
        role = "User" if msg.sender == "user" else "AI Analyst"
        history_string += f"{role}: {msg.text}\n"
    
    # Generate response
    response = rag_chain.invoke({
        "input": request.question,
        "chat_history": history_string
    })
    
    return {"answer": response["answer"]}

@app.get("/")
async def root():
    return {"status": "AI Server is running live!"}