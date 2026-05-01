import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Standard LangChain Imports (Fixed these)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Initialize the FastAPI App
app = FastAPI(title="Cricket AI Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Set up the AI
load_dotenv()

# Safety Check: Use an empty directory if chroma_db isn't found
db_path = "./chroma_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

system_prompt = (
    "You are a hyper-niche, expert cricket sports analyst. "
    "Use the following pieces of retrieved context AND the chat history to answer the user's question. "
    "Chat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "If you don't know the answer, just say you don't have that data yet."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

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
    history_string = ""
    for msg in request.history:
        role = "User" if msg.sender == "user" else "AI Analyst"
        history_string += f"{role}: {msg.text}\n"
    
    response = rag_chain.invoke({
        "input": request.question,
        "chat_history": history_string
    })
    return {"answer": response["answer"]}

@app.get("/")
async def root():
    return {"status": "AI Server is running live!"}