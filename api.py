import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Cricket AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# We use a simple cache to hold the AI in memory
ai_cache = {}

def get_ai_chain():
    if "chain" not in ai_cache:
        print("Loading AI Brain...")
        
        # Setup LLM - Direct connection to Groq
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.6)
        
        # Create a simple conversational prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a hyper-niche, expert cricket sports analyst. "
                       "You know everything about cricket history and stats. "
                       "Always write like an energetic, passionate sports commentator!"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Chain the prompt directly to the LLM (No tools, no agents!)
        chain = prompt | llm
        
        ai_cache["chain"] = chain
        print("AI loaded and ready to answer!")
    
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
    
    # Invoke the chain directly!
    response = chain.invoke({
        "input": request.question,
        "chat_history": langchain_history
    })
    
    # Because it's a direct LLM chain, the answer is in response.content
    return {"answer": response.content}

@app.get("/")
async def root():
    return {"status": "AI Live Agent Server is running!"}