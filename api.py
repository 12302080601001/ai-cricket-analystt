import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# NEW AGENT IMPORTS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_tool_calling_agent, AgentExecutor

app = FastAPI(title="Cricket AI Agent API")

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

ai_cache = {}

def get_ai_agent():
    if "agent_executor" not in ai_cache:
        print("Loading AI Agent and equipping tools...")
        
        # 1. Load Embeddings (Using API to prevent Render crashes)
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.environ.get("HF_TOKEN")
        )
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 2. CREATE TOOL 1: The Historical Database
        retriever_tool = create_retriever_tool(
            retriever,
            "historical_cricket_data",
            "Search this tool FIRST for past cricket matches, historical stats, player info, and rules."
        )
        
        # 3. CREATE TOOL 2: Live Internet Search
        search_tool = DuckDuckGoSearchRun(
            name="live_internet_search",
            description="Search the internet for LIVE cricket scores, breaking news, or matches happening today."
        )
        
        tools = [retriever_tool, search_tool]
        
        # 4. Setup LLM (Temperature 0 is best for agents so they think logically)
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        
        # 5. Create the Agent Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a hyper-niche, expert cricket sports analyst. "
                       "You have access to tools. Use them to answer the user's questions! "
                       "If the question is about historical data, use the historical_cricket_data tool. "
                       "If it is about a live match or current news, use the live_internet_search tool. "
                       "Always write like an energetic, passionate sports commentator!"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 6. Build the Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # verbose=True lets you see the agent's thought process in the Render logs!
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        ai_cache["agent_executor"] = agent_executor
        print("AI Agent loaded and ready to answer!")
    
    return ai_cache["agent_executor"]

class Message(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []

@app.post("/ask")
async def ask_analyst(request: ChatRequest):
    agent_executor = get_ai_agent()
    
    # Convert your frontend history into LangChain's memory format
    langchain_history = []
    for msg in request.history:
        if msg.sender == "user":
            langchain_history.append(HumanMessage(content=msg.text))
        else:
            langchain_history.append(AIMessage(content=msg.text))
    
    # Invoke the agent! It will automatically think, pick a tool, search, and answer.
    response = agent_executor.invoke({
        "input": request.question,
        "chat_history": langchain_history
    })
    
    return {"answer": response["output"]}

@app.get("/")
async def root():
    return {"status": "AI Live Agent Server is running!"}