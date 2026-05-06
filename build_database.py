import os
import json
from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Groq Client!
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

folder_path = "data" 
documents = []

print("Reading JSON files and generating Groq summaries...")

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            match_data = json.load(file)
            
            info = match_data.get('info', {})
            teams = info.get('teams', ['Unknown', 'Unknown'])
            city = info.get('city', info.get('venue', 'Unknown Venue'))
            date = info.get('dates', ['Unknown Date'])[0]
            
            match_report = f"Match Details: {teams[0]} vs {teams[1]} played on {date} in {city}.\n"
            
            if 'innings' in match_data:
                for inning in match_data['innings']:
                    team_name = inning.get('team', 'Unknown Team')
                    total_runs = 0
                    total_wickets = 0
                    
                    for over in inning.get('overs', []):
                        for delivery in over.get('deliveries', []):
                            total_runs += delivery.get('runs', {}).get('total', 0)
                            if 'wickets' in delivery:
                                total_wickets += len(delivery['wickets'])
                    
                    match_report += f"{team_name} scored {total_runs}/{total_wickets}.\n"
            
            # --- THE GROQ MAGIC ---
            print(f"🤖 Asking Groq to summarize match: {filename}...")
            prompt = f"You are an expert cricket commentator. Summarize this raw match data into a short, exciting paragraph for an AI database: {match_report}"
            
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                )
                groq_summary = chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Groq API Error: {e}")
                groq_summary = match_report # Fallback if Groq fails
            
            # Save the Groq summary to our document list
            doc = Document(
                page_content=f"AI Match Summary: {groq_summary}\n\nRaw Stats: {match_report}", 
                metadata={"source": filename, "date": date}
            )
            documents.append(doc)

print(f"Successfully prepared {len(documents)} deep-dive match documents!")

print("Pushing data to Pinecone cloud...")
index_name = os.getenv("PINECONE_INDEX_NAME", "cricket-index")

# 🛡️ THE MAGIC SHIELD: Extract match IDs to prevent duplicates!
match_ids = [doc.metadata["source"].replace(".json", "") for doc in documents]

vectorstore = PineconeVectorStore.from_documents(
    documents=documents, 
    embedding=embeddings,
    index_name=index_name,
    ids=match_ids 
)

print("Boom! Database built successfully with Groq summaries uploaded to Pinecone! 🏏")