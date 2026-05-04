import os
import requests
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

def fetch_and_update():
    print("🏏 Starting Daily Cricket Data Pipeline...")
    
    # 1. Extract: (Replace this URL with your actual free Cricket API if you have one)
    # For now, we simulate fetching today's match result
    match_summary = "Latest Match: India beat Australia by 6 wickets in a thrilling chase."
    
    print("🧠 Loading AI Model...")
    # 2. Transform: Convert text to vectors
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector = model.encode(match_summary).tolist()
    
    print("☁️ Pushing to Pinecone...")
    # 3. Load: Push to your Pinecone index
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("cricket-index") # Make sure this matches your exact Pinecone index name!
    
    # We use a unique ID for each daily update (like the date)
    import datetime
    today_id = f"match_{datetime.date.today().strftime('%Y%m%d')}"
    
    index.upsert(vectors=[{"id": today_id, "values": vector, "metadata": {"text": match_summary}}])
    print("✅ Successfully updated AI Brain with latest match!")

if __name__ == "__main__":
    fetch_and_update()