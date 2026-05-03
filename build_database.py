import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# --- THE MAGIC WAND ---
# This looks for your hidden .env file and secretly loads your API keys!
load_dotenv()

# 2. Load the embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

folder_path = "data" 
documents = []

print("Reading JSON files and extracting ball-by-ball player stats...")

# 3. Loop through the JSON files
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            match_data = json.load(file)
            
            # Extract basic info
            info = match_data.get('info', {})
            teams = info.get('teams', ['Unknown', 'Unknown'])
            city = info.get('city', info.get('venue', 'Unknown Venue'))
            date = info.get('dates', ['Unknown Date'])[0]
            
            # Start writing a detailed match report
            match_report = f"Match Details: {teams[0]} vs {teams[1]} played on {date} in {city}.\n"
            
            # 4. The Deep Dive: Extract ball-by-ball stats
            if 'innings' in match_data:
                for inning in match_data['innings']:
                    team_name = inning.get('team', 'Unknown Team')
                    total_runs = 0
                    total_wickets = 0
                    batter_stats = {}
                    bowler_stats = {}
                    
                    # Loop through every over and delivery
                    for over in inning.get('overs', []):
                        for delivery in over.get('deliveries', []):
                            # Tally team totals
                            total_runs += delivery.get('runs', {}).get('total', 0)
                            
                            # Tally batter runs
                            batter = delivery.get('batter')
                            batter_runs = delivery.get('runs', {}).get('batter', 0)
                            batter_stats[batter] = batter_stats.get(batter, 0) + batter_runs
                            
                            # Tally bowler wickets
                            if 'wickets' in delivery:
                                total_wickets += len(delivery['wickets'])
                                bowler = delivery.get('bowler')
                                bowler_stats[bowler] = bowler_stats.get(bowler, 0) + len(delivery['wickets'])
                    
                    # 5. Add the stats to our match report text
                    match_report += f"\n{team_name} Innings Summary: {total_runs} runs for {total_wickets} wickets.\n"
                    
                    match_report += f"Batting for {team_name}: "
                    match_report += ", ".join([f"{player} scored {runs} runs" for player, runs in batter_stats.items()]) + ".\n"
                    
                    match_report += f"Bowling against {team_name}: "
                    match_report += ", ".join([f"{player} took {w} wickets" for player, w in bowler_stats.items()]) + ".\n"
            
            # Create a LangChain Document
            doc = Document(
                page_content=match_report, 
                metadata={"source": filename, "date": date}
            )
            documents.append(doc)

print(f"Successfully prepared {len(documents)} deep-dive match documents!")

# 6. Save the upgraded data to PINECONE!
print("Pushing data to Pinecone cloud...")
index_name = "cricket-db"

vectorstore = PineconeVectorStore.from_documents(
    documents=documents, 
    embedding=embeddings,
    index_name=index_name
)

print("Boom! Database built successfully with player stats uploaded to Pinecone! 🏏")