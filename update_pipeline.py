import os
import requests
import zipfile
import shutil

# --- Configuration ---
URL = "https://cricsheet.org/downloads/recently_added_2_male_json.zip"
ZIP_FILENAME = "temp_latest_data.zip"
EXTRACT_FOLDER = "temp_extracted"
DATA_FOLDER = "data"

print("\n🚀 Starting the Automated Cricket Data Pipeline...")

# 1. Ensure the main data folder exists so we don't get errors
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    print(f"📁 Created '{DATA_FOLDER}' folder.")

# 2. Download the latest ZIP file directly from Cricsheet
print(f"⬇️ Downloading latest matches from Cricsheet...")
response = requests.get(URL)

if response.status_code == 200:
    with open(ZIP_FILENAME, "wb") as file:
        file.write(response.content)
    print("✅ Download complete.")
else:
    print(f"❌ Failed to download data. Status code: {response.status_code}")
    exit()

# 3. Extract the ZIP file into a temporary folder
print("📦 Unzipping files...")
with zipfile.ZipFile(ZIP_FILENAME, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_FOLDER)

# 4. Check for new matches without caring about the specific file names
print("🔍 Checking for new matches...")
new_files_count = 0

for filename in os.listdir(EXTRACT_FOLDER):
    if filename.endswith(".json"):
        source_path = os.path.join(EXTRACT_FOLDER, filename)
        destination_path = os.path.join(DATA_FOLDER, filename)
        
        # Check if we already have this specific file
        if not os.path.exists(destination_path):
            shutil.move(source_path, destination_path)
            print(f"   ➕ New match found & added: {filename}")
            new_files_count += 1

if new_files_count == 0:
    print("🤷 No new matches played recently. Your database is already up to date!")
else:
    print(f"🎉 Successfully added {new_files_count} new match(es) to your data folder.")

# 5. Clean up the temporary junk so your project stays clean
print("🧹 Cleaning up temporary files...")
if os.path.exists(ZIP_FILENAME):
    os.remove(ZIP_FILENAME)
if os.path.exists(EXTRACT_FOLDER):
    shutil.rmtree(EXTRACT_FOLDER)

# 6. The Magic Step: Automatically update the database!
if new_files_count > 0:
    print("\n🧠 Waking up the Database Builder to ingest the new data...")
    # This automatically runs your other Python script!
    os.system("python build_database.py")
    
print("\n✅ Pipeline Finished! Your AI is fully up to date.")