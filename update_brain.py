import os
import requests
import zipfile
import shutil

URL = "https://cricsheet.org/downloads/recently_added_2_json.zip"
ZIP_FILENAME = "temp_latest_data.zip"
EXTRACT_FOLDER = "temp_extracted"
DATA_FOLDER = "data"

print("\n🚀 Starting the Automated Cricket Data Pipeline...")

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    print(f"📁 Created '{DATA_FOLDER}' folder.")

print(f"⬇️ Downloading latest matches from Cricsheet...")
response = requests.get(URL)

if response.status_code == 200:
    with open(ZIP_FILENAME, "wb") as file:
        file.write(response.content)
    print("✅ Download complete.")
else:
    print(f"❌ Failed to download data. Status code: {response.status_code}")
    exit()

print("📦 Unzipping files...")
with zipfile.ZipFile(ZIP_FILENAME, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_FOLDER)

print("🔍 Checking for matches...")
new_files_count = 0

for filename in os.listdir(EXTRACT_FOLDER):
    if filename.endswith(".json") and filename != "README.txt":
        source_path = os.path.join(EXTRACT_FOLDER, filename)
        destination_path = os.path.join(DATA_FOLDER, filename)
        
        shutil.move(source_path, destination_path)
        print(f"   ➕ Match added for processing: {filename}")
        new_files_count += 1

print("🧹 Cleaning up temporary files...")
if os.path.exists(ZIP_FILENAME):
    os.remove(ZIP_FILENAME)
if os.path.exists(EXTRACT_FOLDER):
    shutil.rmtree(EXTRACT_FOLDER)

if new_files_count > 0:
    print("\n🧠 Waking up the Database Builder to ingest the data...")
    os.system("python build_database.py")
    
print("\n✅ Pipeline Finished! Your AI is fully up to date.")