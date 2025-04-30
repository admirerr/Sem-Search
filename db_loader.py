import pandas as pd
import uuid
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re

client = QdrantClient(host="localhost", port=6333)


df = pd.read_csv("data.csv")


model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')


collection_name = "products"


try:
    client.delete_collection(collection_name=collection_name)
except Exception as e:
    print(f"No existing collection to delete: {e}")

client.recreate_collection(
    collection_name=collection_name,
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    },
)

chunk_size = 1000

points = []

for idx, row in df.iterrows():
    first_word = extract_first_word(row['Product_Name'])
    
    point = {
        "id": str(uuid.uuid4()),
        "vector": model.encode(f"{row['Product_ID']} {row['Product_Name']}").tolist(),
        "payload": {
            "ID": row['ID'],
            "Name": row['Name'],
            "Description": row.get('description', None),
        }
    }

    points.append(point)

    if len(points) >= chunk_size:
        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Uploaded {idx + 1} points...")
        points = []

# Upload any remaining
if points:
    client.upsert(collection_name=collection_name, points=points)
    print("✅ Uploaded remaining points!")

print("🎯 Done updating payloads safely without re-embedding!")
