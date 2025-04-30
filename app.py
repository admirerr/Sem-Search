import logging
import traceback
import re
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

app = Flask(__name__)


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(message)s'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


client = QdrantClient(host="localhost", port=6333)
model  = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

def normalize_text(s: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()

@app.route("/search", methods=["GET"])
def search():
    try:
       

        for bucket in (exact_hits, prefix_hits, substr, rest):
            bucket.sort(key=lambda h: h.score, reverse=True)

        final = (exact_hits + prefix_hits + substr + rest)[:10]

        
        results = [{
            "ID":   h.payload.get("ID"),
            "Name": h.payload.get("Name"),
            "Description":    h.payload.get("Description"),
            "score":        h.score
        } for h in final]

        return jsonify({"status": True, "results": results}), 200

    except Exception:
        tb = traceback.format_exc()
        app.logger.error("Search error:\n%s", tb)
        return jsonify({"status": False, "error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal(e):
    app.logger.error("Unhandled exception: %s", e)
    return jsonify({"status": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9876)
