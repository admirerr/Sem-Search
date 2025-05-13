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
        q = (request.args.get("query") or "").strip()
        if not q:
            return jsonify({"status": False, "error": "Query text is required"}), 400

        nq   = normalize_text(q)
        fw   = nq.split()[0] if nq.split() else ""
        vec  = model.encode(q).tolist()

        exact_filter = models.Filter(
            must=[models.FieldCondition(
                key="first_word",
                match=models.MatchValue(value=fw)
            )]
        )
        exact_hits = client.search(
            collection_name="products",
            query_filter=exact_filter,
            query_vector=vec,
            limit=10
        )

        prefix_hits = []
        if len(exact_hits) < 10:
            prefix_filter = models.Filter(
                must=[models.FieldCondition(
                    key="first_word_prefixes",
                    match=models.MatchValue(value=fw)
                )]
            )
            prefix_hits = client.search(
                collection_name="products",
                query_filter=prefix_filter,
                query_vector=vec,
                limit=10 - len(exact_hits)
            )

        remaining = 10 - len(exact_hits) - len(prefix_hits)
        semantic_hits = []
        if remaining > 0:
            candidates = client.search(
                collection_name="products",
                query_vector=vec,
                limit=50
            )
            seen = {h.id for h in exact_hits + prefix_hits}
            for h in candidates:
                if h.id not in seen:
                    semantic_hits.append(h)
                if len(semantic_hits) >= remaining:
                    break
        substr, rest = [], []
        for h in semantic_hits:
            nm = normalize_text(h.payload.get("Name",""))
            if nq in nm:
                substr.append(h)
            else:
                rest.append(h)

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
