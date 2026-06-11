from pathlib import Path
import os
import shutil
import sys
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.pipeline.rag_pipeline import RAGPipeline

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
FAISS_DIR = BASE_DIR / "vectorstore" / "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

pipeline = RAGPipeline()
try:
    pipeline.load()
except Exception:
    pass


def reset_index() -> None:
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
    pipeline.vectorstore = None
    pipeline.all_chunks = []
    pipeline.retriever = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    saved_paths = []
    for f in files:
        if not f:
            continue
        filename = secure_filename(f.filename)
        dest = Path(app.config["UPLOAD_FOLDER"]) / filename
        f.save(dest)
        saved_paths.append(str(dest))

    try:
        reset_index()
        pipeline.ingest(str(UPLOAD_DIR), update=False)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "saved": saved_paths, "chunk_count": len(pipeline.all_chunks)})


@app.route("/retrieve", methods=["POST"])
def retrieve():
    data = request.get_json() or {}
    query = data.get("query") or data.get("question")
    if not query:
        return jsonify({"ok": False, "error": "Missing query"}), 400

    try:
        if not pipeline.retriever:
            try:
                pipeline.load()
            except Exception:
                pass

        result = pipeline.retrieve(query, verbose=True)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/chunks", methods=["GET"])
def chunks():
    """Return all ingested chunks (for downstream agent consumption)."""
    try:
        if not pipeline.all_chunks:
            try:
                pipeline.load()
            except Exception:
                pass
        return jsonify({"ok": True, "chunks": pipeline.get_all_chunks()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
