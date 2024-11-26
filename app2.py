import os
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from pymongo import MongoClient
from PIL import Image
import torch
import numpy as np
import clip
from io import BytesIO
from scipy.spatial.distance import cosine

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://localhost:27017/")
db = client["cbir"]
collection = db["image_features"]

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("CLIP model loaded.")

@app.route('/upload', methods=['POST'])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image = request.files["image"]
    original_filename = image.filename
    if original_filename == '':
        return jsonify({"error": "Invalid image name"}), 400
    filename = secure_filename(original_filename)
    image_path = os.path.join("uploads/animal2", filename)
    try:
        image.save(image_path)
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return jsonify({"error": "Failed to save image"}), 500
    try:
        image_input = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy()[0]
        document = {
            "image_name": filename,
            "image_path": os.path.join("animal2", filename),
            "image_embedding": embedding.tolist()
        }
        collection.insert_one(document)
        return jsonify({"message": "Image uploaded and features stored successfully"}), 200
    except Exception as e:
        print(f"Error processing image {filename}: {e}")
        return jsonify({"error": "Failed to process image"}), 500

@app.route('/query_image', methods=['POST'])
def query_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image = request.files["image"]
    try:
        img_stream = BytesIO(image.read())
        img = Image.open(img_stream).convert("RGB")
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query_embedding = image_features.cpu().numpy()[0]
        cursor = collection.find({}, {"image_name": 1, "image_path": 1, "image_embedding": 1, "_id": 0})
        results = []
        for doc in cursor:
            stored_embedding = np.array(doc["image_embedding"])
            similarity = 1 - cosine(query_embedding, stored_embedding)
            results.append({
                "image_name": doc["image_name"],
                "image_path": doc["image_path"],
                "similarity": float(similarity) * 100
            })
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:10]
        return jsonify({"similar_images": results}), 200
    except Exception as e:
        print(f"Error processing query image: {e}")
        return jsonify({"error": "Failed to process query image"}), 500

@app.route('/query_text', methods=['POST'])
def query_text():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400
    query = data["query"]
    try:
        text_input = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            query_embedding = text_features.cpu().numpy()[0]
        cursor = collection.find({}, {"image_name": 1, "image_path": 1, "image_embedding": 1, "_id": 0})
        results = []
        for doc in cursor:
            stored_embedding = np.array(doc["image_embedding"])
            similarity = 1 - cosine(query_embedding, stored_embedding)
            results.append({
                "image_name": doc["image_name"],
                "image_path": doc["image_path"],
                "similarity": float(similarity) * 100
            })
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:10]
        return jsonify({"similar_images": results}), 200
    except Exception as e:
        print(f"Error processing text query: {e}")
        return jsonify({"error": "Failed to process text query"}), 500

@app.route('/uploads/<path:filename>')
def serve_image(filename):
    filepath = os.path.join("uploads", filename)
    if os.path.exists(filepath):
        return send_from_directory("uploads", filename)
    else:
        print(f"File not found: {filepath}")
        abort(404, description="File not found")

if __name__ == '__main__':
    app.run(debug=True)