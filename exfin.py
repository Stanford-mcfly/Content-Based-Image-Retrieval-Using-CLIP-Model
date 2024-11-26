import os
import logging
from pymongo import MongoClient
from PIL import Image, UnidentifiedImageError
import torch
import numpy as np
import clip
from tqdm import tqdm

BASE_DIRECTORY = "animal2"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
LOG_FILE = "extract.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    try:
        client = MongoClient(mongo_uri)
        logger.info(f"Connected to MongoDB at {mongo_uri}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

client = get_mongo_client()
db = client.get_database(os.getenv("MONGO_DB", "cbir"))
collection = db.get_collection(os.getenv("MONGO_COLLECTION", "image_features"))

def load_clip_model():
    try:
        logger.info("Loading CLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        logger.info("CLIP model loaded successfully.")
        return model, preprocess, device
    except Exception as e:
        logger.error(f"Error loading CLIP model: {e}")
        raise

model, preprocess, device = load_clip_model()

def extract_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, FileNotFoundError) as e:
        logger.error(f"Cannot open image '{image_path}': {e}")
        return None

    try:
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy()[0]
        return embedding
    except Exception as e:
        logger.error(f"Error extracting embedding from '{image_path}': {e}")
        return None

def store_image_embedding(image_name, relative_image_path, embedding):
    try:
        if collection.find_one({"image_path": relative_image_path}):
            logger.info(f"Image already exists in DB: {relative_image_path}")
            return

        document = {
            "image_name": image_name,
            "image_path": relative_image_path,
            "image_embedding": embedding.tolist()
        }
        collection.insert_one(document)
        logger.info(f"Stored embedding for: {relative_image_path}")
    except Exception as e:
        logger.error(f"Error inserting document into MongoDB for '{relative_image_path}': {e}")

def process_images_recursively(base_directory):
    if not os.path.isdir(base_directory):
        logger.error(f"Directory '{base_directory}' does not exist. Please check the path.")
        return

    image_paths = []

    for root, dirs, files in os.walk(base_directory):
        for filename in files:
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                absolute_image_path = os.path.join(root, filename)
                relative_image_path = os.path.relpath(absolute_image_path, base_directory)
                image_paths.append((filename, relative_image_path, absolute_image_path))

    if not image_paths:
        logger.warning(f"No images found in '{base_directory}' and its subdirectories.")
        return

    logger.info(f"Found {len(image_paths)} images in '{base_directory}' and its subdirectories.")

    for filename, relative_image_path, absolute_image_path in tqdm(image_paths, desc="Processing Images"):
        logger.info(f"Processing: {absolute_image_path}")
        embedding = extract_image_embedding(absolute_image_path)
        if embedding is not None:
            store_image_embedding(filename, relative_image_path, embedding)

    logger.info("All images have been processed and stored in MongoDB.")

if __name__ == "__main__":
    process_images_recursively(BASE_DIRECTORY)