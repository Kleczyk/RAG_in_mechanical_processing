import os
import base64
import glob

from dotenv import load_dotenv
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# ─── LOAD CONFIG ────────────────────────────────────────────────────────────────
load_dotenv()  # loads .env from CWD
MILVUS_HOST    = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT    = os.getenv("MILVUS_PORT", "19530")
# OPENAI_API_KEY ONLY needed if you later embed text; not used for images here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_PATH      = os.getenv("BASE_PATH")

# ─── INIT MILVUS ─────────────────────────────────────────────────────────────────
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# ─── 1. DEFINE/RESET SCHEMA ───────────────────────────────────────────────────────
collection_name = "items"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id",           dtype=DataType.INT64,        is_primary=True, auto_id=True),
    FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512),    # CLIP outputs 512-dim
    FieldSchema(name="csv_data",     dtype=DataType.VARCHAR,      max_length=8192),
    FieldSchema(name="image_path",   dtype=DataType.VARCHAR,      max_length=512),
]
schema     = CollectionSchema(fields, description="Item images + CSV with primary key")
collection = Collection(name=collection_name, schema=schema)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params":       {"nlist": 128}
}
collection.create_index(field_name="image_vector", index_params=index_params)
collection.load()

# ─── 2. INIT CLIP EMBEDDER ────────────────────────────────────────────────────────
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path: str):
    """Use HuggingFace CLIP to generate a 512-dim embedding for an image."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    # normalize to unit length (optional but common)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].tolist()

def read_csv(csv_path: str) -> str:
    """Return the raw CSV file as a single string."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return f.read()

# ─── 3. DISCOVER & INGEST ─────────────────────────────────────────────────────────
vecs, csvs, paths = [], [], []

for sub in os.listdir(BASE_PATH):
    subdir = os.path.join(BASE_PATH, sub)
    if not os.path.isdir(subdir):
        continue

    # pick the first image (png/jpg)
    img_files = glob.glob(os.path.join(subdir, "*.png")) + \
                glob.glob(os.path.join(subdir, "*.jpg"))
    if not img_files:
        print(f"Skip {sub}: no image found")
        continue
    img_path = img_files[0]

    # pick the first CSV
    csv_files = glob.glob(os.path.join(subdir, "*.csv"))
    if not csv_files:
        print(f"Skip {sub}: no CSV found")
        continue
    csv_path = csv_files[0]

    print(f"Processing {sub} …")
    vecs.append(embed_image(img_path))
    csvs.append(read_csv(csv_path))
    paths.append(img_path)

entities      = [vecs, csvs, paths]
insert_result = collection.insert(entities)
collection.flush()

print(f"Inserted {len(vecs)} items; generated IDs: {insert_result.primary_keys}")
