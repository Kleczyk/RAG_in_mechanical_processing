#!/usr/bin/env python3
"""
ingest_items.py

Skrypt do zarządzania kolekcją Milvus:
1. Reset i inicjalizacja schematu.
2. Ingest obrazów i CSV.
3. Funkcja do czyszczenia (usunięcia) wszystkich rekordów w kolekcji.

Usage:
    python ingest_items.py
    # i w Python REPL można wywołać clear_collection()
"""
import os
import glob
import torch
import base64
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

# ─── LOAD CONFIG ───────────────────────────────────────────────────────────────
load_dotenv()  # ładuje .env z CWD
MILVUS_HOST    = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT    = os.getenv("MILVUS_PORT", "19530")
BASE_PATH      = os.getenv("BASE_PATH", "/home/daniel/repos/RAG_in_mechanical_processing/src/output")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "items")

# ─── INIT MILVUS ───────────────────────────────────────────────────────────────
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# ─── 1. DEFINE/RESET SCHEMA ─────────────────────────────────────────────────────
def reset_schema():
    # Usuwa istniejącą kolekcję i tworzy od nowa
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'")

    fields = [
        FieldSchema(name="id",           dtype=DataType.INT64,        is_primary=True, auto_id=True),
        FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="csv_data",     dtype=DataType.VARCHAR,      max_length=8192),
        FieldSchema(name="image_path",   dtype=DataType.VARCHAR,      max_length=512),
    ]
    schema     = CollectionSchema(fields, description="Item images + CSV data")
    Collection(name=COLLECTION_NAME, schema=schema)
    print(f"Created collection '{COLLECTION_NAME}' with schema.")

    # Tworzenie indeksu
    coll = Collection(COLLECTION_NAME)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params":       {"nlist": 128}
    }
    coll.create_index(field_name="image_vector", index_params=index_params)
    coll.load()
    print("Index created and collection loaded.")

# ─── 2. EMBEDDER CLIP ────────────────────────────────────────────────────────────
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def embed_image(image_path: str) -> list[float]:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].tolist()

# ─── 3. READ CSV ─────────────────────────────────────────────────────────────────
def read_csv(csv_path: str) -> str:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return f.read()

# ─── 4. INGEST DATA ─────────────────────────────────────────────────────────────
def ingest_data():
    vecs, csvs, paths = [], [], []
    for sub in os.listdir(BASE_PATH):
        subdir = os.path.join(BASE_PATH, sub)
        if not os.path.isdir(subdir):
            continue

        img_files = glob.glob(os.path.join(subdir, "*.png")) + \
                    glob.glob(os.path.join(subdir, "*.jpg"))
        if not img_files:
            print(f"Skip {sub}: no image found")
            continue
        img_path = img_files[0]

        csv_files = glob.glob(os.path.join(subdir, "*.csv"))
        if not csv_files:
            print(f"Skip {sub}: no CSV found")
            continue
        csv_path = csv_files[0]

        print(f"Processing {sub} …")
        vecs.append(embed_image(img_path))
        csvs.append(read_csv(csv_path))
        paths.append(img_path)

    # Wstawianie do kolekcji
    coll = Collection(COLLECTION_NAME)
    insert_result = coll.insert([vecs, csvs, paths])
    coll.flush()
    print(f"Inserted {len(vecs)} items; IDs: {insert_result.primary_keys}")

# ─── 5. CLEAR COLLECTION ─────────────────────────────────────────────────────────
def clear_collection():
    """Usuwa wszystkie rekordy z kolekcji zachowując schemat i indeks."""
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist.")
        return
    coll = Collection(COLLECTION_NAME)
    coll.load()
    # Wyrażenie usuwające wszystkie rekordy z id >= 0
    coll.delete(expr="id >= 0")
    coll.flush()
    print(f"Cleared all records from collection '{COLLECTION_NAME}'.")

# ─── ENTRYPOINT ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reset_schema()
    ingest_data()
    # clear_collection()
