#!/usr/bin/env python3
"""
Test vector search in Milvus using a query image and CLIP embeddings.

Usage:
    python test_search.py <path_to_query_image>

Environment variables (via .env):
    MILVUS_HOST       Milvus host (default: localhost)
    MILVUS_PORT       Milvus port (default: 19530)
    COLLECTION_NAME   Name of the Milvus collection (default: items)
    TOP_K             Number of nearest neighbors to retrieve (default: 5)
"""
import os
import sys

torch = __import__('torch')
from dotenv import load_dotenv
from pymilvus import connections, Collection
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load environment
load_dotenv()
MILVUS_HOST     = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT     = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "items")
TOP_K           = int(os.getenv("TOP_K", "5"))

# Initialize CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


def embed_image(image_path: str) -> list[float]:
    """Generate a normalized 512-dim embedding for an image."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].tolist()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_query_image>")
        sys.exit(1)

    query_path = sys.argv[1]
    if not os.path.exists(query_path):
        print(f"Error: File not found: {query_path}")
        sys.exit(1)

    # Embed the query image
    print(f"Embedding query image: {query_path}")
    query_vec = embed_image(query_path)

    # Connect to Milvus
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # Perform vector search
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vec],
        anns_field="image_vector",
        param=search_params,
        limit=TOP_K,
        output_fields=["image_path", "csv_data"]
    )

    # Display results
    print(f"Top {TOP_K} results:")
    for hits in results:
        for hit in hits:
            print(f"ID: {hit.id}, Distance: {hit.distance:.4f}")
            print(f"Image Path: {hit.entity.get('image_path')}")
            print("CSV Data:")
            print(hit.entity.get('csv_data'))
            print("-" * 40)


if __name__ == "__main__":
    main()
