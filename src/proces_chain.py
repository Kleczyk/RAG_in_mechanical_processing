#!/usr/bin/env python3
"""process_chain.py

PyMilvus + CLIP + GPT-4o pipeline without LangChain:
1. CLI: user provides a new image path.
2. Embed with HuggingFace CLIP.
3. Query Milvus via PyMilvus: retrieve top-K similar records from "items" collection.
4. Build a Polish prompt including CSV tables.
5. Call GPT-4o multimodal ChatCompletion via openai python client with image and prompt.

Prereqs:
• .env:
    MILVUS_HOST=localhost
    MILVUS_PORT=19530
    OPENAI_API_KEY=sk-...
    COLLECTION_NAME=items
    K=3
• pip install python-dotenv pymilvus transformers torch pillow openai

Usage:
    python process_chain.py /path/to/query_image.png
"""
import os
import sys
import base64
from dotenv import load_dotenv
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, Collection
import openai

# ── LOAD CONFIG ───────────────────────────────────────────────────────────────
load_dotenv()
MILVUS_HOST     = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT     = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "items")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
openai.api_key  = OPENAI_API_KEY
TOP_K           = int(os.getenv("K", "3"))
MODEL_NAME      = "gpt-4o"

# ── INIT CLIP ──────────────────────────────────────────────────────────────────
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def embed_image(img_path: str) -> list[float]:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats[0].tolist()

def image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python process_chain.py <path_to_query_image>")
    query_img = sys.argv[1]
    if not os.path.exists(query_img):
        sys.exit(f"File not found: {query_img}")

    print("Embedding query image...")
    q_vec = embed_image(query_img)

    print("Connecting to Milvus...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    coll = Collection(COLLECTION_NAME)
    coll.load()

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = coll.search(
        data=[q_vec],
        anns_field="image_vector",
        param=search_params,
        limit=TOP_K,
        output_fields=["csv_data", "image_path"]
    )
    hits = results[0]

    # Build Polish prompt from retrieved CSVs
    tables = []
    for hit in hits:
        tables.append(hit.entity.get("csv_data", ""))
    prompt_text = (
        "### Kontekst\n"
        "Poniżej znajdują się trzy tabele CSV opisujące parametry obróbki podobnych elementów.\n\n"
        + "\n\n".join(tables)
        + "\n\n### Zadanie\n"
        "Na podstawie referencyjnych tabel zaproponuj parametry procesu obróbki dla elementu widocznego na przesłanym zdjęciu.\n"
        "Zwróć wynik w postaci tabeli CSV z kolumnami: operacja, narzędzie, prędkość_[m/min], posuw_[mm/obr], głębokość_skrawania_[mm]."
    )

    data_uri = image_to_data_uri(query_img)
    messages = [
        {"role": "system", "content": "Jesteś ekspertem w dziedzinie obróbki skrawaniem i generujesz szczegółowe tabele procesów."},
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": data_uri}
    ]

    print("Querying GPT-4o...")
    response = openai.ChatCompletion.create(model=MODEL_NAME, messages=messages)

    print("\n=== Wygenerowana tabela procesu ===\n")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
