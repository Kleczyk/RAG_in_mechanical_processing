#!/usr/bin/env python3
"""process_chain_gradio.py

Gradio interface for PyMilvus + CLIP + multimodal GPT-4 Vision pipeline:
1. Prześlij obraz lub wybierz z folderu.
2. Embed z HuggingFace CLIP.
3. Query Milvus: pobierz top-K podobnych zapisów (CSV).
4. Zbuduj prompt po polsku i wywołaj GPT-4 Vision (OpenAI Python SDK v1.x).
5. Wyświetl znalezione tabele i wygenerowaną odpowiedź.

Usage:
    python process_chain_gradio.py
"""
import os
import glob
import base64
from dotenv import load_dotenv
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, Collection
from openai import OpenAI
import gradio as gr

# ── LOAD CONFIG ───────────────────────────────────────────────────────────────
load_dotenv()
MILVUS_HOST     = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT     = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "items")
TOP_K           = int(os.getenv("K", "3"))
MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-4.1-2025-04-14")
API_KEY         = os.getenv("OPENAI_API_KEY")
SAMPLE_IMAGES_FOLDER = os.getenv("SAMPLE_IMAGES_FOLDER", "/home/daniel/repos/RAG_in_mechanical_processing/src/data/test/pdf2png")

# ── INIT CLIENTS ───────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=API_KEY)

# ── INIT CLIP ──────────────────────────────────────────────────────────────────
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
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


def list_sample_images(folder: str) -> list[str]:
    paths = glob.glob(os.path.join(folder, "*.png")) + \
            glob.glob(os.path.join(folder, "*.jpg")) + \
            glob.glob(os.path.join(folder, "*.jpeg"))
    return [os.path.basename(p) for p in sorted(paths)]


# ── PROCESS FUNCTION ───────────────────────────────────────────────────────────
def process_image(uploaded_img_path, selected_filename):
    # Wybór obrazu: dropdown ma pierwszeństwo
    if selected_filename:
        img_path = os.path.join(SAMPLE_IMAGES_FOLDER, selected_filename)
    elif uploaded_img_path:
        img_path = uploaded_img_path
    else:
        return "Brak obrazu wejściowego.", ""

    # 1) Embed obraz
    q_vec = embed_image(img_path)

    # 2) Połączenie i wyszukiwanie w Milvus
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    coll = Collection(COLLECTION_NAME)
    coll.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = coll.search(
        data=[q_vec], anns_field="image_vector",
        param=search_params, limit=TOP_K,
        output_fields=["csv_data", "image_path"]
    )
    hits = results[0]

    # 3) Przygotowanie tabel CSV
    tables = [hit.entity.get("csv_data", "") for hit in hits]
    csv_text = "\n\n".join(tables) if tables else "Brak danych CSV."

    # 4) Budowa promptu
    prompt_text = (
        "### Kontekst\n"
        "Poniżej znajdują się trzy tabele CSV opisujące parametry obróbki podobnych elementów oraz rysunek przesłany rysunek. \n\n"
        + csv_text +
        "\n\n### Zadanie\n"
        "Na podstawie referencyjnych tabel zaproponuj etapy procesu obróbki dla elementu widocznego na przesłanym rysunku "
        "zwracając uwagę na wymiary z rysunku tak aby odpowiednio wypełnić prawdziwimi wartościami z rysunku w etapach obróbki w formie tabeli | OP | NR. STANU | OPIS |.\n"
    )

    data_uri = image_to_data_uri(img_path)
    user_segments = [
        {"type": "text",      "text": prompt_text},
        {"type": "image_url", "image_url": {"url": data_uri}}
    ]
    messages = [
        {"role": "system", "content": "Jesteś ekspertem w dziedzinie obróbki elementów metalowych i generujesz szczegółowe tabele procesów."},
        {"role": "user",   "content": user_segments}
    ]

    # 5) Zapytanie do GPT-4 Vision
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    gpt_out = response.choices[0].message.content

    return csv_text, gpt_out


# ── GRADIO INTERFACE ────────────────────────────────────────────────────────────
image_choices = list_sample_images(SAMPLE_IMAGES_FOLDER)
with gr.Blocks() as demo:
    gr.Markdown("## RAG in Mechanical Processing")
    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.Image(type="filepath", label="Send image")
            dropdown = gr.Dropdown(choices=[""] + image_choices,
                                   value="", label="Select image from folder",
                                   interactive=True)
            btn = gr.Button("Process image")
        with gr.Column(scale=2):
            csv_output = gr.Textbox(label="3 most similar table to element", lines=10)
            gpt_output = gr.Markdown(label="Generated response")

    btn.click(process_image,
              inputs=[upload, dropdown],
              outputs=[csv_output, gpt_output])

if __name__ == "__main__":
    demo.launch(share=True)
