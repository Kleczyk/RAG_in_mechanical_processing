#!/usr/bin/env python3
"""
Core processing pipeline: embedding, Milvus search, and GPT-4 Vision invocation.
"""
import os
import glob
import base64
import logging
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, Collection, MilvusException
from openai import OpenAI, OpenAIError

# Load environment variables
load_dotenv()
MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "items")
TOP_K: int = int(os.getenv("TOP_K", "3"))
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-2025-04-14")
API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
SAMPLE_IMAGES_FOLDER: str = os.getenv(
    "SAMPLE_IMAGES_FOLDER", os.path.expanduser("/home/daniel/repos/RAG_in_mechanical_processing/src/data/test/pdf2png")
)

if not API_KEY:
    logging.error("OPENAI_API_KEY is not set.")
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

# Initialize clients
openai_client = OpenAI(api_key=API_KEY)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


def embed_image(image_path: str) -> List[float]:
    """
    Compute a normalized CLIP embedding for the given image.

    Args:
        image_path: Path to the image file.

    Returns:
        A list of floats representing the normalized feature vector.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return normalized[0].tolist()


def image_to_data_uri(path: str) -> str:
    """
    Convert an image file to a Base64-encoded data URI.

    Args:
        path: Local path to the image file.

    Returns:
        A data URI string for embedding in prompts.
    """
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def list_sample_images(folder: str = SAMPLE_IMAGES_FOLDER) -> List[str]:
    """
    List supported sample images in the given folder.

    Args:
        folder: Directory containing sample images.

    Returns:
        Sorted list of image filenames.
    """
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(os.path.basename(p) for p in files)


def process_image(
    uploaded_path: Optional[str],
    selected_filename: Optional[str]
) -> Tuple[str, str]:
    """
    Execute the processing pipeline on an input image.

    Steps:
      1. Determine image source (upload vs sample).
      2. Embed image via CLIP.
      3. Query Milvus for top-K similar vectors.
      4. Extract CSV tables from hits.
      5. Build GPT-4 Vision prompt and call API.

    Args:
        uploaded_path: Filepath of the uploaded image.
        selected_filename: Sample image filename if selected.

    Returns:
        Tuple of (retrieved CSV tables, GPT-4 Vision response).
    """
    # Select image path
    if selected_filename:
        image_path = os.path.join(SAMPLE_IMAGES_FOLDER, selected_filename)
    elif uploaded_path:
        image_path = uploaded_path
    else:
        return "No input image provided.", ""

    try:
        # Embed image
        query_vector = embed_image(image_path)

        # Connect to Milvus and search
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector],
            anns_field="image_vector",
            param=params,
            limit=TOP_K,
            output_fields=["csv_data", "image_path"]
        )
        hits = results[0]

        # Extract CSV data
        tables = [hit.entity.get("csv_data", "") for hit in hits]
        csv_output = "\n\n".join(tables) if tables else "No CSV data found."

        # Build GPT prompt
        prompt = (
            "### Context\n"
            "Below are reference CSV tables describing machining parameters for similar parts:\n"
            f"{csv_output}\n\n"
            "### Task\n"
            "Propose a sequence of machining steps for the uploaded part,"
            " including actual dimensions from the drawing,"
            " formatted as a table: | Operation | Status Number | Description |."
        )
        data_uri = image_to_data_uri(image_path)
        user_segments = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}}
        ]
        messages = [
            {"role": "system", "content": "You are an expert in metal part machining."},
            {"role": "user", "content": user_segments}
        ]

        # Call GPT-4 Vision
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        gpt_output = response.choices[0].message.content

        return csv_output, gpt_output

    except MilvusException as me:
        logging.error("Milvus error: %s", me)
        return "Error querying Milvus.", str(me)
    except OpenAIError as oe:
        logging.error("OpenAI API error: %s", oe)
        return csv_output if 'csv_output' in locals() else "", f"API error: {oe}"
    except Exception as e:
        logging.exception("Unexpected error during processing.")
        return "", f"Unexpected error: {e}"
