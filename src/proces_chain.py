# File: pipeline.py
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
dotenv_path = os.getenv('.env', '.env')
load_dotenv(dotenv_path)
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'items')
TOP_K = int(os.getenv('TOP_K', '3'))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4.1-2025-04-14')
API_KEY = os.getenv('OPENAI_API_KEY')
SAMPLE_FOLDER = os.getenv(
    'SAMPLE_IMAGES_FOLDER',
    os.path.expanduser('data/test/pdf2png')
)

if not API_KEY:
    logging.error('OPENAI_API_KEY is not set.')
    raise EnvironmentError('Missing OPENAI_API_KEY')

# Initialize clients
openai_client = OpenAI(api_key=API_KEY)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_model.eval()


def list_sample_images(folder: str = SAMPLE_FOLDER) -> List[str]:
    patterns = ['*.png', '*.jpg', '*.jpeg']
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(os.path.basename(f) for f in files)


def embed_image(path: str) -> List[float]:
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt')
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    norm = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return norm[0].tolist()


def image_to_data_uri(path: str) -> str:
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def process_image(path: Optional[str], sample: Optional[str]) -> Tuple[str, str]:
    if sample:
        img_path = os.path.join(SAMPLE_FOLDER, sample)
    elif path:
        img_path = path
    else:
        return 'No image provided.', ''
    try:
        vec = embed_image(img_path)
        connections.connect(alias='default', host=MILVUS_HOST, port=MILVUS_PORT)
        col = Collection(COLLECTION_NAME)
        col.load()
        params = {'metric_type': 'L2', 'params': {'nprobe': 10}}
        res = col.search(
            data=[vec], anns_field='image_vector', param=params,
            limit=TOP_K, output_fields=['csv_data']
        )
        tables = [hit.entity.get('csv_data', '') for hit in res[0]]
        csv_str = '\n\n'.join(tables) or 'No CSV data.'
        # Invoke GPT-Vision
        prompt = (
            '### Context\n'
            'Reference CSV tables for similar parts:\n'
            f'{csv_str}\n\n'
            '### Task\n'
            'Propose machining steps with dimensions in a Markdown table.'
        )
        data_uri = image_to_data_uri(img_path)
        messages = [
            {'role': 'system', 'content': 'Expert in metal machining.'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': data_uri}}
            ]}
        ]
        resp = openai_client.chat.completions.create(
            model=MODEL_NAME, temperature=0, top_p=0,
            messages=messages
        )
        return csv_str, resp.choices[0].message.content
    except (MilvusException, OpenAIError) as e:
        logging.error(e)
        return '', f'Error: {e}'


def followup_plan(initial_plan: str, user_msg: str) -> str:
    """
    Generate a follow-up updated plan based on the initial GPT output.
    """
    prompt = (
        'Based on the initial plan and CSV context below:\n'
        f'{initial_plan}\n'
        'User adjustment:\n'
        f'{user_msg}\n'
        'Provide an updated Markdown table of operations.'
    )
    resp = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': 'Expert in metal machining.'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content