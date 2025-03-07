import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage  # Import Image from PIL and alias it as PILImage
from io import BytesIO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import streamlit as st
import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Union
from datasets import Dataset
from torch.utils.data import DataLoader

# Set device (CPU or GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load a pre-trained ResNet model for advertisement detection
resnet_model = models.resnet18(pretrained=True)
resnet_model.eval()

# Preprocess the image for ResNet
resnet_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to scrape images from a URL
def scrape_images(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    images = soup.find_all('img')
    image_urls = [img['src'] for img in images if 'src' in img.attrs]
    return image_urls

# Function to download an image from a URL
def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        image = PILImage.open(BytesIO(response.content))  # Use PILImage instead of Image
        if image.mode != "RGB":  # Convert grayscale or RGBA to RGB
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# Function to check if an image is an advertisement
def is_advertisement(image):
    try:
        input_tensor = resnet_preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = resnet_model(input_batch)
        
        # Assuming class 0 is for advertisements (this is just an example)
        return torch.argmax(output, dim=1).item() == 0
    except Exception as e:
        print(f"Error checking advertisement: {e}")
        return False

# Function to generate image embeddings using CLIP
def get_image_embedding(image):
    try:
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs)
        return embeddings.numpy()
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# Function to create a FAISS index for semantic search
def create_faiss_index(embeddings):
    #embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # exacte search with L2 distance
    # index = faiss.IndexFlatIP(dimension)  # Use inner product (dot product)
    # index = faiss.IndexFlatIP(dimension)  # use cosine similarity(after normalizing the vectors)
    index.add(embeddings)
    return index



# Function to perform semantic search using FAISS
def search_similar_images(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices, distances

# Function to encode images
def encode_images(images: Union[List[str], List[PILImage.Image]], batch_size: int):
    def transform_fn(el):
        # Check if the image is a PIL Image instance
        if isinstance(el['image'], PILImage.Image):
            imgs = el['image']
        else:
            # Fallback: decode the image using datasets' Image decode (if applicable)
            imgs = [PILImage.open(_) for _ in el['image']]  # Use PILImage.open
        return clip_processor(images=imgs, return_tensors='pt')

    dataset = Dataset.from_dict({'image': images})
    dataset = dataset.cast_column('image', Image(decode=False)) if isinstance(images[0], str) else dataset
    dataset.set_format('torch')
    dataset.set_transform(transform_fn)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    image_embeddings = []
    pbar = tqdm(total=len(images) // batch_size, position=0)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}
            image_embeddings.extend(clip_model.get_image_features(**batch).detach().cpu().numpy())
            pbar.update(1)
    
    pbar.close()
    return np.stack(image_embeddings)

# Function to encode text
def encode_text(text: List[str], batch_size: int):
    dataset = Dataset.from_dict({'text': text})
    dataset = dataset.map(lambda el: clip_processor(text=el['text'], return_tensors="pt",
                                                max_length=77, padding="max_length", truncation=True),
                          batched=True,
                          remove_columns=['text'])
    dataset.set_format('torch')
    dataloader = DataLoader(dataset, batch_size=batch_size)

    text_embeddings = []
    pbar = tqdm(total=len(text) // batch_size, position=0)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}
            text_embeddings.extend(clip_model.get_text_features(**batch).detach().cpu().numpy())
            pbar.update(1)

    pbar.close()
    return np.stack(text_embeddings)

# Streamlit UI
st.title("Image Scraping, Filtering & Semantic Search")

# Input URL to scrape images
url = st.text_input("Enter the URL to scrape images from:")
if url:
    image_urls = scrape_images(url)
    st.write(f"Found {len(image_urls)} images.")
    
    embeddings = []
    valid_images = []
    for image_url in image_urls:
        try:
            image = download_image(image_url)
            if image is not None:
                print(f"Processing image: {image_url}")
                if not is_advertisement(image):
                    embedding = get_image_embedding(image)
                    if embedding is not None:
                        embeddings.append(embedding)
                        valid_images.append(image)
        except Exception as e:
            st.write(f"Error processing image: {e}")
    
    if valid_images:
        st.write(f"After filtering, {len(valid_images)} images remain.")
        embeddings = np.vstack(embeddings)
        index = create_faiss_index(embeddings)
        
        # Upload a query image for semantic search
        query_image = st.file_uploader("Upload an image for semantic search:", type=["jpg", "png"])
        if query_image:
            query_image = PILImage.open(query_image)  # Use PILImage.open
            if query_image.mode != "RGB":  # Convert grayscale or RGBA to RGB
                query_image = query_image.convert("RGB")
            query_embedding = get_image_embedding(query_image)
            if query_embedding is not None:
                indices,distances = search_similar_images(index, query_embedding.reshape(1, -1))
                indices_distances = list(zip(indices[0], distances[0]))
                indices_distances.sort(key=lambda x: x[1])
                
                st.write("Top similar images:")
                st.write(distances)
                for idx, distance in indices_distances:
                    st.write(f"Distance: {distance:.4f}")
                    st.image(valid_images[idx], use_container_width=True)

        # Text-based search
        search_text = st.text_input("Enter text to search for similar images:")
        if search_text:
            text_embedding = encode_text([search_text], batch_size=32)
            text_embedding = text_embedding / np.linalg.norm(text_embedding, ord=2, axis=-1, keepdims=True)
            indices, distances = search_similar_images(index, text_embedding.reshape(1, -1))
            indices_distances = list(zip(indices[0], distances[0]))
            indices_distances.sort(key=lambda x: x[1])
            
            st.write(f"Top images matching '{search_text}':")
            for idx, distance in indices_distances:
                st.write(f"Distance: {distance:.4f}")
                st.image(valid_images[idx], use_container_width=True)

# Allow duplicate OpenMP runtimes (use with caution)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"