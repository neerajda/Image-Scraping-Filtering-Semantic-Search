import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import streamlit as st

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
        image = Image.open(BytesIO(response.content))
        if image.mode != "RGB":  # Convert grayscale or RGBA to RGB
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

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

# Load the CLIP model and processor for generating embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index



# Function to perform semantic search using FAISS
def search_similar_images(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices

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
            query_image = Image.open(query_image)
            if query_image.mode != "RGB":  # Convert grayscale or RGBA to RGB
                query_image = query_image.convert("RGB")
            query_embedding = get_image_embedding(query_image)
            if query_embedding is not None:
                indices = search_similar_images(index, query_embedding.reshape(1, -1))
                
                st.write("Top similar images:")
                for idx in indices[0]:
                    st.image(valid_images[idx], use_column_width=True)

