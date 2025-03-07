# Image Scraping, Filtering & Semantic Search

## Overview
This project is a Streamlit-based application that scrapes images from a given URL, filters out advertisements, generates image embeddings using CLIP, and performs semantic search using FAISS. It supports both image-based and text-based search for finding visually similar images.

## Features
- **Scrape images**: Extracts images from a given URL.
- **Filter advertisements**: Uses a pre-trained ResNet model to detect and remove advertisements.
- **Generate image embeddings**: Uses OpenAI's CLIP model to create vector embeddings for images.
- **Build FAISS index**: Stores image embeddings for efficient similarity search.
- **Semantic search**: Allows users to search for similar images using an uploaded image or text query.
- **Optimized FAISS indexing**: Supports multiple FAISS indexing methods (Flat, HNSW, IVF).

## Technologies Used
- **Python**
- **Streamlit**
- **BeautifulSoup** (for web scraping)
- **PIL (Pillow)** (for image processing)
- **Torch & torchvision** (for ResNet & CLIP)
- **Transformers** (Hugging Face CLIP model)
- **FAISS** (for fast similarity search)
- **NumPy & Pandas** (for data handling)
- **Matplotlib** (for visualization)

## Installation
Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Use
### Running the Application
Run the Streamlit app using:
```bash
streamlit run st.py
```

### Steps to Use the Application
1. **Enter a URL**: Input a website URL containing images to scrape.
2. **Scrape & Filter**: The app extracts and filters images (non-advertisements).
3. **View Filtered Images**: Displays the valid images after filtering.
4. **Perform Image Search**:
   - **Upload an image**: Finds similar images from the dataset.
   - **Enter text query**: Searches for relevant images using text descriptions.
5. **View Search Results**: Displays similar images along with similarity scores.

## FAISS Indexing Methods
- **Flat Index**: Exact nearest neighbor search with L2 distance.
- **HNSW (Hierarchical Navigable Small World)**: Fast approximate nearest neighbor search.
- **IVF (Inverted File Index)**: Efficient clustering-based nearest neighbor search.

## Customization
- Modify `create_faiss_index()` in `st.py` to use different FAISS indexing methods.
- Adjust `is_advertisement()` logic to fine-tune ad filtering.
- Change `CLIPProcessor` settings for different embedding behavior.







