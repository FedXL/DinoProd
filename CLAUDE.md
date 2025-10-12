# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an image embedding extraction service that provides REST API endpoints for generating embeddings from images using various computer vision models. The service supports multiple model architectures including DINOv2 and InternViT variants, with different extraction strategies for various use cases.

## Development Commands

### Running the Application

**Batch Processing Mode** (processes all collections and building images):
```bash
python3.11 start.py
```

**FastAPI Service Mode** (REST API server):
```bash
# Direct uvicorn command - starts service on port 8000
uvicorn api:app --host 0.0.0.0 --port 8000

# Alternative using Python 3.11
python3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker Build and Run
```bash
# Build Docker image
docker build -t dino-embedding-service .

# Run container
docker run -p 8000:8000 dino-embedding-service
```

### Installing Dependencies
```bash
# Install Python dependencies
pip install -r req.txt

# Or using Python 3.11 specifically
python3.11 -m pip install -r req.txt
```

## Architecture

### Core Components

**Embedding Extractors** (`embedding_handler.py`):
- `Dino2ExtractorV1`: Uses Facebook's DINOv2 model (dinov2_vitg14) for feature extraction
- `Dino3ExtractorV1`: Uses Facebook's DINOv3 model for global image embeddings (currently active in API)
- `InternVIT600mbExtractor`: Uses OpenGVLab's InternViT-300M model for lightweight extraction
- `InternVITThreeLevelExtractor`: Advanced extractor with global, focused, and tile-based features
- `InternVITSimpleExtractor`: Simple InternViT feature extractor using mean pooling

**Image Loading** (`embedding_handler.py`):
- `URLImageLoader`: Downloads and processes images from URLs

**Service Layer** (`embedding_handler.py`):
- `EmbeddingService`: Coordinates image loading and embedding extraction

**API Layer** (`api.py`):
- FastAPI application with REST endpoints:
  - `/embedding/fast_extract`: Uses DINOv3 for extraction (main endpoint)
  - `/embedding/test_extract`: Uses InternViT with concurrency control (commented out)
  - `/`: Health check endpoint
- Automatic IP registration with ArtCracker backend on startup

**Controller Layer** (`controller.py`):
- Batch processing workflows via `main_flow2()` function
- `extract_embedding_from_building_images_flow()`: Processes building images
- `extract_embedding_from_task_collection()`: Processes task collections
- Handles interaction with external ArtCracker API for bulk operations

**Request Handler** (`request_handler.py`):
- HTTP client for ArtCracker API endpoints
- Handles authentication and data transfer

### Key Features

1. **Multiple Model Support**: DINOv2, DINOv3, and InternViT models with different configurations
2. **Dual Operation Modes**: 
   - REST API service for real-time embedding extraction
   - Batch processing mode for bulk collection processing
3. **Concurrency Control**: Semaphore-based request limiting for model inference
4. **Three-Level Feature Extraction**: Global, focused, and tile-based features for comprehensive analysis
5. **External API Integration**: Communicates with ArtCracker backend for data synchronization
6. **Automatic Service Registration**: IP address auto-registration with backend on startup

### Configuration

The application requires a `.env` file with:
- `TOKEN`: Authentication token for ArtCracker API

### Device Requirements

- CUDA-compatible GPU (models default to 'cuda' device)
- Python 3.11
- NVIDIA CUDA 11.8+ (as specified in Dockerfile)

### Model Loading

Models are loaded on service startup and cached in memory. The service automatically handles:
- Model downloading from HuggingFace Hub
- GPU memory management
- Image preprocessing and normalization

### API Endpoints

**Embedding Extraction:**
- `POST /embedding/fast_extract`: Fast embedding extraction using DINOv3 (active)
- `POST /embedding/test_extract`: InternViT extraction with queue management (currently disabled)

**Image Classification:**
- `POST /classifier/classify`: Zero-shot image classification using SigLIP-2

**Health Check:**
- `GET /`: Health check endpoint

**Embedding Request format:**
```json
{
  "url": "https://example.com/image.jpg"
}
```

**Embedding Response format:**
```json
{
  "embedding": [0.1, 0.2, ...],
  "url": "https://example.com/image.jpg"
}
```

**Classification Request format:**
```json
{
  "url": "https://example.com/image.jpg"
}
```

**Classification Response format:**
```json
{
  "success": true,
  "category": "building",
  "confidence": 0.87,
  "error": null
}
```

### Batch Processing Workflows

The `start.py` entry point runs batch processing workflows via `controller.main_flow2()`:

1. **Collection Processing**: Processes all available task collections from ArtCracker API
   - Retrieves collection names via `get_collections_names_list()`
   - Extracts embeddings from all images in each collection
   - Sends results back to ArtCracker backend

2. **Building Images Processing**: Processes building image datasets
   - Retrieves building images via `get_building_images()`
   - Generates embeddings using DINOv2 model
   - Uploads embeddings to ArtCracker database

The batch processor automatically handles:
- Error recovery and logging
- Progress tracking across large datasets
- API rate limiting and authentication

### Zero-Shot Image Classification

The service includes a zero-shot image classifier using Google's SigLIP-2 model:

**Features:**
- Classifies images into predefined categories: "building", "painting", or "other"
- Uses text prompt variations for improved accuracy
- In-memory category embedding caching for fast inference
- Configurable confidence threshold for "other" classification

**Configuration:**
- Categories defined in `config/categories.json`
- Model and threshold configurable via environment variables
- Text embeddings computed once at startup and cached in memory

**Architecture:**
- `classifier/model.py`: SigLIP-2 model wrapper for text and image encoding
- `classifier/image_loader.py`: Async image downloading with error handling
- `classifier/classifier_service.py`: Main classification logic with embedding cache
- `classifier/config.py`: Configuration management and validation

**Startup Process:**
1. Load SigLIP-2 model to GPU (~25-30 seconds)
2. Load category definitions from JSON
3. Compute text embeddings for all category prompts (~2-3 seconds)
4. Cache embeddings in memory for fast classification