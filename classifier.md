# Zero-Shot Image Classifier Documentation

## Overview

This document provides a comprehensive guide to the zero-shot image classification system built using Google's SigLIP-2 model. The classifier can categorize images into predefined categories without requiring training data for those specific categories.

## How Zero-Shot Classification Works

### Core Concept

Zero-shot classification leverages a pre-trained vision-language model (SigLIP-2) that understands both images and text in a shared embedding space. Instead of training on specific categories, the model:

1. **Encodes text descriptions** of categories into embeddings
2. **Encodes input images** into embeddings  
3. **Compares similarities** between image and text embeddings
4. **Predicts the most similar** category based on cosine similarity

### SigLIP-2 Model

SigLIP (Sigmoid Loss for Language-Image Pre-training) is Google's vision-language model that:
- Uses contrastive learning to align image and text representations
- Supports zero-shot classification without category-specific training
- Provides normalized embeddings for efficient similarity computation
- Handles various image types and text prompts effectively

### Text Prompt Engineering

The classifier uses multiple text prompts per category to improve accuracy:

```json
{
  "building": [
    "a photo of a building",
    "building facade", 
    "architectural photo",
    "exterior of a building",
    "building photograph"
  ]
}
```

**Why multiple prompts?**
- Different phrasings capture various aspects of the category
- Using max similarity (not averaging) preserves strong matches
- Reduces sensitivity to specific wording choices

**Max vs Average Similarity:**
For an image of a modern glass building:
- Prompt "building facade" → similarity: 0.9 (excellent match!)
- Prompt "old brick building" → similarity: 0.3 (poor match)

**Averaging**: (0.9 + 0.3) / 2 = **0.6** ❌ (loses the strong signal)  
**Max**: max(0.9, 0.3) = **0.9** ✅ (preserves the strong match)

## Architecture Deep Dive

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Image URL     │───▶│  AsyncImageLoader │───▶│  PIL Image      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Category Text   │───▶│   SigLIP Model   │◀───│ Image Embedding │
│ Embeddings      │    │                  │    │                 │
│ (Cached)        │    └──────────────────┘    └─────────────────┘
└─────────────────┘             │
         ▲                      ▼
         │              ┌─────────────────┐
         │              │ Cosine Similarity│
         │              │   Computation    │
         │              └─────────────────┘
         │                       │
         │                       ▼
┌─────────────────┐    ┌──────────────────┐
│ Startup Process │    │ Classification   │
│ Text Encoding   │    │    Result        │
└─────────────────┘    └──────────────────┘
```

### Memory Architecture

**Startup (One-time):**
```python
# 1. Load categories from JSON
categories = {
    "building": ["a photo of a building", "building facade", ...],
    "painting": ["a painting", "artwork", ...]
}

# 2. Encode all text prompts
text_embeddings = model.encode_text(all_prompts)

# 3. Keep ALL individual embeddings per category (don't average!)
category_embeddings = {
    "building": [building_prompt_1_emb, building_prompt_2_emb, ...],
    "painting": [painting_prompt_1_emb, painting_prompt_2_emb, ...]
}

# 4. Cache in memory (persist for service lifetime)
self.category_embeddings = category_embeddings
```

**Runtime (Per Request):**
```python
# 1. Download image
image = await load_image(url)

# 2. Encode image
image_embedding = model.encode_image(image)

# 3. Compare with cached categories (use MAX similarity)
similarities = {}
for category, text_embeddings in self.category_embeddings.items():
    # Compare to ALL prompts for this category
    category_similarities = [
        cosine_similarity(image_embedding, text_emb) 
        for text_emb in text_embeddings
    ]
    # Take the BEST (maximum) similarity for this category
    similarities[category] = max(category_similarities)

# 4. Apply threshold and return result
best_category = max(similarities, key=similarities.get)
confidence = similarities[best_category]

if confidence >= threshold:
    return best_category
else:
    return "other"
```

## Developer Guide

### Setting Up the Classifier

#### 1. Configuration Setup

Create or update your `.env` file:
```bash
# Required
TOKEN=your_artcracker_token

# Classifier settings (optional, defaults shown)
CLASSIFIER_MODEL=google/siglip-base-patch16-512
CLASSIFIER_THRESHOLD=0.35
CATEGORIES_FILE=config/categories.json
CUDA_AVAILABLE=true
```

#### 2. Category Configuration

Edit `config/categories.json` to define your categories:
```json
{
  "building": [
    "a photo of a building",
    "building facade",
    "architectural photo",
    "exterior of a building",
    "building photograph"
  ],
  "painting": [
    "a painting",
    "artwork", 
    "photo of a painting",
    "painting of a person",
    "artistic work"
  ]
}
```

**Best Practices for Categories:**
- Use 3-7 prompt variations per category
- Include both formal and informal descriptions
- Consider edge cases (e.g., "photo of a painting" vs "painting")
- Test different phrasings to improve accuracy

#### 3. Starting the Service

```bash
# Start the FastAPI service
uvicorn api:app --host 0.0.0.0 --port 8000

# Or using Python 3.11
python3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**Startup logs to expect:**
```
[lifespan] Starting service initialization
[lifespan] Initializing classifier service...
Loading SigLIP model: google/siglip-base-patch16-512
SigLIP model loaded in 28.45 seconds on cuda
Computing embeddings for 12 text prompts across 2 categories...
Category 'building': averaged 5 text prompts
Category 'painting': averaged 7 text prompts
Classifier initialized in 31.23 seconds
Loaded 2 categories with 12 total text variations
[lifespan] Classifier service initialized successfully
```

### Using the API

#### Basic Classification Request

```bash
curl -X POST "http://localhost:8000/classifier/classify" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/building.jpg"}'
```

#### Response Format

**Success:**
```json
{
  "success": true,
  "category": "building",
  "confidence": 0.87,
  "error": null
}
```

**Error:**
```json
{
  "success": false,
  "category": null,
  "confidence": null,
  "error": "Failed to download image: Connection timeout"
}
```

### Programmatic Usage

#### Direct Service Usage

```python
from classifier.config import ClassifierConfig
from classifier.classifier_service import ClassifierService

# Initialize
config = ClassifierConfig()
service = ClassifierService(config)
await service.initialize()

# Classify an image
result = await service.classify_image("https://example.com/image.jpg")
print(f"Category: {result.category}, Confidence: {result.confidence}")
```

#### Batch Classification

```python
import asyncio

async def classify_batch(urls):
    results = []
    for url in urls:
        result = await service.classify_image(url)
        results.append((url, result.category, result.confidence))
    return results

# Usage
urls = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
results = await classify_batch(urls)
```

### Extending the Classifier

#### Adding New Categories

1. **Update categories.json:**
```json
{
  "building": [...],
  "painting": [...],
  "vehicle": [
    "a photo of a car",
    "automobile",
    "vehicle",
    "photo of a truck",
    "motorcycle"
  ]
}
```

2. **Restart the service** (embeddings are computed at startup)

3. **Test new category:**
```bash
curl -X POST "http://localhost:8000/classifier/classify" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/car.jpg"}'
```

#### Tuning the Threshold

The threshold determines when an image is classified as "other":

- **Lower threshold (0.2-0.3):** More permissive, fewer "other" classifications
- **Higher threshold (0.4-0.6):** More strict, more "other" classifications
- **Default (0.35):** Balanced approach

**Testing threshold values:**
```python
# Test with different thresholds
for threshold in [0.2, 0.3, 0.35, 0.4, 0.5]:
    config.threshold = threshold
    # Test on validation set and measure accuracy
```

#### Using Different Models

**Available SigLIP variants:**
- `google/siglip-base-patch16-512` (default, good balance)
- `google/siglip-large-patch16-512` (better accuracy, slower)
- `google/siglip-base-patch16-224` (faster, lower accuracy)

**Switching models:**
```bash
# In .env file
CLASSIFIER_MODEL=google/siglip-large-patch16-512
```

### Performance Optimization

#### Memory Usage

**Category embeddings:** ~45KB for typical setup (15 prompts)
- Each embedding: ~3KB (768 dimensions × 4 bytes)
- Scales linearly with number of prompts
- Maximum realistic usage: ~1-2MB (hundreds of categories)

#### Inference Speed

**Breakdown per request:**
- Image download: 0.5-3.0 seconds (network dependent)
- Image encoding: 0.1-0.3 seconds (GPU dependent)  
- Similarity computation: <0.01 seconds (CPU)
- **Total:** ~0.6-3.3 seconds per image

**Optimization strategies:**
1. **Pre-downloaded images:** Skip download step for local files
2. **Batch processing:** Process multiple images in parallel
3. **GPU acceleration:** Ensure CUDA is available and used
4. **Model size:** Use smaller models for speed-critical applications

#### Concurrent Requests

The classifier supports concurrent requests:
```python
# Multiple requests can be processed simultaneously
async def process_concurrent(urls):
    tasks = [service.classify_image(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results
```

### Troubleshooting

#### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
*Solution:* Use CPU mode or smaller model:
```bash
CUDA_AVAILABLE=false  # Force CPU mode
# OR
CLASSIFIER_MODEL=google/siglip-base-patch16-224  # Smaller model
```

**2. Image Download Timeouts**
```
"error": "Download timeout after 30.0 seconds"
```
*Solution:* Check network connectivity and image URL validity

**3. Invalid Image Format**
```
"error": "Failed to process image: cannot identify image file"
```
*Solution:* Ensure URL points to valid image format (JPEG, PNG, etc.)

**4. Low Classification Accuracy**
```
Many images classified as "other"
```
*Solution:* 
- Lower the threshold value
- Add more prompt variations
- Improve prompt phrasing

#### Debugging Tips

**1. Enable detailed logging:**
```python
import logging
logging.basicConfig(level=logging.INFO)
```

**2. Check similarity scores:**
```python
# The service logs all similarity scores
# Look for patterns in misclassified images
```

**3. Test individual components:**
```python
# Test image loading
image, error = await loader.load_image(url)

# Test model encoding  
embedding = model.encode_image(image)

# Test similarity computation
similarities = model.compute_similarity(image_emb, text_emb)
```

### Best Practices

#### Prompt Engineering

1. **Use concrete descriptions:** "a photo of a building" vs "building"
2. **Include variations:** formal and informal language  
3. **Consider context:** "photo of a painting" for artworks in images
4. **Test iteratively:** Monitor classification results and refine

#### Production Deployment

1. **Health checks:** Monitor startup success and model availability
2. **Error monitoring:** Track classification failures and timeouts
3. **Performance metrics:** Monitor response times and accuracy
4. **Resource monitoring:** Track GPU memory and CPU usage

#### Category Design

1. **Mutually exclusive:** Avoid overlapping categories when possible
2. **Balanced coverage:** Similar numbers of prompts per category
3. **Domain-specific:** Tailor prompts to your specific use case
4. **Regular evaluation:** Test accuracy on representative images

This classifier provides a flexible, production-ready solution for zero-shot image classification that can be easily extended and customized for different use cases.