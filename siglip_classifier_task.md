# Zero-Shot Image Classifier Implementation Task

## Project Overview

Implement a zero-shot image classifier service that distinguishes between photos of buildings, paintings/artwork, and other images using Google's SigLIP-2 model family. The service should provide a REST API endpoint with category text embeddings computed once at startup and cached in memory.

## Technical Stack

- **Python Version**: 3.11
- **Framework**: FastAPI (async)
- **Model**: google/siglip2-base-patch16-512 (or other SigLIP-2 variants based on performance)
- **Storage**: In-memory (no database required)
- **Package Manager**: pip
- **Compute**: CUDA (with CPU fallback)

## Project Structure

All new code should be organized in a `classifier/` folder:

```
.
├── api.py                          # Add new endpoint here
├── config/
│   └── categories.json             # Category definitions with prompt variations
├── classifier/                     # NEW - All classifier code goes here
│   ├── __init__.py
│   ├── model.py                    # SigLIP model loading and inference
│   ├── image_loader.py             # Async image downloading and preprocessing
│   ├── classifier_service.py       # Main classification logic with in-memory embedding cache
│   └── config.py                   # Configuration management
├── .env                            # Environment variables
├── req.txt                         # Add new dependencies here
└── CLAUDE.md                       # Project documentation (already exists)
```

## In-Memory Embedding Storage

Category text embeddings are computed once at startup and stored in memory for the lifetime of the service:

- **Storage**: Python dictionary mapping category text → embedding vector
- **Compute time**: ~2-3 seconds at startup for all categories
- **Memory usage**: ~45 KB for typical setup (15 text prompts), max ~1-2 MB even with hundreds of variations
- **Persistence**: None needed - categories rarely change, and recomputation is fast

### Environment Variables

Add these to `.env`:

```bash
# Classifier Configuration
CLASSIFIER_MODEL=google/siglip2-base-patch16-512
CLASSIFIER_THRESHOLD=0.35
CATEGORIES_FILE=config/categories.json

# Existing TOKEN for ArtCracker API (already present)
# TOKEN=your_token
```

## Categories Configuration

Create `config/categories.json` with category definitions:

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
    "painting of a building",
    "artistic work",
    "photo of a sketch"
  ]
}
```

**Note**: No "other" category needed - handled by threshold logic.

## Implementation Requirements

### 1. Model Loading (`classifier/model.py`)

- Load SigLIP-2 model at service startup (similar to existing embedding_handler.py pattern)
- Default to CUDA device with CPU fallback
- Cache model in memory for the lifecycle of the application
- Implement methods:
  - `encode_image(image_tensor)` → returns image embedding
  - `encode_text(text_list)` → returns text embeddings for multiple prompts

### 2. Image Loading (`classifier/image_loader.py`)

- Async image downloading from URL
- Image preprocessing for SigLIP model
- Handle common errors:
  - Invalid URLs
  - Download timeouts (30 seconds max)
  - Unsupported image formats
  - Corrupted images
- Return preprocessed image tensor ready for model

### 3. Classification Service (`classifier/classifier_service.py`)

Main business logic coordinating all components:

**Initialization Flow:**
1. Load categories from `config/categories.json`
2. For each category, expand all prompt variations into a flat list
3. Compute text embeddings for all prompt variations using SigLIP model
4. Store embeddings in memory as a dictionary: `{category_name: embedding_vector}`
5. Log completion: "Loaded X categories with Y total text variations"

**Classification Flow:**
1. Download and preprocess image from URL
2. Compute image embedding using SigLIP model
3. Compare image embedding against all category embeddings (using numpy cosine similarity)
4. Find category with highest similarity score
5. Apply threshold:
   - If `max_score >= threshold`: return that category
   - If `max_score < threshold`: return "other"

**Methods:**
- `async initialize()` → precompute all category embeddings at startup
- `async classify_image(image_url: str)` → main classification method
- `_compute_similarity(image_embedding, category_embeddings)` → helper for cosine similarity

### 4. Configuration Management (`classifier/config.py`)

- Load and validate environment variables
- Load categories from JSON file
- Provide typed configuration object
- Validate required settings on startup

### 5. API Endpoint (`api.py`)

Add new endpoint following existing patterns:

```python
@app.post("/classifier/classify")
async def classify_image(request: ClassifyRequest):
    """
    Classify an image into predefined categories.
    
    Request body:
    {
        "url": "https://example.com/image.jpg"
    }
    
    Response:
    {
        "success": true,
        "category": "building",  // or "painting" or "other"
        "confidence": 0.87,
        "error": null
    }
    
    Error response:
    {
        "success": false,
        "category": null,
        "confidence": null,
        "error": "Failed to download image: timeout"
    }
    """
```

## Response Format Specification

### Success Response
```json
{
  "success": true,
  "category": "building",
  "confidence": 0.87,
  "error": null
}
```

### Error Response
```json
{
  "success": false,
  "category": null,
  "confidence": null,
  "error": "Failed to download image: Connection timeout"
}
```

**Error Types to Handle:**
- Invalid URL format
- Image download failures (timeout, 404, network errors)
- Unsupported image format
- Image preprocessing errors
- Model inference errors

## Dependencies to Add

Add to `req.txt`:

```
Pillow>=10.0.0
numpy>=1.24.0
transformers>=4.35.0
torch>=2.0.0
```

(Note: FastAPI, uvicorn, etc. already present)

## Code Quality Guidelines

- **Async/await**: All I/O operations must be async (image download, database, model inference where possible)
- **Error handling**: Comprehensive try/catch blocks with specific error messages
- **Type hints**: Use Python type hints throughout
- **Logging**: Use Python logging module (similar to existing codebase patterns)
- **No overcomplication**: Keep code clean and straightforward
- **Follow existing patterns**: Match code style from `embedding_handler.py` and `api.py`

## Testing Considerations

After implementation, test with:
1. Building photo URL
2. Painting/artwork photo URL  
3. Random object photo (should return "other")
4. Invalid URL (should return error)
5. Very slow image URL (should timeout)

## Performance Optimization Strategy

1. **Model**: Loaded once at startup, kept in GPU memory
2. **Category embeddings**: Precomputed at startup, always cached in memory
3. **Database**: Optional optimization for faster cold starts
   - If DB available: embeddings persisted and reused across restarts
   - If DB unavailable: embeddings computed at each startup (still fast, ~2-3 seconds)
4. **Image processing**: Async download to not block other requests
5. **Threshold**: Configurable via env var for easy tuning

**Database failure behavior**: Logging will indicate DB status:
- `INFO: Database not configured, running without persistence`
- `WARNING: Failed to connect to database: <error>. Running without caching.`
- `INFO: Successfully connected to database, using cached embeddings`

## Startup Sequence

When the service starts:
1. Load environment configuration
2. Load SigLIP model to GPU
3. Load categories from `config/categories.json`
4. Compute all category text embeddings (~2-3 seconds)
5. Store embeddings in memory
6. Log: "Classifier initialized with X categories (Y text variations)"
7. Start FastAPI server

**Total startup time**: ~30-40 seconds (mostly model loading)

## Success Criteria

- [ ] Classifier service successfully classifies test images
- [ ] Category embeddings computed at startup and cached in memory
- [ ] API returns proper error responses for invalid inputs
- [ ] Service runs asynchronously without blocking
- [ ] Code follows existing project patterns and best practices
- [ ] All dependencies properly added to req.txt
- [ ] Startup completes in ~30-40 seconds
- [ ] Memory usage is reasonable (embeddings < 2 MB)

## Notes

- The SigLIP model family supports various sizes - start with `siglip2-base-patch16-512`, but allow easy swapping via env var
- Threshold value (0.35) is a starting point - may need tuning based on testing
- Each category can have multiple text prompt variations to improve accuracy
- The service is stateless except for the model and in-memory category embedding cache
- Category embeddings are recomputed on each service restart (~2-3 seconds) - this is acceptable given categories rarely change
