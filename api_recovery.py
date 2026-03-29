import asyncio
import logging
import os
from typing import List, Optional, Dict
import requests
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from embedding_handler import Dino2ExtractorV1, EmbeddingService, URLImageLoader, InternVIT600mbExtractor, \
    Dino3ExtractorV1
from classifier.config import ClassifierConfig
from classifier.classifier_service import ClassifierService
from dotenv import load_dotenv

load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

fastapi_logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    url: str


class EmbeddingItem(BaseModel):
    id: int
    url: str


class BatchEmbeddingRequestDict(BaseModel):
    items: Dict[int, str]


# Выходная модель: словарь id -> embedding
class BatchEmbeddingResponseDict(BaseModel):
    embeddings: Dict[int, Optional[list[float]]]
    errors: Dict[int, str]
    elapsed_sec: float


class BatchClassifyItem(BaseModel):
    id: int
    url: str


class BatchClassifyRequest(BaseModel):
    items: List[BatchClassifyItem]


class BatchClassifyResponseItem(BaseModel):
    id: int
    success: bool
    category: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    elapsed_sec: Optional[float] = None


class BatchClassifyResponse(BaseModel):
    results: Dict[int, BatchClassifyResponseItem]
    total_processed: int
    total_time_sec: float
    errors_count: int


results = {}
AUTH_TOKEN = os.getenv('TOKEN')
embedding_service = EmbeddingService(URLImageLoader(), Dino3ExtractorV1())

# Initialize classifier service
classifier_config = ClassifierConfig()
classifier_service = ClassifierService(classifier_config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    fastapi_logger.info("Starting service initialization")

    # Initialize classifier service
    try:
        fastapi_logger.info("Initializing classifier service...")
        await classifier_service.initialize()
        fastapi_logger.info("Classifier service initialized successfully")
    except Exception as e:
        fastapi_logger.error(f"CRITICAL: Failed to initialize classifier service: {str(e)}")
        fastapi_logger.error("Service will continue but classification endpoints will not work")
        # Note: We continue startup to allow health checks and embedding endpoints to work

    # IP handler
    fastapi_logger.info("IP handler starting")
    try:
        ip = requests.get("https://api.ipify.org").text
        response = requests.post(
            "https://mb.artcracker.io/api/v1/update_embedding_api",
            json={"ip": ip},
            headers={
                "Authorization": f"Token {AUTH_TOKEN}",
                "Content-Type": "application/json",
                "User-Agent": "embedding-service/1.0"
            },
            timeout=10
        )
        if response.status_code in (200, 201):
            fastapi_logger.info(f"IP registered successfully: {ip}")
        else:
            fastapi_logger.warning(f"Failed to register IP: {response.status_code}, {response.text}")
    except Exception as e:
        fastapi_logger.error(f"Error registering IP: {str(e)}")

    fastapi_logger.info("Service initialization complete")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/embedding/fast_extract")
async def extract_embedding(request: EmbeddingRequest):
    start = time.perf_counter()
    fastapi_logger.info(f"Embedding extraction request: {request.url}")

    result = await run_in_threadpool(embedding_service.extract, request.url)
    embedding = result.tolist()

    elapsed = time.perf_counter() - start
    fastapi_logger.info(f"Embedding extraction completed in {elapsed:.2f}s")

    return {"embedding": embedding, "url": request.url}


@app.get("/")
async def root():
    return {"message": "embedding service is up and running!"}


class ClassifyRequest(BaseModel):
    url: str


@app.post("/classifier/classify")
async def classify_image(request: ClassifyRequest):
    """
    Classify an image into predefined categories using zero-shot classification.

    Request:
        {"url": "https://example.com/image.jpg"}

    Response:
        {
            "success": true,
            "category": "building",
            "confidence": 0.87,
            "error": null
        }
    """
    start_time = time.perf_counter()
    fastapi_logger.info(f"Classification request: {request.url}")

    # Check if classifier service is initialized
    if not classifier_service._initialized:
        fastapi_logger.error("Classification request rejected: service not initialized")
        return {
            "success": False,
            "category": None,
            "confidence": None,
            "error": "Classification service is not available. Please check server logs."
        }

    # Classify image (service handles all errors internally)
    result = await classifier_service.classify_image(request.url)

    elapsed = time.perf_counter() - start_time

    if result.success:
        fastapi_logger.info(
            f"Classification completed in {elapsed:.2f}s: {result.category} (confidence: {result.confidence:.3f})")
    else:
        fastapi_logger.warning(f"Classification failed in {elapsed:.2f}s: {result.error}")

    return result.to_dict()


@app.post("/classifier/classify_batch", response_model=BatchClassifyResponse)
async def classify_batch(request: BatchClassifyRequest):
    """
    Batch zero-shot classification of multiple images

    Пример запроса:
    {
      "items": [
        {"id": 1, "url": "https://example.com/img1.jpg"},
        {"id": 2, "url": "https://example.com/img2.png"},
        {"id": 5, "url": "https://..."}
      ]
    }
    """
    if not classifier_service._initialized:
        raise HTTPException(
            status_code=503,
            detail="Classification service is not initialized"
        )

    if not request.items:
        return BatchClassifyResponse(
            results={},
            total_processed=0,
            total_time_sec=0.0,
            errors_count=0
        )

    start_total = time.perf_counter()

    # Будем обрабатывать изображения последовательно из-за семафора
    # (если хочешь параллелизм → увеличивай Semaphore или делай пул задач)
    results: Dict[int, BatchClassifyResponseItem] = {}

    semaphore = embedding_semaphore  # используем тот же лимит, что и для эмбеддингов (1)

    async def classify_one(item: BatchClassifyItem) -> BatchClassifyResponseItem:
        start = time.perf_counter()
        try:
            async with semaphore:
                result = await classifier_service.classify_image(item.url)

            elapsed = time.perf_counter() - start

            if result.success:
                fastapi_logger.info(
                    f"[batch] #{item.id} → {result.category} ({result.confidence:.3f}) "
                    f"in {elapsed:.2f}s"
                )
            else:
                fastapi_logger.warning(
                    f"[batch] #{item.id} failed: {result.error} ({elapsed:.2f}s)"
                )

            return BatchClassifyResponseItem(
                id=item.id,
                success=result.success,
                category=result.category,
                confidence=result.confidence,
                error=result.error,
                elapsed_sec=round(elapsed, 3)
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            msg = f"Unexpected error: {str(e)}"
            fastapi_logger.exception(f"[batch] #{item.id} → {msg}")
            return BatchClassifyResponseItem(
                id=item.id,
                success=False,
                category=None,
                confidence=None,
                error=msg,
                elapsed_sec=round(elapsed, 3)
            )

    # Последовательное выполнение (с семафором внутри)
    # Если хочешь распараллелить → см. ниже вариант с gather + ограничением concurrency
    for item in request.items:
        res = await classify_one(item)
        results[item.id] = res

    total_time = time.perf_counter() - start_total
    errors_count = sum(1 for r in results.values() if not r.success)

    return BatchClassifyResponse(
        results=results,
        total_processed=len(request.items),
        total_time_sec=round(total_time, 3),
        errors_count=errors_count
    )


embedding_semaphore = asyncio.Semaphore(1)  # максимум 1 запрос к модели одновременно



