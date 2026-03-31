import asyncio
import logging
import os
import time
from typing import List, Optional, Dict

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from embedding_handler import EmbeddingService, URLImageLoader, Dino3ExtractorV1, Dino2ExtractorV1, InternVIT600mbExtractor
from classifier.config import ClassifierConfig
from classifier.classifier_service import ClassifierService

from dotenv import load_dotenv

load_dotenv()

# --- Logging Configuration ---
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
fastapi_logger = logging.getLogger(__name__)

# --- Pydantic Models ---

# Embedding Models
class EmbeddingRequest(BaseModel):
    url: str

class BatchEmbeddingRequestDict(BaseModel):
    items: Dict[int, str]

class BatchEmbeddingResponseDict(BaseModel):
    embeddings: Dict[int, Optional[list[float]]]
    errors: Dict[int, str]
    elapsed_sec: float

# Classifier Models
class ClassifyRequest(BaseModel):
    url: str

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

# --- Global Resources ---
AUTH_TOKEN = os.getenv('TOKEN')
embedding_semaphore = asyncio.Semaphore(1)  # Ограничение на 1 одновременный прогон через модель

embedding_service = EmbeddingService(URLImageLoader(), Dino3ExtractorV1())
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
    
    # IP registration handler
    fastapi_logger.info("IP handler starting")
    try:
        ip = requests.get("https://api.ipify.org").text
        print(f'ЗАГЛУШКА ПОКА НЕ ОТСЫЛАЕМ ЭТОГО айпи микросервиса {ip}')
        # FIXME
        # response = requests.post(
        #     "https://mb.artcracker.io/api/v1/update_embedding_api",
        #     json={"ip": ip},
        #     headers={
        #         "Authorization": f"Token {AUTH_TOKEN}",
        #         "Content-Type": "application/json",
        #         "User-Agent": "embedding-service/1.0"
        #     },
        #     timeout=10
        # )
        # if response.status_code in (200, 201):
        #     fastapi_logger.info(f"IP registered successfully: {ip}")
        # else:
        #     fastapi_logger.warning(f"Failed to register IP: {response.status_code}, {response.text}")
    except Exception as e:
        fastapi_logger.error(f"Error registering IP: {str(e)}")
    
    fastapi_logger.info("Service initialization complete")
    yield

app = FastAPI(lifespan=lifespan)

# --- Endpoints: Embeddings ---

@app.get("/")
async def root():
    return {"message": "embedding service is up and running!"}

@app.post("/embedding/fast_extract")
async def extract_embedding(request: EmbeddingRequest):
    start = time.perf_counter()
    fastapi_logger.info(f"Embedding extraction request: {request.url}")

    async with embedding_semaphore:
        result = await run_in_threadpool(embedding_service.extract, request.url)
    
    embedding = result.tolist()
    elapsed = time.perf_counter() - start
    fastapi_logger.info(f"Embedding extraction completed in {elapsed:.2f}s")

    return {"embedding": embedding, "url": request.url}

@app.post("/embedding/fast_extract_batch", response_model=BatchEmbeddingResponseDict)
async def extract_embeddings_dict(request: BatchEmbeddingRequestDict):
    """
    Батч-обработка эмбеддингов из первого файла.
    """
    start = time.perf_counter()
    embeddings: Dict[int, Optional[list[float]]] = {}
    errors: Dict[int, str] = {}

    for id_, url in request.items.items():
        try:
            # Используем семафор, чтобы не перегрузить память/GPU
            async with embedding_semaphore:
                embedding = await run_in_threadpool(embedding_service.extract, url)
                embeddings[id_] = embedding.tolist()
        except Exception as e:
            fastapi_logger.error(f"Error processing item {id_}: {str(e)}")
            embeddings[id_] = None
            errors[id_] = str(e)

    elapsed = time.perf_counter() - start
    fastapi_logger.info(f"Batch embedding finished in {elapsed:.2f}s")

    return BatchEmbeddingResponseDict(
        embeddings=embeddings,
        errors=errors,
        elapsed_sec=round(elapsed, 3)
    )

# --- Endpoints: Classification ---

@app.post("/classifier/classify")
async def classify_image(request: ClassifyRequest):
    start_time = time.perf_counter()
    fastapi_logger.info(f"Classification request: {request.url}")
    
    if not classifier_service._initialized:
        return {
            "success": False,
            "category": None,
            "confidence": None,
            "error": "Classification service is not available."
        }
    
    async with embedding_semaphore:
        result = await classifier_service.classify_image(request.url)
    
    elapsed = time.perf_counter() - start_time
    if result.success:
        fastapi_logger.info(f"Classification completed in {elapsed:.2f}s: {result.category}")
    else:
        fastapi_logger.warning(f"Classification failed: {result.error}")
    
    return result.to_dict()

@app.post("/classifier/classify_batch", response_model=BatchClassifyResponse)
async def classify_batch(request: BatchClassifyRequest):
    if not classifier_service._initialized:
        raise HTTPException(status_code=503, detail="Classification service not initialized")

    if not request.items:
        return BatchClassifyResponse(results={}, total_processed=0, total_time_sec=0.0, errors_count=0)

    start_total = time.perf_counter()
    final_results: Dict[int, BatchClassifyResponseItem] = {}

    for item in request.items:
        start_item = time.perf_counter()
        try:
            async with embedding_semaphore:
                result = await classifier_service.classify_image(item.url)
            
            elapsed_item = time.perf_counter() - start_item
            
            final_results[item.id] = BatchClassifyResponseItem(
                id=item.id,
                success=result.success,
                category=result.category,
                confidence=result.confidence,
                error=result.error,
                elapsed_sec=round(elapsed_item, 3)
            )
        except Exception as e:
            elapsed_item = time.perf_counter() - start_item
            final_results[item.id] = BatchClassifyResponseItem(
                id=item.id,
                success=False,
                error=str(e),
                elapsed_sec=round(elapsed_item, 3)
            )

    total_time = time.perf_counter() - start_total
    errors_count = sum(1 for r in final_results.values() if not r.success)

    return BatchClassifyResponse(
        results=final_results,
        total_processed=len(request.items),
        total_time_sec=round(total_time, 3),
        errors_count=errors_count
    )
