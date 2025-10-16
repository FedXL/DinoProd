import asyncio
import logging
import os
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
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

fastapi_logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    url: str

results = {}
AUTH_TOKEN = os.getenv('TOKEN')
embedding_service = EmbeddingService(URLImageLoader(), Dino3ExtractorV1())
# embedding_vit_600m = EmbeddingService(URLImageLoader(), InternVIT600mbExtractor())

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
        fastapi_logger.info(f"Classification completed in {elapsed:.2f}s: {result.category} (confidence: {result.confidence:.3f})")
    else:
        fastapi_logger.warning(f"Classification failed in {elapsed:.2f}s: {result.error}")
    
    return result.to_dict()




embedding_semaphore = asyncio.Semaphore(1)  # максимум 1 запрос к модели одновременно



# @app.post("/embedding/test_extract")
# async def extract_embedding(request: EmbeddingRequest):
#     start = time.perf_counter()
#     print(f"\n[{request.url}] 🌐 Запрос получен")
#
#     # 🔄 1. Параллельно загружаем изображение
#     try:
#         image, message = embedding_vit_600m.loader.load(request.url)
#         if image is None:
#             raise ValueError(message)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
#
#     loaded = time.perf_counter()
#     print(f"[{request.url}] ✅ Изображение загружено за {loaded - start:.2f} сек")
#
#     try:
#         async with asyncio.timeout(10):  # таймаут ожидания очереди
#             queue_start = time.perf_counter()
#             print(f"[{request.url}] ⏳ Ожидаем доступ к модели...")
#
#             async with embedding_semaphore:
#                 waited = time.perf_counter()
#                 print(f"[{request.url}] 🔓 Доступ получен через {waited - queue_start:.2f} сек")
#
#                 # 💡 3. Извлекаем эмбеддинг
#                 result = embedding_vit_600m.extractor.extract(image)
#                 embedding = result.tolist()
#
#                 finished = time.perf_counter()
#                 print(f"[{request.url}] 🧠 Обработка завершена за {finished - waited:.2f} сек")
#     except TimeoutError:
#         raise HTTPException(status_code=503, detail="Модель занята. Повторите позже.")
#
#     total = time.perf_counter()
#     print(f"[{request.url}] ✅ Общая длительность: {total - start:.2f} сек")
#
#     return {"embedding": embedding, "url": request.url}