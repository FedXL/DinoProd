# DinoProd — Карта проекта

## Назначение сервиса

Сервис извлечения эмбеддингов и классификации изображений для платформы **ArtCracker** (`mb.artcracker.io`).
Принимает URL изображения → возвращает векторное представление (embedding) или категорию.

Работает в двух режимах:
- **API-режим** — FastAPI сервер на порту 8000, обрабатывает запросы в реальном времени
- **Batch-режим** — обходит все коллекции и здания из ArtCracker, считает эмбеддинги оптом

---

## Структура файлов

```
dino/
├── api.py                   # FastAPI приложение — точка входа для API-режима
├── start.py                 # Точка входа для batch-режима
├── embedding_handler.py     # Модели и сервис извлечения эмбеддингов
├── controller.py            # Batch-воркфлоу (обход коллекций и зданий)
├── request_handler.py       # HTTP-клиент к ArtCracker API
├── classifier/
│   ├── model.py             # SigLIP-2 модель (кодирование текста и изображений)
│   ├── classifier_service.py# Сервис классификации (логика + кэш эмбеддингов)
│   ├── image_loader.py      # Async загрузка изображений по URL
│   └── config.py            # Конфиг классификатора (модель, порог, категории)
├── config/
│   └── categories.json      # Текстовые промпты для zero-shot классификации
├── Dockerfile               # CUDA 11.8 + Python 3.11
├── docker-compose.yml       # Запуск с GPU (nvidia), порт 8000
└── .env                     # TOKEN, HG_TOKEN, LOG_LEVEL (не в репо)
```

---

## API эндпоинты

### Здоровье сервиса

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/` | Health-check. Возвращает `{"message": "embedding service is up and running!"}` |

---

### Эмбеддинги (`/embedding/`)

#### `POST /embedding/fast_extract`
Извлекает эмбеддинг одного изображения через модель **DINOv3** (facebook/dinov3-vitl16).

**Запрос:**
```json
{ "url": "https://example.com/image.jpg" }
```
**Ответ:**
```json
{ "embedding": [0.12, -0.04, ...], "url": "https://example.com/image.jpg" }
```
- Модель: `Dino3ExtractorV1` (DINOv3 vitl16, CLS-token)
- Конкурентность: семафор 1 (одновременно одна инференция)

---

#### `POST /embedding/fast_extract_batch`
Батч-обработка эмбеддингов. Принимает словарь `id → url`, возвращает словарь `id → embedding`.

**Запрос:**
```json
{
  "items": {
    "1": "https://example.com/img1.jpg",
    "2": "https://example.com/img2.jpg"
  }
}
```
**Ответ:**
```json
{
  "embeddings": { "1": [0.12, ...], "2": null },
  "errors":     { "2": "Connection timeout" },
  "elapsed_sec": 3.142
}
```
- При ошибке для конкретного ID: `embeddings[id] = null`, причина в `errors[id]`

---

### Классификация (`/classifier/`)

#### `POST /classifier/classify`
Zero-shot классификация одного изображения в категории: **building / painting / other**.

**Запрос:**
```json
{ "url": "https://example.com/image.jpg" }
```
**Ответ (успех):**
```json
{
  "success": true,
  "category": "building",
  "confidence": 0.87,
  "error": null
}
```
**Ответ (ошибка / сервис не инициализирован):**
```json
{
  "success": false,
  "category": null,
  "confidence": null,
  "error": "Classification service is not available."
}
```
- Модель: `SigLIPModel` (google/siglip2-base-patch16-512)
- Если `confidence < threshold` → категория принудительно `"other"`

---

#### `POST /classifier/classify_batch`
Батч-классификация списка изображений.

**Запрос:**
```json
{
  "items": [
    { "id": 1, "url": "https://example.com/img1.jpg" },
    { "id": 2, "url": "https://example.com/img2.jpg" }
  ]
}
```
**Ответ:**
```json
{
  "results": {
    "1": { "id": 1, "success": true, "category": "painting", "confidence": 0.91, "error": null, "elapsed_sec": 0.45 },
    "2": { "id": 2, "success": false, "category": null, "confidence": null, "error": "Failed to load image", "elapsed_sec": 1.2 }
  },
  "total_processed": 2,
  "total_time_sec": 1.65,
  "errors_count": 1
}
```
- Если сервис не инициализирован → HTTP 503

---

## Модели

| Класс | Модель HuggingFace | Размерность | Применение |
|-------|-------------------|-------------|------------|
| `Dino3ExtractorV1` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | зависит от модели (CLS-token) | Основной экстрактор (активен в API) |
| `Dino2ExtractorV1` | `facebookresearch/dinov2` → `dinov2_vitg14` | 1536 | Batch-обработка зданий |
| `InternVIT600mbExtractor` | `OpenGVLab/InternViT-300M-448px-V2_5` | 512 | Альтернативный экстрактор (не активен) |
| `SigLIPModel` | `google/siglip2-base-patch16-512` | — | Классификатор изображений |

Требования: CUDA GPU, Python 3.11, CUDA 11.8+.

---

## Клиент ArtCracker (`request_handler.py`)

HTTP-клиент к внешнему API `mb.artcracker.io`. Аутентификация через `TOKEN` из `.env`.

| Функция | Метод | URL | Описание |
|---------|-------|-----|----------|
| `get_building_images()` | `GET` | `/api/v1/building-images/` | Список изображений зданий |
| `get_collections_names_list()` | `GET` | `/api/v1/collections/` | Список имён коллекций |
| `get_task_images_from_collection(name)` | `POST` | `/api/v1/collection_tasks` | Изображения задач из коллекции |
| `send_building_image_embedding(data)` | `POST` | `/api/v1/emb_handler` | Отправить эмбеддинг изображения здания |
| `send_task_image_embedding(data)` | `POST` | `/api/v1/emb_handler_task` | Отправить эмбеддинг задачи коллекции |
| `send_image_to_building_images(id, file)` | `POST` | `/api/v1/building-images/` | Загрузить изображение в базу здания |
| `create_new_building_in_mb(name)` | `POST` | `/api/v1/buildings/` | Создать новую запись здания |

---

## Batch-режим (`controller.py` + `start.py`)

`python3.11 start.py` → запускает `main_flow2()`:

1. Получает список всех коллекций из ArtCracker
2. Для каждой коллекции → извлекает эмбеддинги задач (`Dino2ExtractorV1`) → отправляет обратно
3. Получает все изображения зданий → извлекает эмбеддинги → отправляет обратно
4. Результаты сохраняются локально в `results/` и `results2/`

---

## Запуск

```bash
# API-сервер
uvicorn api:app --host 0.0.0.0 --port 8000

# Docker
docker compose up

# Batch-режим
python3.11 start.py
```

**Переменные окружения (`.env`):**

| Переменная | Назначение |
|------------|-----------|
| `TOKEN` | Токен авторизации для ArtCracker API |
| `HG_TOKEN` | Токен HuggingFace для загрузки моделей |
| `LOG_LEVEL` | Уровень логирования (по умолчанию `DEBUG`) |

---

## Жизненный цикл при старте API

1. Загружается `Dino3ExtractorV1` (DINOv3) в GPU-память
2. Загружается `SigLIPModel` (SigLIP-2) — ~25-30 сек
3. Вычисляются текстовые эмбеддинги для всех промптов категорий — ~2-3 сек, кэшируются
4. Регистрируется внешний IP сервиса в ArtCracker (`/api/v1/update_embedding_api`)
5. Сервис готов к приёму запросов
