"""
tests.py — Функциональные тесты для ml-classifier API.

Запуск:
    python3 tests.py
    python3 tests.py --url http://your-server:8000
"""

import sys
import argparse
import requests

# --- Конфиг ---

DEFAULT_URL = "http://localhost:8000"

# Публичные картинки для тестов
IMG_BUILDING = "https://images.unsplash.com/photo-1467269204594-9661b134dd2b?w=400"
IMG_MOUNTAIN = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"
IMG_PAINTING = "https://images.unsplash.com/photo-1551009175-15bdf9dcb580?w=400"
IMG_BROKEN   = "https://example.com/nonexistent_image_404.jpg"

EMBEDDING_DIM = 1024  # DINOv3 vitl16

# --- Утилиты ---

passed = 0
failed = 0

def ok(name):
    global passed
    passed += 1
    print(f"  [PASS] {name}")

def fail(name, reason):
    global failed
    failed += 1
    print(f"  [FAIL] {name} — {reason}")

def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

# --- Тесты ---

def test_health(base):
    section("GET / — health check")
    r = requests.get(f"{base}/")
    if r.status_code == 200:
        ok("статус 200")
    else:
        fail("статус 200", f"получили {r.status_code}")

    data = r.json()
    if "message" in data:
        ok("поле message присутствует")
    else:
        fail("поле message присутствует", f"ответ: {data}")


def test_fast_extract(base):
    section("POST /embedding/fast_extract")
    r = requests.post(f"{base}/embedding/fast_extract", json={"url": IMG_MOUNTAIN})

    if r.status_code == 200:
        ok("статус 200")
    else:
        fail("статус 200", f"получили {r.status_code}: {r.text}")
        return

    data = r.json()

    if "embedding" in data:
        ok("поле embedding присутствует")
    else:
        fail("поле embedding присутствует", str(data))
        return

    emb = data["embedding"]
    if len(emb) == EMBEDDING_DIM:
        ok(f"размерность эмбеддинга = {EMBEDDING_DIM}")
    else:
        fail(f"размерность эмбеддинга = {EMBEDDING_DIM}", f"получили {len(emb)}")

    if all(isinstance(x, float) for x in emb[:10]):
        ok("значения эмбеддинга — float")
    else:
        fail("значения эмбеддинга — float", str(emb[:5]))

    if data.get("url") == IMG_MOUNTAIN:
        ok("url в ответе совпадает с запросом")
    else:
        fail("url в ответе совпадает с запросом", data.get("url"))


def test_fast_extract_batch(base):
    section("POST /embedding/fast_extract_batch")
    payload = {"items": {"1": IMG_MOUNTAIN, "2": IMG_PAINTING}}
    r = requests.post(f"{base}/embedding/fast_extract_batch", json=payload)

    if r.status_code == 200:
        ok("статус 200")
    else:
        fail("статус 200", f"{r.status_code}: {r.text}")
        return

    data = r.json()

    for id_ in ["1", "2"]:
        emb = data["embeddings"].get(id_)
        if emb and len(emb) == EMBEDDING_DIM:
            ok(f"id={id_}: эмбеддинг получен, dim={len(emb)}")
        else:
            fail(f"id={id_}: эмбеддинг получен", f"emb={emb}")

    if isinstance(data.get("elapsed_sec"), float):
        ok(f"elapsed_sec присутствует ({data['elapsed_sec']}s)")
    else:
        fail("elapsed_sec присутствует", str(data.get("elapsed_sec")))


def test_fast_extract_batch_with_broken_url(base):
    section("POST /embedding/fast_extract_batch — сломанный URL")
    payload = {"items": {"1": IMG_MOUNTAIN, "99": IMG_BROKEN}}
    r = requests.post(f"{base}/embedding/fast_extract_batch", json=payload)

    if r.status_code == 200:
        ok("статус 200 даже при частичной ошибке")
    else:
        fail("статус 200 даже при частичной ошибке", f"{r.status_code}")
        return

    data = r.json()

    if data["embeddings"].get("1") and len(data["embeddings"]["1"]) == EMBEDDING_DIM:
        ok("рабочий id=1 вернул эмбеддинг")
    else:
        fail("рабочий id=1 вернул эмбеддинг", str(data["embeddings"].get("1")))

    if data["embeddings"].get("99") is None:
        ok("сломанный id=99 вернул null")
    else:
        fail("сломанный id=99 вернул null", str(data["embeddings"].get("99")))

    if "99" in data["errors"]:
        ok(f"ошибка для id=99 зафиксирована: {data['errors']['99'][:60]}")
    else:
        fail("ошибка для id=99 зафиксирована", str(data.get("errors")))


def test_classify(base):
    section("POST /classifier/classify")

    # здание
    r = requests.post(f"{base}/classifier/classify", json={"url": IMG_BUILDING})
    if r.status_code == 200:
        ok("статус 200 (здание)")
    else:
        fail("статус 200 (здание)", f"{r.status_code}: {r.text}")
        return

    data = r.json()
    if data.get("success") is True:
        ok("success=true")
    else:
        fail("success=true", str(data))

    if data.get("category") in ("building", "painting", "other"):
        ok(f"category валидная: {data['category']}")
    else:
        fail("category валидная", str(data.get("category")))

    conf = data.get("confidence")
    if isinstance(conf, float) and 0.0 <= conf <= 1.0:
        ok(f"confidence в диапазоне [0,1]: {round(conf, 3)}")
    else:
        fail("confidence в диапазоне [0,1]", str(conf))


def test_classify_batch(base):
    section("POST /classifier/classify_batch")
    payload = {
        "items": [
            {"id": 1, "url": IMG_BUILDING},
            {"id": 2, "url": IMG_MOUNTAIN},
        ]
    }
    r = requests.post(f"{base}/classifier/classify_batch", json=payload)

    if r.status_code == 200:
        ok("статус 200")
    else:
        fail("статус 200", f"{r.status_code}: {r.text}")
        return

    data = r.json()

    if data.get("total_processed") == 2:
        ok("total_processed=2")
    else:
        fail("total_processed=2", str(data.get("total_processed")))

    for id_ in ["1", "2"]:
        item = data["results"].get(id_)
        if item and item.get("success"):
            ok(f"id={id_}: category={item['category']}, confidence={round(item['confidence'],3)}")
        else:
            fail(f"id={id_}: успешная классификация", str(item))

    if isinstance(data.get("total_time_sec"), float):
        ok(f"total_time_sec={data['total_time_sec']}s")
    else:
        fail("total_time_sec присутствует", str(data.get("total_time_sec")))


# --- Точка входа ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL сервиса")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"\nТестируем: {base}\n")

    test_health(base)
    test_fast_extract(base)
    test_fast_extract_batch(base)
    test_fast_extract_batch_with_broken_url(base)
    test_classify(base)
    test_classify_batch(base)

    print(f"\n{'='*50}")
    print(f"  Итого: {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    sys.exit(0 if failed == 0 else 1)
