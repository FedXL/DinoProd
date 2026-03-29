FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем зависимости
COPY l-req.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r l-req.txt

# Копируем исходники
COPY . .

EXPOSE 8000

# Запуск (используем порт 8000 внутри контейнера)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
