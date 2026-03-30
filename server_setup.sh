#!/bin/bash
# =============================================================================
# server_setup.sh — Подготовка Ubuntu-сервера для запуска docker-compose
# с поддержкой NVIDIA GPU.
#
# Протестировано на: Ubuntu 22.04, NVIDIA GeForce RTX 3090,
#                    Driver 580.95.05, Docker 24+
#
# Запуск: sudo bash server_setup.sh
# =============================================================================

set -e  # остановить при любой ошибке

echo "=== [1/4] Добавление репозитория nvidia-container-toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --batch --yes \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "=== [2/4] Установка nvidia-container-toolkit ==="
apt-get update -qq
apt-get install -y nvidia-container-toolkit

echo "=== [3/4] Настройка Docker runtime для NVIDIA ==="
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "=== [4/4] Проверка GPU внутри Docker ==="
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "======================================================"
echo " Готово! Теперь можно запускать docker-compose:"
echo "   cd /path/to/dino && docker compose up --build -d"
echo "======================================================"
