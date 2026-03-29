module.exports = {
  apps: [
    // === Основное приложение: fastapi-app (порт 8000, venv) ===
    {
      name: 'fastapi-app',
      cwd: '/home/fedorkuruts/service/dino',
      script: './venv/bin/gunicorn',
      args: 'api:app --bind 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker',
      interpreter: 'none',
      watch: false,
      env: {
        PYTHONPATH: '/home/fedorkuruts/service/dino'
      }
    },

    // === Новое приложение: one-worker (порт 8001, venvexp) ===
    {
      name: 'one-worker',
      cwd: '/home/fedorkuruts/service/dino',
      script: '/home/fedorkuruts/service/dino/venvexp/bin/uvicorn',
      args: 'api_exp:app_exp --host 0.0.0.0 --port 8001 --workers 1',
      interpreter: '/home/fedorkuruts/service/dino/venvexp/bin/python',
      watch: false,
      env: {
        PYTHONPATH: '/home/fedorkuruts/service/dino',
        ENV: 'production'
      },
      autorestart: true,
      max_memory_restart: '10G'
    }
  ]
};
