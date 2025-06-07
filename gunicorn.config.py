# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 1  # Single worker to avoid memory issues
worker_class = "sync"
worker_timeout = 300  # 5 minutes
max_requests = 50
max_requests_jitter = 10
preload_app = True
memory_limit = 400 * 1024 * 1024  # 400MB limit