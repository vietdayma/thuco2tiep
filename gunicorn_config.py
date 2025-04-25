import os

workers = 4
threads = 2
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
worker_class = "gthread"
worker_connections = 1000
timeout = 120
keepalive = 5 