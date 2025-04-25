import os

# Giảm số lượng workers xuống cho bản Render miễn phí
workers = 1
threads = 2
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
worker_class = "gthread"
worker_connections = 500
timeout = 180 # Tăng timeout lên để xử lý các tác vụ lâu hơn
keepalive = 5
preload_app = True # Tải trước ứng dụng để model được load ngay từ đầu 