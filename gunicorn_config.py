import os
import multiprocessing

# Tối ưu cho benchmark với nhiều request đồng thời
workers = 1  # Giảm xuống 1 worker để tránh cạnh tranh tài nguyên
threads = 8  # Tăng threads lên nhiều hơn để xử lý đồng thời nhiều requests
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
worker_class = "gthread"  # Sử dụng gthread để hỗ trợ threading 
worker_connections = 1000  # Tăng số kết nối
max_requests = 10000  # Tăng số request tối đa trước khi restart worker
max_requests_jitter = 1000  # Thêm jitter để tránh restart đồng thời
timeout = 300  # Tăng timeout lên cao hơn để tránh worker bị kill giữa chừng
keepalive = 120  # Tăng keepalive để giữ kết nối HTTP lâu hơn
preload_app = True  # Tải trước ứng dụng để model được load ngay từ đầu
graceful_timeout = 60  # Giảm graceful timeout để giải phóng tài nguyên nhanh hơn

# Giảm overhead
reload_extra_files = []
reload = False
check_config = False

# Giảm logging để tối ưu hiệu suất
loglevel = 'error'
accesslog = None  # Tắt access log
errorlog = '-'  # Ghi error log ra stderr

# Tối ưu buffer
post_worker_init = None
post_fork = None 