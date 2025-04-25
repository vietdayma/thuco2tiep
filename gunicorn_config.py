import os
import multiprocessing

# Đối với môi trường miễn phí của Render, tốt nhất là giữ số workers thấp
# nhưng tăng số threads lên để xử lý nhiều request đồng thời
workers = 2  # Tăng lên 2 workers để cân bằng
threads = 4  # Tăng threads lên 4 để xử lý nhiều requests hơn
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
worker_class = "gthread"  # Sử dụng gthread để hỗ trợ threading 
worker_connections = 500
max_requests = 1000  # Giới hạn số request tối đa trước khi restart worker
max_requests_jitter = 100  # Thêm jitter để tránh restart tất cả workers cùng lúc
timeout = 180  # Tăng timeout lên để xử lý các tác vụ lâu hơn
keepalive = 65  # Tăng keepalive để giữ kết nối lâu hơn
preload_app = True  # Tải trước ứng dụng để model được load ngay từ đầu
graceful_timeout = 120  # Thời gian chờ worker kết thúc gracefully

# Giúp giảm memory leaks
reload_extra_files = []
reload = False

# Giảm level log để cải thiện hiệu suất
loglevel = 'warning' 