import os
import multiprocessing

# Cấu hình worker cho gunicorn - Tối ưu hóa cho trường hợp nhiều request đồng thời
workers = 1  # Giảm xuống 1 worker để tránh cạnh tranh tài nguyên - tốt hơn cho mô hình ML
threads = 8  # Tăng số lượng thread lên 8 để xử lý nhiều request đồng thời trong một worker

# Cấu hình kết nối - Địa chỉ IP và cổng để lắng nghe request
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"  # Lắng nghe trên tất cả các địa chỉ IP, port từ biến môi trường hoặc mặc định 10000

# Cấu hình worker class và connections
worker_class = "gthread"  # Sử dụng gthread để tận dụng khả năng đa luồng Python
worker_connections = 1000  # Mỗi worker có thể xử lý đồng thời tối đa 1000 kết nối

# Cấu hình vòng đời worker và độ tin cậy
max_requests = 10000  # Worker sẽ khởi động lại sau khi xử lý 10000 request để tránh rò rỉ bộ nhớ
max_requests_jitter = 1000  # Thêm sự dao động ngẫu nhiên để tránh tất cả worker khởi động lại cùng lúc
timeout = 300  # Thời gian tối đa (giây) để xử lý một request trước khi worker bị kill
keepalive = 120  # Thời gian (giây) giữ kết nối HTTP mở để tái sử dụng

# Tối ưu hiệu suất khởi động
preload_app = True  # Tải ứng dụng trước khi fork worker - giúp khởi tạo mô hình ML một lần duy nhất
graceful_timeout = 60  # Thời gian chờ tối đa (giây) trước khi buộc worker dừng

# Giảm tải I/O và overhead cho server
reload_extra_files = []  # Không cần theo dõi các file để tự động reload
reload = False  # Tắt auto-reload để tối ưu performance trên production
check_config = False  # Tắt kiểm tra cấu hình khi khởi động để khởi động nhanh hơn

# Giảm logging để tối ưu hiệu suất
loglevel = 'error'  # Chỉ ghi log lỗi nghiêm trọng
accesslog = None  # Tắt access log để giảm I/O
errorlog = '-'  # Ghi error log ra stderr

# Cấu hình loại bỏ - Được giữ lại trong file để dễ tham khảo
# post_worker_init = None  # Không sử dụng hàm callback sau khi worker được khởi tạo
# post_fork = None  # Không sử dụng hàm callback sau khi fork worker 