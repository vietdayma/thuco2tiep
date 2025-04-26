import os
import multiprocessing

# Tối ưu cho benchmark với nhiều request đồng thời
# Cấu hình các worker - đơn vị xử lý request
workers = 1  # Giảm xuống 1 worker để tránh cạnh tranh tài nguyên trên môi trường giới hạn
threads = 8  # Tăng threads lên để xử lý nhiều request đồng thời trong cùng một process

# Cấu hình kết nối
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"  # Địa chỉ và cổng lắng nghe request
worker_class = "gthread"  # Loại worker sử dụng threading để hỗ trợ xử lý đồng thời
worker_connections = 1000  # Số kết nối tối đa mỗi worker có thể xử lý

# Cấu hình quản lý worker
max_requests = 10000  # Số request tối đa mỗi worker xử lý trước khi tự khởi động lại
max_requests_jitter = 1000  # Giá trị ngẫu nhiên thêm vào max_requests để tránh tất cả worker khởi động lại cùng lúc

# Cấu hình timeout
timeout = 300  # Thời gian tối đa (giây) cho một worker xử lý request trước khi bị kill
keepalive = 120  # Thời gian (giây) giữ kết nối HTTP mở sau khi xử lý request
preload_app = True  # Tải ứng dụng trước khi fork worker để tăng tốc khởi động
graceful_timeout = 60  # Thời gian (giây) chờ worker kết thúc "nhẹ nhàng" trước khi bị kill

# Giảm overhead - tối ưu hóa hiệu suất
reload_extra_files = []  # Danh sách file được theo dõi để tự động tải lại (không dùng trong production)
reload = False  # Tắt chế độ tự động tải lại khi có thay đổi mã nguồn

# Cấu hình logging - giảm logging để tăng hiệu suất
check_config = False  # Tắt kiểm tra cấu hình để tăng tốc khởi động
loglevel = 'error'  # Chỉ ghi log lỗi nghiêm trọng
accesslog = None  # Tắt access log để giảm I/O
errorlog = '-'  # Ghi error log ra stderr

# Xóa các cấu hình không hợp lệ
# post_worker_init = None
# post_fork = None 