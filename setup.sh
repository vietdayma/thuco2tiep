mkdir -p ~/.streamlit/  # Tạo thư mục cấu hình Streamlit nếu chưa tồn tại

# Tạo file credentials.toml với thông tin xác thực
# Cần thiết cho Streamlit Cloud để xác định chủ sở hữu ứng dụng
echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

# Tạo file config.toml với cấu hình server
# - headless: chạy mà không mở trình duyệt
# - enableCORS: tắt CORS để cho phép request từ nhiều nguồn
# - port: lấy cổng từ biến môi trường (cần thiết cho nền tảng cloud)
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
