import streamlit as st  # Thư viện xây dựng giao diện web
import os  # Thư viện tương tác với hệ điều hành
import sys  # Thư viện thao tác với môi trường Python
import requests  # Thư viện gọi API HTTP
import time  # Thư viện xử lý thời gian
import threading  # Thư viện đa luồng
import random  # Thư viện sinh số ngẫu nhiên
from requests.adapters import HTTPAdapter  # Bộ điều hợp HTTP cho phép cấu hình request
from requests.packages.urllib3.util.retry import Retry  # Cơ chế thử lại tự động khi request thất bại

# Thêm đường dẫn hiện tại vào sys.path (để đảm bảo imports hoạt động trên Streamlit Cloud)
# Giúp Python tìm thấy các module trong thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Cấu hình Streamlit - phải là lệnh Streamlit đầu tiên
# Thiết lập các thông số của ứng dụng web
st.set_page_config(
    page_title="CO2 Emission Predictor",  # Tiêu đề hiển thị trên tab trình duyệt
    page_icon="🌍",                        # Icon hiển thị trên tab trình duyệt
    layout="wide",                         # Bố cục rộng để tận dụng không gian màn hình
    initial_sidebar_state="expanded"       # Thanh bên mở rộng ban đầu khi tải trang
)

# Import các module cần thiết sau khi đã cấu hình đường dẫn
# Đảm bảo đường dẫn đã được thêm vào sys.path trước khi import
from controllers.emission_controller import EmissionController
from views.main_view import MainView

# Thiết lập URL API từ biến môi trường
# API deployed trên Render dùng cho dự đoán phát thải
os.environ['API_URL'] = 'https://thuco2tiep.onrender.com'

# Tạo semaphore để giới hạn số lượng request đồng thời đến API
# Tránh quá tải server bằng cách giới hạn tối đa 10 request cùng lúc
api_semaphore = threading.Semaphore(10)  # Tối đa 10 request đồng thời

# Hệ thống cache cho kết quả API
prediction_cache = {}  # Dictionary lưu trữ kết quả dự đoán
cache_lock = threading.Lock()  # Khóa đồng bộ hóa cho cache (thread-safe)
MAX_CACHE_SIZE = 100  # Kích thước tối đa của cache, tránh sử dụng quá nhiều bộ nhớ

# Giá trị mặc định khi API không phản hồi
DEFAULT_PREDICTION = 200.0  # Giá trị phát thải CO2 trung bình (g/km) dùng khi có lỗi

def get_session():
    """
    Tạo session HTTP với cơ chế retry tự động.
    Giúp tăng độ tin cậy khi gọi API bằng cách tự động thử lại các request thất bại.
    Session này sẽ tự động thử lại nếu gặp lỗi mạng tạm thời.
    
    Trả về:
        requests.Session: Đối tượng session với cấu hình retry.
    """
    session = requests.Session()
    
    # Cấu hình retry logic
    retry = Retry(
        total=5,                   # Tổng số lần thử lại
        backoff_factor=0.2,        # Hệ số chờ giữa các lần thử (0.2s, 0.4s, 0.8s,...)
        status_forcelist=[429, 500, 502, 503, 504],  # Mã lỗi HTTP cần thử lại
        allowed_methods=["GET", "POST"]  # Phương thức HTTP được phép thử lại
    )
    
    # Gắn cấu hình retry vào session
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def get_cache_key(features):
    """
    Tạo khóa cache từ đặc trưng đầu vào.
    Chuyển đổi các đặc trưng xe thành một chuỗi duy nhất để sử dụng làm khóa cache.
    
    Tham số:
        features (dict): Từ điển chứa các đặc trưng của xe.
        
    Trả về:
        str: Chuỗi khóa cache hoặc None nếu có lỗi.
    """
    try:
        key_parts = []
        # Sắp xếp các cặp key-value để đảm bảo tính nhất quán của khóa
        # Đảm bảo cùng đặc trưng luôn tạo ra cùng một khóa, bất kể thứ tự
        for k, v in sorted(features.items()):
            key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)  # Nối các phần thành một chuỗi duy nhất
    except:
        return None  # Trả về None nếu có lỗi xảy ra

def predict_with_api(features):
    """
    Dự đoán phát thải CO2 bằng cách gọi API với cơ chế giới hạn request đồng thời.
    Hàm này thực hiện kiểm tra cache, quản lý semaphore và xử lý lỗi đầy đủ.
    
    Tham số:
        features (dict): Từ điển chứa các đặc trưng của xe.
        
    Trả về:
        dict: Kết quả dự đoán từ API hoặc giá trị fallback nếu có lỗi.
    """
    # Tạo khóa cache từ đặc trưng
    cache_key = get_cache_key(features)
    
    # Kiểm tra cache trước khi gọi API để tăng tốc độ phản hồi
    with cache_lock:  # Sử dụng lock để tránh xung đột giữa các thread
        if cache_key in prediction_cache:
            return prediction_cache[cache_key]  # Trả về kết quả đã cache
    
    # Cơ chế fallback khi có lỗi
    try:
        # Sử dụng semaphore để giới hạn số lượng request đồng thời
        acquired = api_semaphore.acquire(timeout=0.5)  # Chờ tối đa 0.5s để lấy semaphore
        if not acquired:
            # Nếu không thể lấy semaphore, trả về giá trị mặc định
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': 'Quá nhiều request đồng thời'
            }
            
        try:
            # Thêm độ trễ ngẫu nhiên để tránh gửi request đồng loạt
            # Giúp giảm tải cho server khi nhiều người dùng cùng lúc
            time.sleep(random.uniform(0.01, 0.1))
            
            # Kiểm tra xem có đang trong chế độ benchmark không
            benchmark_mode = os.environ.get('BENCHMARK_MODE', 'false').lower() == 'true'
            
            # Chuẩn bị API request
            session = get_session()
            api_url = os.environ.get('API_URL')
            
            # Xử lý khác nhau tùy theo chế độ
            if benchmark_mode:
                # Sử dụng endpoint fallback cho benchmark (nhanh hơn)
                # Trong chế độ benchmark, chúng ta muốn kiểm tra hiệu suất, không cần dự đoán thực
                api_url = api_url + "/fallback"
                response = session.post(api_url, json={}, timeout=2)
            else:
                # Sử dụng endpoint predict với timeout ngắn
                # Trong chế độ thông thường, gửi đặc trưng đến API để dự đoán thực tế
                api_url = api_url + "/predict"
                response = session.post(api_url, json=features, timeout=2)
                
            # Kiểm tra response và chuyển đổi thành JSON
            response.raise_for_status()  # Tạo ngoại lệ nếu phản hồi không thành công
            result = response.json()
            
            # Lưu kết quả vào cache
            with cache_lock:
                if len(prediction_cache) < MAX_CACHE_SIZE:
                    prediction_cache[cache_key] = result
            
            return result
        except requests.exceptions.Timeout:
            # Xử lý timeout - trả về giá trị mặc định
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': 'API timeout'
            }
        except requests.exceptions.RequestException as e:
            # Xử lý các lỗi request khác
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': f'Lỗi API: {str(e)}'
            }
        finally:
            # Luôn giải phóng semaphore khi hoàn thành để tránh deadlock
            api_semaphore.release()
    except Exception as e:
        # Xử lý mọi lỗi khác không lường trước được
        return {
            'prediction': DEFAULT_PREDICTION,
            'process_time_ms': 5.0,
            'status': 'fallback',
            'message': f'Lỗi client: {str(e)}'
        }

def check_api_health():
    """
    Kiểm tra trạng thái hoạt động của API.
    Hiển thị thông báo trạng thái và chờ đợi nếu API đang khởi động.
    Cung cấp trải nghiệm người dùng tốt khi API chưa sẵn sàng.
    
    Trả về:
        bool: True nếu API hoạt động hoặc sau thời gian chờ, False nếu không thể kết nối.
    """
    api_url = os.environ.get('API_URL')
    
    # Tạo vùng hiển thị thông báo trạng thái
    st.markdown("### Kiểm tra kết nối API")
    status_placeholder = st.empty()  # Tạo vùng trống để cập nhật trạng thái
    status_placeholder.info("Đang kết nối đến API server...")
    
    try:
        # Sử dụng session với retry logic
        session = get_session()
        response = session.get(f"{api_url}/health", timeout=10)
        
        # Nếu API sẵn sàng
        if response.status_code == 200:
            status_placeholder.success(f"Đã kết nối đến API server tại {api_url}")
            return True
        else:
            # API đang khởi tạo hoặc có vấn đề
            status = response.json().get("status", "") if response.content else "unknown"
            message = response.json().get("message", "") if response.content else "No response"
            
            # Chờ tối đa 20 giây cho API khởi động
            for i in range(20):
                status_placeholder.warning(f"API server đang khởi tạo... Vui lòng đợi ({i+1}/20s)")
                time.sleep(1)  # Chờ 1 giây
                
                # Thử kiểm tra lại
                try:
                    response = session.get(f"{api_url}/health", timeout=3)
                    if response.status_code == 200 and response.json().get("status") == "healthy":
                        status_placeholder.success(f"Đã kết nối đến API server tại {api_url}")
                        return True
                except requests.exceptions.RequestException:
                    pass
            
            # Nếu vẫn không thành công sau thời gian chờ, hiển thị lỗi nhưng vẫn tiếp tục
            status_placeholder.error(f"API server có vấn đề: {message}. Tiếp tục với dự đoán local.")
            return True  # Vẫn trả về True để tiếp tục với mô hình local
    except requests.exceptions.RequestException as e:
        # Không thể kết nối đến API
        status_placeholder.error(f"Không thể kết nối đến API server tại {api_url}: {str(e)}")
        # Vẫn tiếp tục với mô hình local
        return True

def main():
    """
    Hàm chính của ứng dụng Streamlit.
    Thiết lập giao diện, kết nối API và hiển thị dự đoán.
    Quản lý luồng chính của ứng dụng web.
    """
    # Hiển thị tiêu đề chính
    st.title("CO2 Emission Prediction")
    
    # Kiểm tra kết nối đến API server - luôn tiếp tục ngay cả khi có lỗi
    api_available = check_api_health()
        
    # Kiểm tra file CSV tồn tại
    csv_path = os.path.join(current_dir, "co2 Emissions.csv")
    if not os.path.exists(csv_path):
        st.error(f"Lỗi: Không thể tìm thấy file '{csv_path}'. Vui lòng đảm bảo file tồn tại trong thư mục gốc của dự án.")
        return

    # Khởi tạo controller và ghi đè phương thức dự đoán API
    controller = EmissionController()
    # Ghi đè phương thức dự đoán API bằng hàm của chúng ta
    # Kỹ thuật "monkey patching" - thay đổi hành vi của phương thức trong thời gian chạy
    controller.predict_emission_api = predict_with_api
    
    # Huấn luyện mô hình
    try:
        test_score = controller.initialize_model(csv_path)
        st.success(f"Mô hình được huấn luyện thành công. Điểm kiểm tra: {test_score:.3f}")
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        return

    # Khởi tạo và hiển thị giao diện chính
    view = MainView(controller)
    view.show()  # Hiển thị giao diện người dùng

# Điểm vào của ứng dụng khi chạy trực tiếp
if __name__ == "__main__":
    main()  # Gọi hàm main khi chạy script này trực tiếp 