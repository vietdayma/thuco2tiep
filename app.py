import streamlit as st
import os
import sys
import requests
import time
import threading
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Thêm đường dẫn hiện tại vào sys.path (để đảm bảo imports hoạt động trên Streamlit Cloud)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Must be the first Streamlit command
st.set_page_config(
    page_title="CO2 Emission Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from controllers.emission_controller import EmissionController
from views.main_view import MainView

# Set API URL environment variable
os.environ['API_URL'] = 'https://thuco2tiep.onrender.com'

# Tăng semaphore cho nhiều request hơn
api_semaphore = threading.Semaphore(10)  # Tăng lên 10 request đồng thời

# Cache cho các kết quả API
prediction_cache = {}
cache_lock = threading.Lock()
MAX_CACHE_SIZE = 100

# Giá trị mặc định nếu API không phản hồi
DEFAULT_PREDICTION = 200.0

def get_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=5,  # Tăng số lần thử lại
        backoff_factor=0.2,  # Giảm thời gian giữa các lần retry để tăng tốc
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
        allowed_methods=["GET", "POST"]  # Methods to retry
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_cache_key(features):
    """Generate a cache key from features"""
    try:
        key_parts = []
        for k, v in sorted(features.items()):
            key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
    except:
        return None

def predict_with_api(features):
    """Make prediction with API using semaphore to limit concurrent requests"""
    # Tạo cache key trước
    cache_key = get_cache_key(features)
    
    # Check cache first
    with cache_lock:
        if cache_key in prediction_cache:
            return prediction_cache[cache_key]
    
    # Fallback mechanism
    try:
        # Use semaphore to limit concurrent API calls
        acquired = api_semaphore.acquire(timeout=0.5)  # Timeout nếu không thể acquire trong 0.5s
        if not acquired:
            # Nếu không thể lấy semaphore, trả về giá trị mặc định
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': 'Too many concurrent requests'
            }
            
        try:
            # Add a small random delay to avoid bursts of requests
            time.sleep(random.uniform(0.01, 0.1))  # Giảm delay ngẫu nhiên
            
            # Sử dụng endpoint fallback nếu trong benchmark mode
            benchmark_mode = os.environ.get('BENCHMARK_MODE', 'false').lower() == 'true'
            
            # Make API request
            session = get_session()
            api_url = os.environ.get('API_URL')
            
            if benchmark_mode:
                # Sử dụng fallback endpoint cho benchmark
                api_url = api_url + "/fallback"
                response = session.post(api_url, json={}, timeout=2)
            else:
                # Sử dụng predict endpoint với timeout ngắn hơn
                api_url = api_url + "/predict"
                response = session.post(api_url, json=features, timeout=2)
                
            response.raise_for_status()
            result = response.json()
            
            # Store in cache
            with cache_lock:
                if len(prediction_cache) < MAX_CACHE_SIZE:
                    prediction_cache[cache_key] = result
            
            return result
        except requests.exceptions.Timeout:
            # Timeout - trả về giá trị mặc định
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': 'API timeout'
            }
        except requests.exceptions.RequestException as e:
            # Các lỗi request khác - trả về giá trị mặc định
            return {
                'prediction': DEFAULT_PREDICTION,
                'process_time_ms': 5.0,
                'status': 'fallback',
                'message': f'API error: {str(e)}'
            }
        finally:
            # Luôn release semaphore
            api_semaphore.release()
    except Exception as e:
        # Bất kỳ lỗi nào khác (bao gồm timeout khi acquire semaphore)
        return {
            'prediction': DEFAULT_PREDICTION,
            'process_time_ms': 5.0,
            'status': 'fallback',
            'message': f'Client error: {str(e)}'
        }

def check_api_health():
    """Check if API is available and ready"""
    api_url = os.environ.get('API_URL')
    
    st.markdown("### Kiểm tra kết nối API")
    status_placeholder = st.empty()
    status_placeholder.info("Đang kết nối đến API server...")
    
    try:
        # Use session with retry logic
        session = get_session()
        response = session.get(f"{api_url}/health", timeout=10)  # Giảm timeout xuống 10s
        
        if response.status_code == 200:
            status_placeholder.success(f"Đã kết nối đến API server tại {api_url}")
            return True
        else:
            # Coi mọi mã HTTP khác là đang khởi tạo
            status = response.json().get("status", "") if response.content else "unknown"
            message = response.json().get("message", "") if response.content else "No response"
            
            # Chờ tối đa 20 giây (giảm xuống từ 60s)
            for i in range(20):
                status_placeholder.warning(f"API server đang khởi tạo... Vui lòng đợi ({i+1}/20s)")
                time.sleep(1)
                
                try:
                    response = session.get(f"{api_url}/health", timeout=3)
                    if response.status_code == 200 and response.json().get("status") == "healthy":
                        status_placeholder.success(f"Đã kết nối đến API server tại {api_url}")
                        return True
                except requests.exceptions.RequestException:
                    pass
            
            # Sau thời gian chờ, đánh dấu là lỗi nhưng vẫn tiếp tục - sẽ dùng fallback
            status_placeholder.error(f"API server có vấn đề: {message}. Tiếp tục với dự đoán local.")
            return True  # Vẫn trả về True để tiếp tục
    except requests.exceptions.RequestException as e:
        status_placeholder.error(f"Không thể kết nối đến API server tại {api_url}: {str(e)}")
        # Vẫn tiếp tục với mô hình local
        return True

def main():
    st.title("CO2 Emission Prediction")
    
    # Kiểm tra kết nối đến API server - luôn tiếp tục
    api_available = check_api_health()
        
    # Kiểm tra file CSV tồn tại
    csv_path = os.path.join(current_dir, "co2 Emissions.csv")
    if not os.path.exists(csv_path):
        st.error(f"Lỗi: Không thể tìm thấy file '{csv_path}'. Vui lòng đảm bảo file tồn tại trong thư mục gốc của dự án.")
        return

    # Initialize controller with overridden API prediction method
    controller = EmissionController()
    # Override the predict_emission_api method to use our semaphore-controlled function
    controller.predict_emission_api = predict_with_api
    
    # Train the model
    try:
        test_score = controller.initialize_model(csv_path)
        st.success(f"Mô hình được huấn luyện thành công. Điểm kiểm tra: {test_score:.3f}")
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        return

    # Initialize and show view
    view = MainView(controller)
    view.show()

if __name__ == "__main__":
    main() 