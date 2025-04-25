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

# Tạo semaphore để giới hạn số lượng request đồng thời
api_semaphore = threading.Semaphore(5)

# Cache cho các kết quả API
prediction_cache = {}
cache_lock = threading.Lock()
MAX_CACHE_SIZE = 50

def get_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=0.5,  # Backoff factor for retries
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
    cache_key = get_cache_key(features)
    
    # Check cache first
    with cache_lock:
        if cache_key in prediction_cache:
            return prediction_cache[cache_key]
    
    # Use semaphore to limit concurrent API calls
    with api_semaphore:
        try:
            # Add a small random delay to avoid bursts of requests
            time.sleep(random.uniform(0.05, 0.2))
            
            # Make API request
            session = get_session()
            api_url = os.environ.get('API_URL') + "/predict"
            response = session.post(api_url, json=features, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Store in cache
            with cache_lock:
                if len(prediction_cache) < MAX_CACHE_SIZE:
                    prediction_cache[cache_key] = result
            
            return result
        except Exception as e:
            st.error(f"Lỗi dự đoán từ API: {str(e)}")
            raise

def check_api_health():
    """Check if API is available and ready"""
    api_url = os.environ.get('API_URL')
    
    st.markdown("### Kiểm tra kết nối API")
    status_placeholder = st.empty()
    status_placeholder.info("Đang kết nối đến API server...")
    
    try:
        # Use session with retry logic
        session = get_session()
        response = session.get(f"{api_url}/health", timeout=30)
        
        if response.status_code == 200:
            status_placeholder.success(f"Đã kết nối đến API server tại {api_url}")
            return True
        elif response.status_code == 503:
            # API is starting up
            status = response.json().get("status", "")
            message = response.json().get("message", "")
            
            if status == "initializing":
                for i in range(60):  # Wait up to 60 seconds
                    status_placeholder.warning(f"API server đang khởi tạo mô hình... Vui lòng đợi ({i+1}/60s)")
                    time.sleep(1)
                    
                    try:
                        response = session.get(f"{api_url}/health", timeout=10)
                        if response.status_code == 200:
                            status_placeholder.success(f"Đã kết nối đến API server tại {api_url}")
                            return True
                    except requests.exceptions.RequestException:
                        pass
                
                status_placeholder.error(f"API server khởi động quá lâu. Vui lòng thử lại sau.")
                return False
            else:
                status_placeholder.error(f"API server trả về trạng thái: {message}")
                return False
        else:
            status_placeholder.error(f"API server tại {api_url} trả về mã trạng thái {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        status_placeholder.error(f"Không thể kết nối đến API server tại {api_url}: {str(e)}")
        return False

def main():
    st.title("CO2 Emission Prediction")
    
    # Kiểm tra kết nối đến API server
    if not check_api_health():
        st.error("Không thể kết nối đến API server. Vui lòng thử lại sau.")
        st.warning("Nếu bạn đang chạy ứng dụng cục bộ, hãy đảm bảo API server đang chạy tại https://thuco2tiep.onrender.com")
        return
        
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