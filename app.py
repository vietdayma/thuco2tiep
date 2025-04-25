import streamlit as st
import os
import sys
import requests
import time
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

def get_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=0.5,  # Backoff factor for retries
        status_forcelist=[500, 502, 503, 504],  # Status codes to retry on
        allowed_methods=["GET", "POST"]  # Methods to retry
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

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

    # Initialize controller
    controller = EmissionController()
    
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