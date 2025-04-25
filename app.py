import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="CO2 Emission Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from controllers.emission_controller import EmissionController
from views.main_view import MainView
import subprocess
import sys
import time
import os
import requests

def start_api_server():
    """Start API server and wait until it's ready"""
    try:
        # Get absolute path to api_server.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_server_path = os.path.join(current_dir, "api_server.py")
        
        if not os.path.exists(api_server_path):
            st.error(f"Could not find API server at {api_server_path}")
            return None
            
        # Khởi động API server với đường dẫn tuyệt đối
        api_process = subprocess.Popen(
            [sys.executable, api_server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Chờ API server khởi động (tối đa 30 giây)
        max_retries = 30
        for i in range(max_retries):
            try:
                # Thử kết nối đến API server
                response = requests.get("http://localhost:5000/health")
                if response.status_code == 200:
                    st.success("API Server started successfully!")
                    return api_process
            except requests.exceptions.ConnectionError:
                time.sleep(1)
                # Kiểm tra xem process có còn chạy không
                if api_process.poll() is not None:
                    stdout, stderr = api_process.communicate()
                    st.error(f"API server crashed. Error: {stderr.decode()}")
                    return None
                continue
            except Exception as e:
                st.error(f"Error connecting to API server: {str(e)}")
                return None
        
        st.error("Timeout waiting for API server to start")
        return None
    except Exception as e:
        st.error(f"Error starting API server: {str(e)}")
        return None

def main():
    st.title("CO2 Emission Prediction")
    
    # Khởi động API server và đợi cho đến khi nó sẵn sàng
    api_process = start_api_server()
    if api_process is None:
        st.error("Failed to start API server. Please check the logs.")
        return
        
    # Kiểm tra file CSV tồn tại
    csv_path = "co2 Emissions.csv"
    if not os.path.exists(csv_path):
        st.error(f"Error: Could not find the file '{csv_path}'. Please make sure it exists in the project root directory.")
        return

    # Initialize controller
    controller = EmissionController()
    
    # Train the model
    try:
        test_score = controller.initialize_model(csv_path)
        st.success(f"Model trained successfully. Test score: {test_score:.3f}")
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return

    # Initialize and show view
    view = MainView(controller)
    view.show()

if __name__ == "__main__":
    main() 