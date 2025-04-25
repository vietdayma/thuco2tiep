import streamlit as st
import os
import sys
import requests
import time

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

def check_api_health():
    """Check if API is available and ready"""
    api_url = os.environ.get('API_URL')
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            st.success(f"Connected to API server at {api_url}")
            return True
        else:
            st.error(f"API server at {api_url} returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API server at {api_url}: {str(e)}")
        return False

def main():
    st.title("CO2 Emission Prediction")
    
    # Kiểm tra kết nối đến API server
    if not check_api_health():
        st.error("Cannot connect to API server. Please try again later.")
        st.warning("If you're running this app locally, make sure the API server is running at https://thuco2tiep.onrender.com")
        return
        
    # Kiểm tra file CSV tồn tại
    csv_path = os.path.join(current_dir, "co2 Emissions.csv")
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