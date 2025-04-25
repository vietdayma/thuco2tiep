import streamlit as st
from controllers.emission_controller import EmissionController
from views.main_view import MainView
import subprocess
import sys
import os

def start_api_server():
    """Start API server in a separate process"""
    try:
        # Khởi động API server
        subprocess.Popen([sys.executable, "api_server.py"])
        st.success("API Server started successfully!")
    except Exception as e:
        st.error(f"Error starting API server: {str(e)}")

def main():
    # Khởi động API server
    start_api_server()
    
    # Initialize controller
    controller = EmissionController()
    
    # Train the model
    try:
        test_score = controller.initialize_model('co2 Emissions.csv')
        print(f"Model trained successfully. Test score: {test_score:.3f}")
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return

    # Initialize and show view
    view = MainView(controller)
    view.show()

if __name__ == "__main__":
    main() 