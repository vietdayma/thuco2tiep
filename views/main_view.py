import streamlit as st
from utils.visualization import (
    plot_feature_importance,
    plot_emission_comparison,
    create_gauge_chart,
    style_metric_cards
)
import pandas as pd
import time
import numpy as np
from utils.benchmark_utils import BenchmarkUtils
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class MainView:
    def __init__(self, controller):
        self.controller = controller
        self.benchmark_utils = BenchmarkUtils()

    def show(self):
        """Display the main application interface"""
        # Add custom CSS
        st.markdown(style_metric_cards(), unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("# 🚗 CO2 Emission Predictor")
            st.markdown("---")
            page = st.radio("Navigation", ["Prediction", "Analysis", "Benchmark"])

        if page == "Prediction":
            self._show_prediction_page()
        elif page == "Analysis":
            self._show_analysis_page()
        else:
            self._show_benchmark_page()

    def _show_prediction_page(self):
        """Display the prediction interface"""
        st.title("🌍 Predict Vehicle CO2 Emissions")
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px'>
            <h4 style='margin: 0; color: #0f4c81'>Enter your vehicle specifications to predict CO2 emissions</h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            engine_size = st.number_input("🔧 Engine Size (L)", 
                                        min_value=0.1, 
                                        max_value=10.0, 
                                        value=2.0,
                                        step=0.1)
            
            cylinders = st.number_input("⚙️ Number of Cylinders",
                                      min_value=2,
                                      max_value=16,
                                      value=4,
                                      step=1)
            
            fuel_consumption = st.number_input("⛽ Fuel Consumption (L/100 km)",
                                             min_value=1.0,
                                             max_value=30.0,
                                             value=8.0,
                                             step=0.1)

        with col2:
            horsepower = st.number_input("🏎️ Horsepower",
                                       min_value=50,
                                       max_value=1000,
                                       value=200,
                                       step=10)
            
            weight = st.number_input("⚖️ Vehicle Weight (kg)",
                                   min_value=500,
                                   max_value=5000,
                                   value=1500,
                                   step=100)
            
            year = st.number_input("📅 Vehicle Year",
                                 min_value=2015,
                                 max_value=2024,
                                 value=2023,
                                 step=1)

        if st.button("🔍 Predict Emissions", type="primary"):
            features = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption Comb (L/100 km)': fuel_consumption,
                'Horsepower': horsepower,
                'Weight (kg)': weight,
                'Year': year
            }

            try:
                prediction = self.controller.predict_emission(features)
                avg_emission = self.controller.get_average_emission()
                rating = self.controller.get_emission_rating(prediction)
                tips = self.controller.get_eco_tips(prediction)

                # Display results
                st.markdown("### 📊 Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>🎯 Predicted CO2 Emission</h3>
                            <div class="metric-value">{prediction:.1f} g/km</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    rating_colors = {
                        'A': '🟢', 'B': '🟡', 'C': '🟠',
                        'D': '🔴', 'E': '🟣', 'F': '⚫'
                    }
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>📈 Emission Rating</h3>
                            <div class="metric-value">{rating_colors.get(rating, '⚪')} {rating}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col3:
                    comparison = ((prediction - avg_emission) / avg_emission * 100)
                    icon = "🔽" if comparison < 0 else "🔼"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>📊 Compared to Average</h3>
                            <div class="metric-value">
                                {icon} {'+' if comparison > 0 else ''}{comparison:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Visualization
                st.markdown("### 📈 Visualization")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(plot_emission_comparison(prediction, avg_emission))
                
                with col2:
                    st.pyplot(create_gauge_chart(prediction, 0, 300, "Emission Meter"))

                # Eco Tips
                st.markdown("### 🌱 Eco-friendly Tips")
                for tip in tips:
                    st.markdown(f"- {tip}")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    def _show_analysis_page(self):
        """Display the analysis interface"""
        st.title("📊 CO2 Emission Analysis")
        
        # Feature Importance
        st.subheader("🎯 Feature Importance Analysis")
        try:
            importance_dict = self.controller.get_feature_importance()
            st.pyplot(plot_feature_importance(importance_dict))
            
            # Add explanation
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 20px'>
                <h4 style='margin: 0; color: #0f4c81'>Understanding Feature Importance</h4>
                <p style='margin-top: 10px'>
                    This chart shows how much each vehicle characteristic influences CO2 emissions. 
                    Longer bars indicate stronger influence on the prediction.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error getting feature importance: {str(e)}")

        # Additional analysis sections can be added here 

    def _show_benchmark_page(self):
        st.title("⏱️ Benchmark 1000 Requests")
        
        # Lấy API URL từ environment hoặc dùng giá trị mặc định
        API_URL = os.environ.get('API_URL', 'https://thuco2tiep.onrender.com')
        st.info(f"Using API endpoint: {API_URL}")
        
        # Kiểm tra API health
        try:
            health_response = requests.get(f"{API_URL}/health")
            if health_response.status_code == 200:
                st.success("API is healthy and ready!")
            else:
                st.warning(f"API health check failed: {health_response.json().get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Could not connect to API: {str(e)}")
            return

        # Chọn chế độ test
        test_mode = st.radio(
            "Chế độ kiểm tra",
            ["Tham số cố định", "Tham số ngẫu nhiên"]
        )

        # Hiển thị form nhập tham số nếu chọn chế độ cố định
        if test_mode == "Tham số cố định":
            st.subheader("Nhập tham số kiểm tra:")
            col1, col2 = st.columns(2)
            
            with col1:
                engine_size = st.number_input("Engine Size (L)", 
                    min_value=0.1, max_value=10.0, value=2.0, step=0.1)
                cylinders = st.number_input("Cylinders", 
                    min_value=2, max_value=16, value=4, step=1)
                fuel_consumption = st.number_input("Fuel Consumption (L/100km)", 
                    min_value=1.0, max_value=30.0, value=8.0, step=0.1)
            
            with col2:
                horsepower = st.number_input("Horsepower", 
                    min_value=50, max_value=1000, value=200, step=10)
                weight = st.number_input("Weight (kg)", 
                    min_value=500, max_value=5000, value=1500, step=100)
                year = st.number_input("Year", 
                    min_value=2015, max_value=2024, value=2023, step=1)

            features = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption Comb (L/100 km)': fuel_consumption,
                'Horsepower': horsepower,
                'Weight (kg)': weight,
                'Year': year
            }
        else:
            features = self.generate_random_features()
            st.info("Mỗi request sẽ sử dụng một bộ tham số ngẫu nhiên khác nhau")
            st.write("Ví dụ tham số ngẫu nhiên:", features)
        
        if st.button("Chạy Benchmark"):
            # Container cho log
            log_container = st.empty()
            progress_bar = st.progress(0)
            
            # Bắt đầu đo thời gian
            start_time = time.perf_counter()
            
            # Thực hiện 1000 request
            n_requests = 1000
            successful_requests = 0
            completed_requests = 0

            def make_request():
                try:
                    request_features = (
                        self.generate_random_features() 
                        if test_mode == "Tham số ngẫu nhiên" 
                        else features
                    )
                    
                    # Gọi API với timeout
                    start_time = time.perf_counter()
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=request_features,
                        timeout=5  # 5 seconds timeout
                    )
                    total_time = (time.perf_counter() - start_time) * 1000  # ms
                    
                    if response.status_code == 200:
                        result = response.json()
                        if completed_requests == 0:
                            st.write("Debug - First request:", {
                                'features': request_features,
                                'prediction': result['prediction'],
                                'api_process_time': result['process_time_ms'],
                                'total_time': total_time,
                                'network_latency': total_time - result['process_time_ms']
                            })
                        return True
                    else:
                        if completed_requests == 0:
                            st.error(f"API Error: {response.text}")
                        return False
                        
                except Exception as e:
                    if completed_requests == 0:
                        st.error(f"Request Error: {str(e)}")
                    return False

            # Sử dụng ThreadPoolExecutor với 50 luồng cố định
            with ThreadPoolExecutor(max_workers=50) as executor:
                # Submit tất cả requests
                future_to_request = {
                    executor.submit(make_request): i 
                    for i in range(n_requests)
                }
                
                # Xử lý kết quả khi hoàn thành
                for future in as_completed(future_to_request):
                    completed_requests += 1
                    if future.result():
                        successful_requests += 1
                    
                    # Cập nhật progress
                    progress = completed_requests / n_requests
                    progress_bar.progress(progress)
                    
                    # Cập nhật log mỗi 100 request
                    if completed_requests % 100 == 0:
                        current_time = time.perf_counter() - start_time
                        log_container.text(
                            f"Đã xử lý {completed_requests}/{n_requests} requests... "
                            f"({current_time:.1f}s)"
                        )
            
            # Kết thúc đo thời gian
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Hiển thị kết quả
            st.success("Benchmark hoàn thành!")
            st.markdown(f"""
            ### Kết quả:
            - Chế độ kiểm tra: {test_mode}
            - Tổng thời gian: {total_time:.2f} giây
            - Số request thành công: {successful_requests}/{n_requests}
            - Tốc độ trung bình: {n_requests/total_time:.1f} requests/giây
            """)

    def generate_random_features(self):
        """Tạo bộ tham số ngẫu nhiên"""
        return {
            'Engine Size(L)': np.random.uniform(1.0, 8.0),
            'Cylinders': np.random.randint(3, 12),
            'Fuel Consumption Comb (L/100 km)': np.random.uniform(4.0, 20.0),
            'Horsepower': np.random.uniform(100, 800),
            'Weight (kg)': np.random.uniform(1000, 4000),
            'Year': np.random.randint(2015, 2024)
        } 