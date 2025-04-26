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
    """
    MainView là lớp chính quản lý giao diện người dùng của ứng dụng Streamlit
    Lớp này chịu trách nhiệm hiển thị các trang và tương tác với người dùng
    Kết nối với controller để thực hiện các dự đoán và phân tích dữ liệu
    """
    def __init__(self, controller):
        """
        Khởi tạo đối tượng MainView
        
        Parameters:
            controller: EmissionController - Đối tượng controller để xử lý logic nghiệp vụ và dự đoán
        """
        self.controller = controller
        self.benchmark_utils = BenchmarkUtils()

    def show(self):
        """
        Hiển thị giao diện chính của ứng dụng với thanh điều hướng bên và các trang tương ứng
        Người dùng có thể chuyển đổi giữa các trang: Dự đoán, Phân tích và Benchmark
        """
        # Thêm CSS tùy chỉnh để làm đẹp giao diện
        st.markdown(style_metric_cards(), unsafe_allow_html=True)
        
        # Thiết lập thanh điều hướng bên trái
        with st.sidebar:
            st.markdown("# 🚗 CO2 Emission Predictor")
            st.markdown("---")
            page = st.radio("Navigation", ["Prediction", "Analysis", "Benchmark"])

        # Hiển thị trang tương ứng theo lựa chọn người dùng
        if page == "Prediction":
            self._show_prediction_page()
        elif page == "Analysis":
            self._show_analysis_page()
        else:
            self._show_benchmark_page()

    def _show_prediction_page(self):
        """
        Hiển thị trang dự đoán phát thải CO2
        Cho phép người dùng nhập các đặc điểm của phương tiện và nhận dự đoán phát thải CO2
        Hiển thị kết quả dưới dạng số và biểu đồ trực quan
        """
        st.title("🌍 Predict Vehicle CO2 Emissions")
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px'>
            <h4 style='margin: 0; color: #0f4c81'>Enter your vehicle specifications to predict CO2 emissions</h4>
        </div>
        """, unsafe_allow_html=True)

        # Chia layout thành 2 cột để nhập thông tin
        col1, col2 = st.columns(2)

        # Cột bên trái cho các thông số đầu tiên
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

        # Cột bên phải cho các thông số còn lại
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

        # Nút dự đoán để kích hoạt quá trình dự đoán
        if st.button("🔍 Predict Emissions", type="primary"):
            # Tạo dictionary thông số xe để truyền vào controller
            features = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption Comb (L/100 km)': fuel_consumption,
                'Horsepower': horsepower,
                'Weight (kg)': weight,
                'Year': year
            }

            try:
                # Thực hiện dự đoán và lấy các thông tin liên quan
                prediction = self.controller.predict_emission(features)
                avg_emission = self.controller.get_average_emission()
                rating = self.controller.get_emission_rating(prediction)
                tips = self.controller.get_eco_tips(prediction)

                # Hiển thị kết quả
                st.markdown("### 📊 Results")
                col1, col2, col3 = st.columns(3)
                
                # Cột 1: Kết quả dự đoán CO2
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

                # Cột 2: Xếp hạng phát thải
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

                # Cột 3: So sánh với mức trung bình
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

                # Hiển thị biểu đồ trực quan
                st.markdown("### 📈 Visualization")
                col1, col2 = st.columns(2)
                
                # Biểu đồ so sánh phát thải
                with col1:
                    st.pyplot(plot_emission_comparison(prediction, avg_emission))
                
                # Biểu đồ đồng hồ đo
                with col2:
                    st.pyplot(create_gauge_chart(prediction, 0, 300, "Emission Meter"))

                # Hiển thị mẹo thân thiện môi trường
                st.markdown("### 🌱 Eco-friendly Tips")
                for tip in tips:
                    st.markdown(f"- {tip}")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    def _show_analysis_page(self):
        """
        Hiển thị trang phân tích các tính năng quan trọng ảnh hưởng đến phát thải CO2
        Hiển thị biểu đồ độ quan trọng của các đặc trưng trong mô hình dự đoán
        """
        st.title("📊 CO2 Emission Analysis")
        
        # Phân tích độ quan trọng của từng đặc trưng
        st.subheader("🎯 Feature Importance Analysis")
        try:
            # Lấy thông tin độ quan trọng của các đặc trưng từ controller
            importance_dict = self.controller.get_feature_importance()
            st.pyplot(plot_feature_importance(importance_dict))
            
            # Thêm giải thích về biểu đồ độ quan trọng
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

        # Phần này có thể mở rộng để thêm các phân tích khác

    def _show_benchmark_page(self):
        """
        Hiển thị trang benchmark để kiểm tra hiệu suất của API
        Cho phép người dùng thực hiện 1000 request đến API để đánh giá thời gian đáp ứng
        Hỗ trợ hai chế độ: tham số cố định hoặc tham số ngẫu nhiên
        """
        st.title("⏱️ Benchmark 1000 Requests")
        
        # Lấy URL API từ biến môi trường hoặc sử dụng giá trị mặc định
        API_URL = os.environ.get('API_URL', 'https://thuco2tiep.onrender.com')
        st.info(f"Using API endpoint: {API_URL}")
        
        # Kiểm tra trạng thái khả dụng của API
        try:
            health_response = requests.get(f"{API_URL}/health")
            if health_response.status_code == 200:
                st.success("API is healthy and ready!")
            else:
                st.warning(f"API health check failed: {health_response.json().get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Could not connect to API: {str(e)}")
            return

        # Lựa chọn chế độ kiểm tra: Tham số cố định hoặc tham số ngẫu nhiên
        test_mode = st.radio(
            "Chế độ kiểm tra",
            ["Tham số cố định", "Tham số ngẫu nhiên"]
        )

        # Hiển thị form nhập thông số cho chế độ tham số cố định
        if test_mode == "Tham số cố định":
            st.subheader("Nhập tham số kiểm tra:")
            col1, col2 = st.columns(2)
            
            # Cột bên trái cho các thông số đầu tiên
            with col1:
                engine_size = st.number_input("Engine Size (L)", 
                    min_value=0.1, max_value=10.0, value=2.0, step=0.1)
                cylinders = st.number_input("Cylinders", 
                    min_value=2, max_value=16, value=4, step=1)
                fuel_consumption = st.number_input("Fuel Consumption (L/100km)", 
                    min_value=1.0, max_value=30.0, value=8.0, step=0.1)
            
            # Cột bên phải cho các thông số còn lại
            with col2:
                horsepower = st.number_input("Horsepower", 
                    min_value=50, max_value=1000, value=200, step=10)
                weight = st.number_input("Weight (kg)", 
                    min_value=500, max_value=5000, value=1500, step=100)
                year = st.number_input("Year", 
                    min_value=2015, max_value=2024, value=2023, step=1)

            # Tạo dictionary các thông số xe
            features = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption Comb (L/100 km)': fuel_consumption,
                'Horsepower': horsepower,
                'Weight (kg)': weight,
                'Year': year
            }
        else:
            # Nếu chọn chế độ tham số ngẫu nhiên, tạo mẫu tham số ngẫu nhiên
            features = self.generate_random_features()
            st.info("Mỗi request sẽ sử dụng một bộ tham số ngẫu nhiên khác nhau")
            st.write("Ví dụ tham số ngẫu nhiên:", features)
        
        # Nút kích hoạt quá trình benchmark
        if st.button("Chạy Benchmark"):
            # Tạo container cho log và thanh tiến trình
            log_container = st.empty()
            progress_bar = st.progress(0)
            
            # Bắt đầu đo thời gian
            start_time = time.perf_counter()
            
            # Thiết lập thông số cho benchmark thực hiện 1000 request
            n_requests = 1000
            successful_requests = 0
            completed_requests = 0

            # Hàm thực hiện một request đến API
            def make_request():
                try:
                    # Tạo tham số: cố định hoặc ngẫu nhiên tùy chế độ đã chọn
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
                        
                        # Thêm dòng này để ghi lại kết quả benchmark
                        self.benchmark_utils.add_result(
                            total_time=total_time,
                            network_time=total_time - result.get('process_time_ms', 0),
                            processing_time=result.get('process_time_ms', 0),
                            prediction=result.get('prediction', 0),
                            status=result.get('status', 'unknown')
                        )
                        
                        return True
                    else:
                        if completed_requests == 0:
                            st.error(f"API Error: {response.text}")
                        return False
                        
                except Exception as e:
                    if completed_requests == 0:
                        st.error(f"Request Error: {str(e)}")
                    return False

            # Sử dụng ThreadPoolExecutor để gửi nhiều request đồng thời
            with ThreadPoolExecutor(max_workers=50) as executor:
                # Gửi tất cả request
                future_to_request = {
                    executor.submit(make_request): i 
                    for i in range(n_requests)
                }
                
                # Xử lý kết quả khi các request hoàn thành
                for future in as_completed(future_to_request):
                    completed_requests += 1
                    if future.result():
                        successful_requests += 1
                    
                    # Cập nhật thanh tiến trình
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
            
            # Hiển thị kết quả benchmark
            st.success("Benchmark hoàn thành!")
            st.markdown(f"""
            ### Kết quả:
            - Chế độ kiểm tra: {test_mode}
            - Tổng thời gian: {total_time:.2f} giây
            - Số request thành công: {successful_requests}/{n_requests}
            - Tốc độ trung bình: {n_requests/total_time:.1f} requests/giây
            """)
            
            # Hiển thị bảng kết quả chi tiết
            st.subheader("Kết quả benchmark chi tiết")

            # Lấy DataFrame kết quả từ benchmark_utils
            results_df = self.benchmark_utils.get_results_df()

            # Định dạng lại các cột để hiển thị đẹp hơn
            if not results_df.empty:
                # Format các cột thời gian để hiển thị 2 chữ số thập phân
                for time_col in ['total_time', 'network_time', 'processing_time']:
                    results_df[time_col] = results_df[time_col].round(2).astype(str) + " ms"
                
                # Format các cột phần trăm
                for pct_col in ['network_percentage', 'processing_percentage']:
                    results_df[pct_col] = results_df[pct_col].astype(str) + " %"
                
                # Format giá trị dự đoán
                results_df['prediction'] = results_df['prediction'].round(2).astype(str) + " g/km"
                
                # Hiển thị bảng với định dạng màu sắc
                st.dataframe(
                    results_df,
                    column_config={
                        "request_number": st.column_config.NumberColumn("STT", help="Số thứ tự request"),
                        "timestamp": st.column_config.DatetimeColumn("Thời điểm", format="HH:mm:ss.SSS"),
                        "total_time": st.column_config.TextColumn("Tổng thời gian"),
                        "network_time": st.column_config.TextColumn("Thời gian mạng"),
                        "processing_time": st.column_config.TextColumn("Thời gian xử lý"),
                        "network_percentage": st.column_config.TextColumn("% Mạng"),
                        "processing_percentage": st.column_config.TextColumn("% Xử lý"),
                        "prediction": st.column_config.TextColumn("Kết quả dự đoán"),
                        "status": st.column_config.TextColumn("Trạng thái"),
                        "error": st.column_config.TextColumn("Lỗi (nếu có)")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Hiển thị tổng kết về nguồn kết quả
                fallback_count = (results_df['status'] == 'fallback').sum()
                cached_count = results_df['status'].str.contains('cached', case=False).sum() if 'status' in results_df.columns else 0
                api_count = len(results_df) - fallback_count - cached_count
                
                st.info(f"""
                **Phân tích nguồn kết quả:**
                - Kết quả từ API: {api_count} ({api_count/len(results_df)*100:.1f}%)
                - Kết quả từ cache: {cached_count} ({cached_count/len(results_df)*100:.1f}%)
                - Kết quả dự phòng: {fallback_count} ({fallback_count/len(results_df)*100:.1f}%)
                """)
            else:
                st.info("Chưa có dữ liệu benchmark. Hãy chạy benchmark trước.")

    def generate_random_features(self):
        """
        Tạo bộ tham số ngẫu nhiên cho phương tiện
        
        Returns:
            dict: Dictionary chứa các tham số ngẫu nhiên của phương tiện
        """
        return {
            'Engine Size(L)': np.random.uniform(1.0, 8.0),
            'Cylinders': np.random.randint(3, 12),
            'Fuel Consumption Comb (L/100 km)': np.random.uniform(4.0, 20.0),
            'Horsepower': np.random.uniform(100, 800),
            'Weight (kg)': np.random.uniform(1000, 4000),
            'Year': np.random.randint(2015, 2024)
        } 