import streamlit as st  # Thư viện xây dựng giao diện web
from utils.visualization import (  # Module chứa các hàm trực quan hóa dữ liệu
    plot_feature_importance,  # Vẽ biểu đồ độ quan trọng của các đặc trưng
    plot_emission_comparison,  # Vẽ biểu đồ so sánh phát thải
    create_gauge_chart,  # Tạo biểu đồ đồng hồ đo
    style_metric_cards  # Tạo CSS cho các thẻ hiển thị thông số
)
import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng
import time  # Thư viện đo thời gian
import numpy as np  # Thư viện tính toán số học
from utils.benchmark_utils import BenchmarkUtils  # Công cụ đánh giá hiệu suất API
import requests  # Thư viện gọi API HTTP
from concurrent.futures import ThreadPoolExecutor, as_completed  # Hỗ trợ đa luồng
import os  # Thư viện tương tác với hệ điều hành

class MainView:
    def __init__(self, controller):
        """
        Khởi tạo lớp hiển thị chính của ứng dụng.
        
        Tham số:
            controller (EmissionController): Bộ điều khiển xử lý logic dự đoán phát thải.
        """
        self.controller = controller  # Lưu trữ bộ điều khiển để tương tác với mô hình
        self.benchmark_utils = BenchmarkUtils()  # Khởi tạo công cụ benchmark

    def show(self):
        """
        Hiển thị giao diện chính của ứng dụng.
        Điều hướng giữa các trang chức năng khác nhau.
        """
        # Thêm CSS tùy chỉnh cho các thẻ hiển thị
        st.markdown(style_metric_cards(), unsafe_allow_html=True)
        
        # Tạo thanh bên để điều hướng
        with st.sidebar:
            st.markdown("# 🚗 CO2 Emission Predictor")  # Hiển thị tiêu đề trong thanh bên
            st.markdown("---")  # Đường ngăn cách
            page = st.radio("Navigation", ["Prediction", "Analysis", "Benchmark"])  # Các tùy chọn điều hướng

        # Hiển thị trang tương ứng với lựa chọn người dùng
        if page == "Prediction":
            self._show_prediction_page()  # Hiển thị trang dự đoán
        elif page == "Analysis":
            self._show_analysis_page()  # Hiển thị trang phân tích
        else:
            self._show_benchmark_page()  # Hiển thị trang benchmark

    def _show_prediction_page(self):
        """
        Hiển thị giao diện trang dự đoán phát thải CO2.
        Cho phép người dùng nhập các thông số xe và xem kết quả dự đoán.
        """
        st.title("🌍 Predict Vehicle CO2 Emissions")  # Tiêu đề trang
        # Thêm hướng dẫn sử dụng
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px'>
            <h4 style='margin: 0; color: #0f4c81'>Nhập thông số kỹ thuật của xe để dự đoán lượng phát thải CO2</h4>
        </div>
        """, unsafe_allow_html=True)

        # Chia thành hai cột để giao diện gọn gàng hơn
        col1, col2 = st.columns(2)

        # Cột 1: Các thông số cơ bản về động cơ
        with col1:
            # Nhập dung tích động cơ (L)
            engine_size = st.number_input("🔧 Dung tích động cơ (L)", 
                                        min_value=0.1,  # Giá trị tối thiểu
                                        max_value=10.0,  # Giá trị tối đa
                                        value=2.0,  # Giá trị mặc định
                                        step=0.1)  # Bước nhảy
            
            # Nhập số xi-lanh
            cylinders = st.number_input("⚙️ Số xi-lanh",
                                      min_value=2,
                                      max_value=16,
                                      value=4,
                                      step=1)
            
            # Nhập mức tiêu thụ nhiên liệu
            fuel_consumption = st.number_input("⛽ Mức tiêu thụ nhiên liệu (L/100 km)",
                                             min_value=1.0,
                                             max_value=30.0,
                                             value=8.0,
                                             step=0.1)

        # Cột 2: Các thông số khác về xe
        with col2:
            # Nhập công suất động cơ
            horsepower = st.number_input("🏎️ Công suất (mã lực)",
                                       min_value=50,
                                       max_value=1000,
                                       value=200,
                                       step=10)
            
            # Nhập trọng lượng xe
            weight = st.number_input("⚖️ Trọng lượng xe (kg)",
                                   min_value=500,
                                   max_value=5000,
                                   value=1500,
                                   step=100)
            
            # Nhập năm sản xuất
            year = st.number_input("📅 Năm sản xuất",
                                 min_value=2015,
                                 max_value=2024,
                                 value=2023,
                                 step=1)

        # Nút dự đoán phát thải
        if st.button("🔍 Dự đoán phát thải", type="primary"):
            # Tạo từ điển chứa các đặc trưng đầu vào
            features = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption Comb (L/100 km)': fuel_consumption,
                'Horsepower': horsepower,
                'Weight (kg)': weight,
                'Year': year
            }

            # Thực hiện dự đoán và hiển thị kết quả
            try:
                # Lấy kết quả dự đoán và thông tin liên quan
                prediction = self.controller.predict_emission(features)  # Dự đoán phát thải
                avg_emission = self.controller.get_average_emission()  # Lấy mức phát thải trung bình
                rating = self.controller.get_emission_rating(prediction)  # Xếp hạng phát thải (A-F)
                tips = self.controller.get_eco_tips(prediction)  # Lời khuyên thân thiện môi trường

                # Hiển thị kết quả dự đoán
                st.markdown("### 📊 Kết quả dự đoán")
                col1, col2, col3 = st.columns(3)  # Chia thành 3 cột để hiển thị kết quả
                
                # Cột 1: Giá trị phát thải dự đoán
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>🎯 Lượng phát thải CO2</h3>
                            <div class="metric-value">{prediction:.1f} g/km</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Cột 2: Xếp hạng phát thải
                with col2:
                    # Định nghĩa màu cho từng xếp hạng
                    rating_colors = {
                        'A': '🟢', 'B': '🟡', 'C': '🟠',
                        'D': '🔴', 'E': '🟣', 'F': '⚫'
                    }
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>📈 Xếp hạng phát thải</h3>
                            <div class="metric-value">{rating_colors.get(rating, '⚪')} {rating}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Cột 3: So sánh với mức trung bình
                with col3:
                    # Tính phần trăm so với mức trung bình
                    comparison = ((prediction - avg_emission) / avg_emission * 100)
                    icon = "🔽" if comparison < 0 else "🔼"  # Biểu tượng tăng/giảm
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>📊 So với mức trung bình</h3>
                            <div class="metric-value">
                                {icon} {'+' if comparison > 0 else ''}{comparison:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Hiển thị trực quan hóa kết quả
                st.markdown("### 📈 Trực quan hóa")
                col1, col2 = st.columns(2)
                
                # Biểu đồ so sánh với mức trung bình
                with col1:
                    st.pyplot(plot_emission_comparison(prediction, avg_emission))
                
                # Biểu đồ đồng hồ đo mức phát thải
                with col2:
                    st.pyplot(create_gauge_chart(prediction, 0, 300, "Đồng hồ đo phát thải"))

                # Hiển thị các lời khuyên thân thiện môi trường
                st.markdown("### 🌱 Lời khuyên thân thiện môi trường")
                for tip in tips:
                    st.markdown(f"- {tip}")

            # Xử lý lỗi nếu có
            except Exception as e:
                st.error(f"Lỗi khi thực hiện dự đoán: {str(e)}")

    def _show_analysis_page(self):
        """
        Hiển thị trang phân tích các yếu tố ảnh hưởng đến phát thải CO2.
        Trực quan hóa độ quan trọng của các đặc trưng trong mô hình dự đoán.
        """
        st.title("📊 Phân tích phát thải CO2")
        
        # Phân tích độ quan trọng của các đặc trưng
        st.subheader("🎯 Phân tích độ quan trọng của các yếu tố")
        try:
            # Lấy thông tin độ quan trọng các đặc trưng từ controller
            importance_dict = self.controller.get_feature_importance()
            # Hiển thị biểu đồ độ quan trọng
            st.pyplot(plot_feature_importance(importance_dict))
            
            # Thêm giải thích về biểu đồ độ quan trọng
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 20px'>
                <h4 style='margin: 0; color: #0f4c81'>Hiểu về độ quan trọng của các yếu tố</h4>
                <p style='margin-top: 10px'>
                    Biểu đồ này cho thấy mức độ ảnh hưởng của từng đặc điểm xe đến lượng phát thải CO2. 
                    Các thanh dài hơn biểu thị ảnh hưởng mạnh hơn đến kết quả dự đoán.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Xử lý lỗi nếu có
        except Exception as e:
            st.error(f"Lỗi khi lấy thông tin độ quan trọng: {str(e)}")

        # Có thể thêm các phần phân tích khác ở đây 

    def _show_benchmark_page(self):
        """
        Hiển thị trang benchmark để đánh giá hiệu suất của API.
        Cho phép chạy nhiều request đồng thời và đo thời gian phản hồi.
        """
        st.title("⏱️ Benchmark 1000 yêu cầu")
        
        # Lấy API URL từ environment hoặc dùng giá trị mặc định
        API_URL = os.environ.get('API_URL', 'https://thuco2tiep.onrender.com')
        st.info(f"Sử dụng API endpoint: {API_URL}")
        
        # Kiểm tra API health - xem API có sẵn sàng không
        try:
            health_response = requests.get(f"{API_URL}/health")
            if health_response.status_code == 200:
                st.success("API đang hoạt động và sẵn sàng!")
            else:
                st.warning(f"Kiểm tra sức khỏe API thất bại: {health_response.json().get('message', 'Lỗi không xác định')}")
        except Exception as e:
            st.error(f"Không thể kết nối đến API: {str(e)}")
            return

        # Chọn chế độ test
        test_mode = st.radio(
            "Chế độ kiểm tra",
            ["Tham số cố định", "Tham số ngẫu nhiên"]
        )

        # Hiển thị form nhập tham số nếu chọn chế độ cố định
        if test_mode == "Tham số cố định":
            st.subheader("Nhập tham số kiểm tra:")
            col1, col2 = st.columns(2)  # Chia thành 2 cột
            
            # Cột 1: Nhập thông số động cơ
            with col1:
                engine_size = st.number_input("Dung tích động cơ (L)", 
                    min_value=0.1, max_value=10.0, value=2.0, step=0.1)
                cylinders = st.number_input("Số xi-lanh", 
                    min_value=2, max_value=16, value=4, step=1)
                fuel_consumption = st.number_input("Mức tiêu thụ nhiên liệu (L/100km)", 
                    min_value=1.0, max_value=30.0, value=8.0, step=0.1)
            
            # Cột 2: Nhập thông số khác
            with col2:
                horsepower = st.number_input("Công suất (mã lực)", 
                    min_value=50, max_value=1000, value=200, step=10)
                weight = st.number_input("Trọng lượng (kg)", 
                    min_value=500, max_value=5000, value=1500, step=100)
                year = st.number_input("Năm sản xuất", 
                    min_value=2015, max_value=2024, value=2023, step=1)

            # Tạo đặc trưng từ thông số nhập vào
            features = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption Comb (L/100 km)': fuel_consumption,
                'Horsepower': horsepower,
                'Weight (kg)': weight,
                'Year': year
            }
        else:
            # Sử dụng tham số ngẫu nhiên
            features = self.generate_random_features()
            st.info("Mỗi request sẽ sử dụng một bộ tham số ngẫu nhiên khác nhau")
            st.write("Ví dụ tham số ngẫu nhiên:", features)
        
        # Đặt các tham số benchmark
        num_requests = st.slider("Số lượng request", min_value=10, max_value=1000, value=1000, step=10)
        concurrency = st.slider("Số lượng request đồng thời", min_value=1, max_value=50, value=50, step=1)
        
        # Hàm thực hiện một request
        def make_request():
            """
            Hàm nội bộ để thực hiện một request API và đo thời gian.
            
            Trả về:
                dict: Kết quả bao gồm thời gian xử lý và thông tin dự đoán.
            """
            # Tạo đặc trưng ngẫu nhiên nếu đang ở chế độ ngẫu nhiên
            request_features = self.generate_random_features() if test_mode == "Tham số ngẫu nhiên" else features
            
            # Đo thời gian bắt đầu
            start_time = time.perf_counter()
            
            # Thực hiện request và đo thời gian
            try:
                # Tạo session với retry
                session = requests.Session()
                retries = 3
                
                # Tính thời gian mạng và xử lý riêng biệt
                network_start = time.perf_counter()
                response = session.post(f"{API_URL}/predict", json=request_features, timeout=5)
                network_time = (time.perf_counter() - network_start) * 1000  # ms
                
                # Tính tổng thời gian
                total_time = (time.perf_counter() - start_time) * 1000  # ms
                
                # Nếu request thành công
                if response.status_code == 200:
                    data = response.json()
                    processing_time = total_time - network_time
                    
                    return {
                        'total_time': total_time,
                        'network_time': network_time,
                        'processing_time': processing_time,
                        'prediction': data.get('prediction'),
                        'status': 'success'
                    }
                else:
                    # Xử lý lỗi HTTP
                    return {
                        'total_time': total_time,
                        'network_time': network_time,
                        'processing_time': 0,
                        'prediction': None,
                        'status': 'error',
                        'error': f'Mã lỗi HTTP: {response.status_code}'
                    }
            except Exception as e:
                # Xử lý các lỗi khác
                total_time = (time.perf_counter() - start_time) * 1000
                return {
                    'total_time': total_time,
                    'network_time': total_time,  # Gán bằng tổng thời gian vì không có xử lý
                    'processing_time': 0,
                    'prediction': None,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Nút chạy benchmark
        if st.button("▶️ Chạy Benchmark", type="primary"):
            # Khởi tạo công cụ benchmark
            self.benchmark_utils.start_benchmark()
            
            # Tạo placeholder để hiển thị tiến trình
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Thông báo bắt đầu
            status_text.info(f"Đang chạy {num_requests} request với {concurrency} request đồng thời...")
            
            # Sử dụng ThreadPoolExecutor để gửi nhiều request cùng lúc
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Gửi tất cả request
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                
                # Xử lý kết quả khi hoàn thành
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self.benchmark_utils.record_prediction(result)
                    
                    # Cập nhật tiến trình
                    progress = (i + 1) / num_requests
                    progress_bar.progress(progress)
                    status_text.info(f"Đã hoàn thành {i+1}/{num_requests} request ({progress:.1%})")
            
            # Kết thúc benchmark
            self.benchmark_utils.end_benchmark()
            stats = self.benchmark_utils.get_statistics()
            
            # Hiển thị kết quả
            status_text.success("Benchmark hoàn tất!")
            
            # Hiển thị thống kê
            st.subheader("📊 Kết quả Benchmark")
            
            # Hiển thị các thông số chính
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số yêu cầu", f"{stats['total_requests']}")
                st.metric("Yêu cầu thành công", f"{stats['successful_requests']} ({stats['success_rate']:.1f}%)")
            
            with col2:
                st.metric("Thời gian trung bình", f"{stats['avg_total_time']/1000:.3f} s")
                st.metric("Thời gian mạng", f"{stats['avg_network_time']/1000:.3f} s")
            
            with col3:
                st.metric("Thời gian xử lý", f"{stats['avg_processing_time']/1000:.3f} s")
                st.metric("Yêu cầu/giây", f"{stats['requests_per_second']:.2f}")
            
            # Hiển thị biểu đồ
            st.subheader("📈 Biểu đồ thời gian phản hồi")
            st.pyplot(self.benchmark_utils.plot_response_times())
            
            st.subheader("📊 Phân phối thời gian phản hồi")
            st.pyplot(self.benchmark_utils.plot_response_distribution())
            
            # Hiển thị bảng dữ liệu chi tiết
            st.subheader("📋 Dữ liệu chi tiết")
            results_df = self.benchmark_utils.get_results_df()
            st.dataframe(results_df)

    def generate_random_features(self):
        """
        Tạo các đặc trưng ngẫu nhiên để dùng trong benchmark.
        
        Trả về:
            dict: Từ điển chứa các đặc trưng ngẫu nhiên của xe.
        """
        # Tạo các giá trị ngẫu nhiên trong phạm vi hợp lệ
        return {
            'Engine Size(L)': round(np.random.uniform(1.0, 6.0), 1),  # Dung tích động cơ 1.0-6.0L
            'Cylinders': np.random.randint(3, 12),  # Số xi-lanh 3-12
            'Fuel Consumption Comb (L/100 km)': round(np.random.uniform(4.0, 20.0), 1),  # Mức tiêu thụ 4-20L/100km
            'Horsepower': np.random.randint(100, 600),  # Công suất 100-600 mã lực
            'Weight (kg)': np.random.randint(1000, 3000),  # Trọng lượng 1000-3000kg
            'Year': np.random.randint(2015, 2024)  # Năm sản xuất 2015-2023
        } 