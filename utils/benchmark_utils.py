import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Lớp BenchmarkUtils: Công cụ đánh giá hiệu năng API và dịch vụ dự đoán khí thải CO2
# Lớp này giúp ghi lại thời gian phản hồi, tỷ lệ thành công và phân tích hiệu suất của quá trình dự đoán
class BenchmarkUtils:
    def __init__(self):
        """
        Khởi tạo đối tượng BenchmarkUtils với các thuộc tính cơ bản
        - results: danh sách lưu kết quả các lần dự đoán
        - start_time: thời điểm bắt đầu phiên benchmark
        - end_time: thời điểm kết thúc phiên benchmark
        """
        self.results = []
        self.start_time = None
        self.end_time = None
        
    def start_benchmark(self):
        """
        Bắt đầu phiên benchmark mới và khởi tạo lại danh sách kết quả
        Phương thức này ghi lại thời điểm bắt đầu sử dụng high-precision timer
        """
        self.start_time = time.perf_counter()
        self.results = []
        
    def record_prediction(self, timing_data):
        """
        Ghi lại kết quả và số liệu của một lần dự đoán
        
        Tham số:
            timing_data (dict): Dictionary chứa dữ liệu về thời gian và kết quả dự đoán:
                - total_time: tổng thời gian thực hiện (ms)
                - network_time: thời gian truyền dữ liệu qua mạng (ms)
                - processing_time: thời gian xử lý dự đoán (ms)
                - prediction: giá trị dự đoán
                - status: trạng thái ('success' hoặc 'error')
                - error: thông tin lỗi (nếu có)
        """
        # Đảm bảo các trường dữ liệu cần thiết đều tồn tại với giá trị mặc định
        timing_data = {
            'timestamp': datetime.now(),
            'total_time': timing_data.get('total_time', 0),
            'network_time': timing_data.get('network_time', 0),
            'processing_time': timing_data.get('processing_time', 0),
            'prediction': timing_data.get('prediction'),
            'status': timing_data.get('status', 'error'),
            'error': timing_data.get('error')
        }
        
        self.results.append(timing_data)
        
    def end_benchmark(self):
        """
        Kết thúc phiên benchmark và ghi lại thời điểm kết thúc
        """
        self.end_time = time.perf_counter()
        
    def get_statistics(self):
        """
        Tính toán các số liệu thống kê từ kết quả benchmark
        
        Trả về:
            dict: Dictionary chứa các thông số:
                - total_time: tổng thời gian của phiên benchmark (s)
                - total_requests: tổng số lượng request
                - successful_requests: số lượng request thành công
                - requests_per_second: số request/giây
                - success_rate: tỷ lệ thành công (%)
                - avg_total_time: thời gian phản hồi trung bình (ms)
                - avg_network_time: thời gian mạng trung bình (ms)
                - avg_processing_time: thời gian xử lý trung bình (ms)
                - min_response_time: thời gian phản hồi nhỏ nhất (ms)
                - max_response_time: thời gian phản hồi lớn nhất (ms)
        """
        if not self.results:
            return {
                'total_time': 0,
                'total_requests': 0,
                'successful_requests': 0,
                'requests_per_second': 0,
                'success_rate': 0,
                'avg_total_time': 0,
                'avg_network_time': 0,
                'avg_processing_time': 0,
                'min_response_time': 0,
                'max_response_time': 0
            }
            
        df = pd.DataFrame(self.results)
        successful_df = df[df['status'] == 'success']
        
        total_time = self.end_time - self.start_time
        total_requests = len(df)
        successful_requests = len(successful_df)
        
        # Chỉ tính toán thống kê nếu có các request thành công
        if successful_requests > 0:
            avg_total_time = successful_df['total_time'].mean()
            avg_network_time = successful_df['network_time'].mean()
            avg_processing_time = successful_df['processing_time'].mean()
            min_response_time = successful_df['total_time'].min()
            max_response_time = successful_df['total_time'].max()
        else:
            avg_total_time = avg_network_time = avg_processing_time = min_response_time = max_response_time = 0
        
        stats = {
            'total_time': total_time,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'requests_per_second': total_requests / total_time if total_time > 0 else 0,
            'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            'avg_total_time': avg_total_time,
            'avg_network_time': avg_network_time,
            'avg_processing_time': avg_processing_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time
        }
        
        return stats
    
    def plot_response_times(self):
        """
        Tạo biểu đồ xu hướng thời gian phản hồi theo từng request
        
        Biểu đồ hiển thị 3 đường:
        - Tổng thời gian phản hồi
        - Thời gian mạng
        - Thời gian xử lý
        
        Trả về:
            matplotlib.figure: Đối tượng figure chứa biểu đồ
        """
        df = pd.DataFrame(self.results)
        successful_df = df[df['status'] == 'success']
        
        if successful_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'Không có dữ liệu request thành công để hiển thị', 
                   ha='center', va='center')
            ax.set_xlabel('Số thứ tự request')
            ax.set_ylabel('Thời gian (ms)')
            ax.set_title('Biểu đồ phân tích thời gian phản hồi')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Vẽ tổng thời gian
        ax.plot(range(len(successful_df)), successful_df['total_time'], 
                label='Tổng thời gian', color='blue')
        
        # Vẽ thời gian mạng
        ax.plot(range(len(successful_df)), successful_df['network_time'],
                label='Thời gian mạng', color='red', alpha=0.7)
        
        # Vẽ thời gian xử lý
        ax.plot(range(len(successful_df)), successful_df['processing_time'],
                label='Thời gian xử lý', color='green', alpha=0.7)
        
        ax.set_xlabel('Số thứ tự request')
        ax.set_ylabel('Thời gian (ms)')
        ax.set_title('Biểu đồ phân tích thời gian phản hồi')
        ax.legend()
        plt.grid(True, alpha=0.3)
        return fig
    
    def plot_response_distribution(self):
        """
        Tạo biểu đồ phân phối thời gian phản hồi
        
        Bao gồm 3 biểu đồ histogram:
        - Phân phối tổng thời gian phản hồi
        - Phân phối thời gian mạng
        - Phân phối thời gian xử lý
        
        Trả về:
            matplotlib.figure: Đối tượng figure chứa biểu đồ
        """
        df = pd.DataFrame(self.results)
        successful_df = df[df['status'] == 'success']
        
        if successful_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'Không có dữ liệu request thành công để hiển thị', 
                   ha='center', va='center')
            ax.set_xlabel('Thời gian (ms)')
            ax.set_ylabel('Tần suất')
            ax.set_title('Phân phối thời gian phản hồi')
            return fig
            
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Phân phối tổng thời gian
        ax1.hist(successful_df['total_time'], bins=30, color='blue', alpha=0.7)
        ax1.set_xlabel('Tổng thời gian (ms)')
        ax1.set_ylabel('Tần suất')
        ax1.set_title('Tổng thời gian phản hồi')
        ax1.grid(True, alpha=0.3)
        
        # Phân phối thời gian mạng
        ax2.hist(successful_df['network_time'], bins=30, color='red', alpha=0.7)
        ax2.set_xlabel('Thời gian mạng (ms)')
        ax2.set_title('Thời gian mạng')
        ax2.grid(True, alpha=0.3)
        
        # Phân phối thời gian xử lý
        ax3.hist(successful_df['processing_time'], bins=30, color='green', alpha=0.7)
        ax3.set_xlabel('Thời gian xử lý (ms)')
        ax3.set_title('Thời gian xử lý')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_results_df(self):
        """
        Chuyển đổi kết quả benchmark thành DataFrame để phân tích
        
        Bổ sung thêm các cột:
        - request_number: số thứ tự của request
        - network_percentage: phần trăm thời gian mạng so với tổng thời gian
        - processing_percentage: phần trăm thời gian xử lý so với tổng thời gian
        
        Trả về:
            pandas.DataFrame: DataFrame chứa kết quả chi tiết
        """
        df = pd.DataFrame(self.results)
        if not df.empty:
            # Tính toán các phần trăm
            total_time = df['total_time']
            df['network_percentage'] = (df['network_time'] / total_time * 100).round(2)
            df['processing_percentage'] = (df['processing_time'] / total_time * 100).round(2)
            
            # Thêm số thứ tự request
            df['request_number'] = range(1, len(df) + 1)
            
            # Sắp xếp lại các cột
            columns = ['request_number', 'timestamp', 'total_time', 'network_time', 
                      'processing_time', 'network_percentage', 'processing_percentage',
                      'prediction', 'status', 'error']
            df = df[columns]
        return df 