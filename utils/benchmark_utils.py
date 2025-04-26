# Mô tả: Công cụ để chạy và phân tích các bài kiểm tra hiệu suất (benchmark)
# Lớp này theo dõi và tính toán các thông số về tốc độ, độ chính xác và phân phối thời gian

import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class BenchmarkUtils:
    def __init__(self):
        self.results = []  # Danh sách lưu kết quả của từng lần dự đoán
        self.start_time = None  # Thời điểm bắt đầu benchmark
        self.end_time = None  # Thời điểm kết thúc benchmark
        
    def start_benchmark(self):
        """Bắt đầu phiên benchmark"""
        self.start_time = time.perf_counter()  # Lưu thời điểm bắt đầu với độ chính xác cao
        self.results = []  # Xóa kết quả cũ
        
    def record_prediction(self, timing_data):
        """Ghi lại kết quả dự đoán với các số liệu về mạng"""
        # Đảm bảo tất cả các trường cần thiết tồn tại với giá trị mặc định
        timing_data = {
            'timestamp': datetime.now(),  # Thời điểm ghi lại
            'total_time': timing_data.get('total_time', 0),  # Tổng thời gian
            'network_time': timing_data.get('network_time', 0),  # Thời gian mạng
            'processing_time': timing_data.get('processing_time', 0),  # Thời gian xử lý
            'prediction': timing_data.get('prediction'),  # Giá trị dự đoán
            'status': timing_data.get('status', 'error'),  # Trạng thái (thành công/lỗi)
            'error': timing_data.get('error')  # Thông báo lỗi nếu có
        }
        
        self.results.append(timing_data)  # Thêm kết quả vào danh sách
        
    def end_benchmark(self):
        """Kết thúc phiên benchmark"""
        self.end_time = time.perf_counter()  # Lưu thời điểm kết thúc
        
    def get_statistics(self):
        """Tính toán các thống kê benchmark bao gồm các số liệu về mạng"""
        if not self.results:
            # Trả về các giá trị mặc định nếu không có kết quả
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
            
        df = pd.DataFrame(self.results)  # Chuyển đổi kết quả thành DataFrame
        successful_df = df[df['status'] == 'success']  # Lọc các yêu cầu thành công
        
        total_time = self.end_time - self.start_time  # Tính tổng thời gian
        total_requests = len(df)  # Tổng số yêu cầu
        successful_requests = len(successful_df)  # Số yêu cầu thành công
        
        # Tính toán thống kê chỉ khi có các yêu cầu thành công
        if successful_requests > 0:
            avg_total_time = successful_df['total_time'].mean()  # Thời gian trung bình
            avg_network_time = successful_df['network_time'].mean()  # Thời gian mạng trung bình
            avg_processing_time = successful_df['processing_time'].mean()  # Thời gian xử lý trung bình
            min_response_time = successful_df['total_time'].min()  # Thời gian phản hồi tối thiểu
            max_response_time = successful_df['total_time'].max()  # Thời gian phản hồi tối đa
        else:
            avg_total_time = avg_network_time = avg_processing_time = min_response_time = max_response_time = 0
        
        # Tổng hợp tất cả thống kê
        stats = {
            'total_time': total_time,  # Tổng thời gian (giây)
            'total_requests': total_requests,  # Tổng số yêu cầu
            'successful_requests': successful_requests,  # Số yêu cầu thành công
            'requests_per_second': total_requests / total_time if total_time > 0 else 0,  # Tốc độ (yêu cầu/giây)
            'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,  # Tỷ lệ thành công (%)
            'avg_total_time': avg_total_time,  # Thời gian trung bình (ms)
            'avg_network_time': avg_network_time,  # Thời gian mạng trung bình (ms)
            'avg_processing_time': avg_processing_time,  # Thời gian xử lý trung bình (ms)
            'min_response_time': min_response_time,  # Thời gian phản hồi tối thiểu (ms)
            'max_response_time': max_response_time  # Thời gian phản hồi tối đa (ms)
        }
        
        return stats
    
    def plot_response_times(self):
        """Tạo biểu đồ xu hướng thời gian phản hồi với phân tích mạng"""
        df = pd.DataFrame(self.results)  # Chuyển đổi kết quả thành DataFrame
        successful_df = df[df['status'] == 'success']  # Lọc các yêu cầu thành công
        
        if successful_df.empty:
            # Tạo biểu đồ trống nếu không có yêu cầu thành công
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'Không có yêu cầu thành công để vẽ biểu đồ', 
                   ha='center', va='center')
            ax.set_xlabel('Số thứ tự yêu cầu')
            ax.set_ylabel('Thời gian (ms)')
            ax.set_title('Phân tích thời gian phản hồi')
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
        
        ax.set_xlabel('Số thứ tự yêu cầu')
        ax.set_ylabel('Thời gian (ms)')
        ax.set_title('Phân tích thời gian phản hồi')
        ax.legend()
        plt.grid(True, alpha=0.3)
        return fig
    
    def plot_response_distribution(self):
        """Tạo biểu đồ phân phối thời gian phản hồi với phân tích mạng"""
        df = pd.DataFrame(self.results)  # Chuyển đổi kết quả thành DataFrame
        successful_df = df[df['status'] == 'success']  # Lọc các yêu cầu thành công
        
        if successful_df.empty:
            # Tạo biểu đồ trống nếu không có yêu cầu thành công
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, 'Không có yêu cầu thành công để vẽ biểu đồ', 
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
        """Lấy kết quả dưới dạng DataFrame với các số liệu về mạng"""
        df = pd.DataFrame(self.results)  # Chuyển đổi kết quả thành DataFrame
        if not df.empty:
            # Tính toán phần trăm
            total_time = df['total_time']
            df['network_percentage'] = (df['network_time'] / total_time * 100).round(2)  # Phần trăm thời gian mạng
            df['processing_percentage'] = (df['processing_time'] / total_time * 100).round(2)  # Phần trăm thời gian xử lý
            
            # Thêm số thứ tự yêu cầu
            df['request_number'] = range(1, len(df) + 1)
            
            # Sắp xếp lại cột
            columns = ['request_number', 'timestamp', 'total_time', 'network_time', 
                      'processing_time', 'network_percentage', 'processing_percentage',
                      'prediction', 'status', 'error']
            df = df[columns]
        return df 