# Mô tả: Các hàm để tạo biểu đồ và trực quan hóa dữ liệu
# Module này cung cấp các hàm để hiển thị kết quả dự đoán CO2 và phân tích dữ liệu
# Các biểu đồ được sử dụng trong giao diện Streamlit để hiển thị kết quả phân tích

import streamlit as st  # Thư viện tạo giao diện web
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ cơ bản
import seaborn as sns  # Thư viện vẽ biểu đồ nâng cao, dựa trên matplotlib
import pandas as pd  # Thư viện xử lý dữ liệu
import numpy as np  # Thư viện tính toán số học

def plot_feature_importance(importance_dict):
    """Vẽ biểu đồ điểm quan trọng của các đặc trưng
    
    Input: importance_dict - Dictionary chứa tên đặc trưng và giá trị độ quan trọng
    Output: fig - Đối tượng matplotlib Figure chứa biểu đồ thanh ngang
    
    Biểu đồ này hiển thị mức độ ảnh hưởng của từng đặc trưng đến kết quả dự đoán
    Các đặc trưng được sắp xếp tăng dần theo độ quan trọng (thanh dài hơn = quan trọng hơn)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': importance_dict.keys(),  # Tên các đặc trưng
        'Importance': importance_dict.values()  # Giá trị độ quan trọng
    }).sort_values('Importance', ascending=True)  # Sắp xếp tăng dần
    
    # Sử dụng seaborn để vẽ biểu đồ thanh ngang
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    plt.title('Độ quan trọng của các đặc trưng trong dự đoán khí thải CO2')
    return fig

def plot_emission_comparison(prediction, avg_emission):
    """Vẽ biểu đồ so sánh dự đoán với lượng khí thải trung bình
    
    Input:
        prediction - Giá trị dự đoán khí thải CO2 (g/km)
        avg_emission - Giá trị trung bình khí thải CO2 (g/km)
    
    Output: 
        fig - Đối tượng matplotlib Figure chứa biểu đồ cột so sánh
    
    Biểu đồ tạo ra có màu xanh khi dự đoán thấp hơn giá trị trung bình (tốt)
    và màu đỏ khi dự đoán cao hơn giá trị trung bình (xấu)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    emissions = [avg_emission, prediction]  # Giá trị khí thải
    labels = ['Khí thải trung bình', 'Khí thải dự đoán']  # Nhãn
    # Màu sắc: xanh lá nếu dự đoán thấp hơn trung bình, đỏ nếu cao hơn
    colors = ['lightgray', 'lightgreen' if prediction < avg_emission else 'lightcoral']
    
    # Vẽ biểu đồ cột
    ax.bar(labels, emissions, color=colors)
    plt.title('So sánh khí thải dự đoán với khí thải trung bình')
    plt.ylabel('Khí thải (g/km)')
    
    # Thêm các giá trị nhãn trên các cột
    for i, v in enumerate(emissions):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    return fig

def create_gauge_chart(value, min_val, max_val, title):
    """Create a gauge chart for emissions
    
    Input:
        value - Giá trị cần hiển thị (khí thải CO2)
        min_val - Giá trị tối thiểu của thang đo
        max_val - Giá trị tối đa của thang đo
        title - Tiêu đề biểu đồ
    
    Output:
        fig - Đối tượng matplotlib Figure chứa biểu đồ đồng hồ đo
    
    Biểu đồ đồng hồ đo (gauge) hiển thị giá trị khí thải dưới dạng kim chỉ
    Sử dụng biểu đồ cực (polar plot) với góc quay từ 0 đến π để tạo dạng bán nguyệt
    """
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
    
    # Convert value to angle
    angle = (value - min_val) / (max_val - min_val) * np.pi
    
    # Create the gauge
    ax.set_theta_direction(-1)  # Đảo ngược hướng quay để tạo ra thang đo
    ax.set_theta_offset(np.pi/2)  # Bắt đầu từ vị trí giữa (π/2)
    
    # Plot the gauge
    ax.plot([0, angle], [0, 0.9], color='red', linewidth=3)  # Vẽ kim chỉ với màu đỏ
    
    # Customize the chart
    ax.set_rticks([])  # Ẩn đi các vòng tròn đồng tâm
    ax.set_xticks(np.linspace(0, np.pi, 5))  # Tạo 5 điểm chia trên thang đo
    ax.set_xticklabels([f'{v:.0f}' for v in np.linspace(min_val, max_val, 5)])  # Gán nhãn cho các điểm chia
    
    plt.title(title)
    return fig

def style_metric_cards():
    """Return CSS styling for metric cards
    
    Output:
        CSS string - Mã CSS để tạo kiểu cho thẻ hiển thị thông số (metric cards)
    
    CSS này được sử dụng với st.markdown(..., unsafe_allow_html=True) trong Streamlit
    để tạo giao diện đẹp hơn cho các thẻ hiển thị kết quả
    """
    return """
    <style>
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            color: #0f4c81;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #1f77b4;
        }
    </style>
    """ 