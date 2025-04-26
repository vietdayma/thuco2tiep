import streamlit as st  # Thư viện giao diện web
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
import seaborn as sns  # Thư viện vẽ biểu đồ thống kê nâng cao
import pandas as pd  # Thư viện xử lý dữ liệu
import numpy as np  # Thư viện tính toán số học

def plot_feature_importance(importance_dict):
    """
    Vẽ biểu đồ độ quan trọng của các đặc trưng trong mô hình dự đoán.
    
    Biểu đồ thanh ngang cho thấy mức độ ảnh hưởng của từng đặc trưng đến kết quả dự đoán phát thải CO2.
    Giúp người dùng hiểu được yếu tố nào ảnh hưởng nhiều nhất đến lượng khí thải.
    
    Tham số:
        importance_dict (dict): Từ điển chứa tên đặc trưng và giá trị độ quan trọng tương ứng.
        
    Trả về:
        matplotlib.figure.Figure: Đối tượng biểu đồ đã tạo.
    """
    # Tạo hình vẽ với kích thước cụ thể
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Chuyển từ điển thành DataFrame và sắp xếp theo độ quan trọng tăng dần
    # Sắp xếp tăng dần để thanh dài nhất ở trên cùng khi vẽ biểu đồ thanh ngang
    importance_df = pd.DataFrame({
        'Feature': importance_dict.keys(),
        'Importance': importance_dict.values()
    }).sort_values('Importance', ascending=True)
    
    # Vẽ biểu đồ thanh ngang sử dụng seaborn
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    
    # Thêm tiêu đề cho biểu đồ
    plt.title('Độ quan trọng của các yếu tố trong dự đoán phát thải CO2')
    
    return fig

def plot_emission_comparison(prediction, avg_emission):
    """
    Vẽ biểu đồ so sánh giữa giá trị phát thải dự đoán và giá trị trung bình.
    
    Biểu đồ thanh so sánh trực quan giữa lượng phát thải CO2 của xe được dự đoán
    và lượng phát thải trung bình trong tập dữ liệu. Màu sắc biểu thị mức tốt/xấu.
    
    Tham số:
        prediction (float): Giá trị phát thải CO2 được dự đoán (g/km).
        avg_emission (float): Giá trị phát thải CO2 trung bình (g/km).
        
    Trả về:
        matplotlib.figure.Figure: Đối tượng biểu đồ đã tạo.
    """
    # Tạo hình vẽ với kích thước cụ thể
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Tạo danh sách giá trị cần vẽ
    emissions = [avg_emission, prediction]
    labels = ['Mức phát thải trung bình', 'Mức phát thải dự đoán']
    
    # Đặt màu cho các thanh - màu xanh nếu phát thải thấp hơn trung bình, màu đỏ nếu cao hơn
    colors = ['lightgray', 'lightgreen' if prediction < avg_emission else 'lightcoral']
    
    # Vẽ biểu đồ thanh đơn giản
    ax.bar(labels, emissions, color=colors)
    
    # Thêm tiêu đề và nhãn trục y
    plt.title('So sánh mức phát thải CO2')
    plt.ylabel('Phát thải CO2 (g/km)')
    
    # Thêm nhãn giá trị trên mỗi thanh
    for i, v in enumerate(emissions):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    return fig

def create_gauge_chart(value, min_val, max_val, title):
    """
    Tạo biểu đồ đồng hồ đo cho phát thải CO2.
    
    Biểu đồ dạng đồng hồ đo cung cấp cách trực quan để thấy mức phát thải
    nằm ở đâu trong thang đo từ thấp (tốt) đến cao (xấu). Sử dụng đồ thị cực
    để tạo hiệu ứng đồng hồ đo.
    
    Tham số:
        value (float): Giá trị cần hiển thị trên đồng hồ.
        min_val (float): Giá trị tối thiểu trên thang đo.
        max_val (float): Giá trị tối đa trên thang đo.
        title (str): Tiêu đề của biểu đồ.
        
    Trả về:
        matplotlib.figure.Figure: Đối tượng biểu đồ đã tạo.
    """
    # Tạo hình vẽ với tọa độ cực (polar)
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
    
    # Chuyển đổi giá trị thành góc trên đồ thị cực
    # Giá trị min_val tương ứng với góc 0, max_val tương ứng với góc π (180 độ)
    angle = (value - min_val) / (max_val - min_val) * np.pi
    
    # Thiết lập hướng và độ lệch cho đồ thị
    ax.set_theta_direction(-1)  # Ngược chiều kim đồng hồ
    ax.set_theta_offset(np.pi/2)  # Bắt đầu từ trên đỉnh (90 độ)
    
    # Vẽ kim đồng hồ (một đường thẳng từ tâm đến giá trị)
    ax.plot([0, angle], [0, 0.9], color='red', linewidth=3)
    
    # Tùy chỉnh đồ thị
    ax.set_rticks([])  # Ẩn vạch tròn bán kính
    
    # Thiết lập nhãn cho các mức trên đồng hồ
    ax.set_xticks(np.linspace(0, np.pi, 5))  # 5 mức chia đều từ 0 đến π
    ax.set_xticklabels([f'{v:.0f}' for v in np.linspace(min_val, max_val, 5)])
    
    # Thêm tiêu đề
    plt.title(title)
    
    return fig

def style_metric_cards():
    """
    Tạo CSS tùy chỉnh cho các thẻ hiển thị thông số trong giao diện.
    
    Định dạng các thẻ hiển thị kết quả dự đoán, xếp hạng phát thải và
    so sánh với mức trung bình. Tạo giao diện đẹp mắt và dễ đọc.
    
    Trả về:
        str: Chuỗi CSS để nhúng vào Streamlit.
    """
    # Trả về mã CSS để tạo kiểu cho các thẻ thông số
    return """
    <style>
        .metric-card {
            background-color: #f0f2f6;  /* Màu nền xám nhạt */
            border-radius: 10px;  /* Bo góc */
            padding: 20px;  /* Khoảng cách nội dung và viền */
            margin: 10px;  /* Khoảng cách giữa các thẻ */
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);  /* Đổ bóng nhẹ */
        }
        .metric-card h3 {
            color: #0f4c81;  /* Màu xanh đậm cho tiêu đề */
            margin-bottom: 10px;  /* Khoảng cách dưới tiêu đề */
        }
        .metric-value {
            font-size: 24px;  /* Cỡ chữ lớn cho giá trị */
            font-weight: bold;  /* Chữ đậm */
            color: #1f77b4;  /* Màu xanh cho giá trị */
        }
    </style>
    """