from models.emission_model import EmissionModel
import pandas as pd
import requests
import os
import logging

# Cấu hình logging - thiết lập hệ thống ghi log với mức INFO
# (Logging là quá trình ghi lại thông tin về hoạt động của phần mềm để hỗ trợ gỡ lỗi)
logging.basicConfig(
    level=logging.INFO,  # Mức độ chi tiết của log (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Định dạng log: thời gian - mức độ - nội dung
)
logger = logging.getLogger(__name__)  # Tạo logger cho module hiện tại

class EmissionController:
    def __init__(self):
        """
        Khởi tạo bộ điều khiển mô hình phát thải.
        Đây là lớp trung gian giữa mô hình và API/giao diện người dùng.
        Sử dụng mẫu thiết kế MVC (Model-View-Controller) để tách biệt logic
        xử lý dữ liệu và hiển thị kết quả.
        """
        # Tạo đối tượng mô hình phát thải (đây là lớp Model trong mẫu MVC)
        self.model = EmissionModel()
        
        # Cờ đánh dấu trạng thái huấn luyện (True: đã huấn luyện, False: chưa huấn luyện)
        self.trained = False
        
        # Giá trị phát thải trung bình (được tính sau khi huấn luyện)
        # Dùng để so sánh với kết quả dự đoán
        self.avg_emission = None
        
        # URL API để gửi yêu cầu dự đoán (lấy từ biến môi trường hoặc mặc định)
        # Cho phép triển khai linh hoạt trên nhiều môi trường khác nhau
        self.api_url = os.environ.get('API_URL', 'http://localhost:10000') + "/predict"

    def initialize_model(self, data_path):
        """
        Khởi tạo và huấn luyện mô hình phát thải.
        
        Tham số:
            data_path (str): Đường dẫn đến file dữ liệu CSV.
            
        Trả về:
            float: Điểm đánh giá mô hình trên tập kiểm tra (R² score).
        """
        logger.info("Đang khởi tạo mô hình...")  # Ghi log thông tin
        
        # Thử tải mô hình đã lưu trước đó để tiết kiệm thời gian huấn luyện
        if self.model.load_model():
            logger.info("Đã tải thành công mô hình đã huấn luyện trước đó")
            self.trained = True
        else:
            logger.info("Không tìm thấy mô hình đã huấn luyện. Đang huấn luyện mô hình mới...")
            
        # Huấn luyện mô hình và lấy điểm đánh giá (có thể là tải lại mô hình đã huấn luyện)
        test_score = self.model.train(data_path)
        self.trained = True  # Đánh dấu mô hình đã được huấn luyện
        
        # Tính giá trị phát thải trung bình từ dữ liệu để so sánh
        df = self.model.load_and_preprocess_data(data_path)
        self.avg_emission = df['CO2 Emissions(g/km)'].mean()
        
        logger.info(f"Hoàn tất khởi tạo mô hình. Điểm đánh giá tập kiểm tra: {test_score:.3f}")
        return test_score

    def predict_emission(self, features):
        """
        Dự đoán lượng phát thải CO2 dựa trên các đặc trưng xe.
        
        Tham số:
            features (dict): Từ điển chứa các đặc trưng của xe như
                            dung tích động cơ, số xi-lanh, tiêu thụ nhiên liệu...
            
        Trả về:
            float: Lượng phát thải CO2 dự đoán (g/km).
            
        Raises:
            ValueError: Nếu mô hình chưa được huấn luyện.
        """
        # Kiểm tra mô hình đã được huấn luyện chưa để tránh lỗi
        if not self.trained:
            raise ValueError("Mô hình cần phải được huấn luyện trước!")
        
        # Gọi phương thức dự đoán từ mô hình
        return self.model.predict(features)

    def predict_emission_api(self, features):
        """
        Dự đoán lượng phát thải CO2 bằng cách gọi API.
        Trả về toàn bộ phản hồi bao gồm cả thời gian xử lý.
        Phương thức này cho phép phân tán tải khi hệ thống có nhiều người dùng.
        
        Tham số:
            features (dict): Từ điển chứa các đặc trưng của xe.
            
        Trả về:
            dict: Phản hồi từ API chứa kết quả dự đoán và thông tin khác.
            
        Raises:
            Exception: Nếu gọi API thất bại.
        """
        try:
            # Gửi yêu cầu POST đến API với dữ liệu JSON
            response = requests.post(self.api_url, json=features)
            response.raise_for_status()  # Tạo ngoại lệ nếu mã phản hồi không thành công
            
            # Trả về dữ liệu JSON từ phản hồi
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Yêu cầu API thất bại: {str(e)}")

    def get_feature_importance(self):
        """
        Lấy độ quan trọng của từng đặc trưng trong mô hình.
        Điều này cho biết đặc trưng nào ảnh hưởng nhiều nhất đến kết quả dự đoán.
        
        Trả về:
            dict: Từ điển với khóa là tên đặc trưng và giá trị là mức độ quan trọng.
            
        Raises:
            ValueError: Nếu mô hình chưa được huấn luyện.
        """
        # Kiểm tra mô hình đã được huấn luyện chưa
        if not self.trained:
            raise ValueError("Mô hình cần phải được huấn luyện trước!")
        
        # Lấy thông tin độ quan trọng từ mô hình
        return self.model.get_feature_importance()

    def get_average_emission(self):
        """
        Lấy giá trị phát thải CO2 trung bình từ dữ liệu huấn luyện.
        Sử dụng để so sánh với kết quả dự đoán.
        
        Trả về:
            float: Giá trị phát thải trung bình (g/km).
        """
        return self.avg_emission

    def get_emission_rating(self, emission_value):
        """
        Đánh giá mức độ phát thải CO2 theo thang từ A đến F.
        Giúp người dùng dễ dàng hiểu được mức độ phát thải của xe.
        
        Tham số:
            emission_value (float): Giá trị phát thải CO2 (g/km).
            
        Trả về:
            str: Mức đánh giá từ A (tốt nhất) đến F (kém nhất).
        """
        # Đánh giá dựa trên các ngưỡng phát thải
        if emission_value < 100:
            return 'A'  # Rất tốt - mức phát thải rất thấp
        elif emission_value < 120:
            return 'B'  # Tốt - mức phát thải thấp
        elif emission_value < 140:
            return 'C'  # Khá - mức phát thải trung bình thấp
        elif emission_value < 160:
            return 'D'  # Trung bình - mức phát thải trung bình
        elif emission_value < 180:
            return 'E'  # Kém - mức phát thải cao
        else:
            return 'F'  # Rất kém - mức phát thải rất cao

    def get_eco_tips(self, emission_value):
        """
        Cung cấp các lời khuyên thân thiện với môi trường dựa trên mức phát thải.
        Giúp người dùng biết cách giảm thiểu tác động môi trường.
        
        Tham số:
            emission_value (float): Giá trị phát thải CO2 (g/km).
            
        Trả về:
            list: Danh sách các lời khuyên để giảm phát thải.
        """
        tips = []
        
        # Lời khuyên cho xe có mức phát thải cao
        if emission_value > 160:
            tips.extend([
                "Cân nhắc chuyển sang xe tiết kiệm nhiên liệu hơn",
                "Bảo dưỡng xe thường xuyên giúp giảm lượng khí thải",
                "Tránh tăng tốc và phanh đột ngột khi lái xe"
            ])
            
        # Lời khuyên cho xe có mức phát thải trung bình
        if emission_value > 140:
            tips.extend([
                "Kiểm tra áp suất lốp xe thường xuyên",
                "Loại bỏ các trọng lượng không cần thiết khỏi xe"
            ])
            
        # Lời khuyên chung cho tất cả các xe
        tips.extend([
            "Sử dụng kỹ thuật lái xe thân thiện với môi trường",
            "Lên kế hoạch hành trình trước để tránh tắc nghẽn giao thông"
        ])
        
        return tips 