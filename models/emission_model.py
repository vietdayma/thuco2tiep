# Mô tả: Lớp chính cho mô hình dự đoán lượng phát thải CO2
# Lớp này chịu trách nhiệm tải dữ liệu, tiền xử lý, huấn luyện và dự đoán

import pandas as pd
import numpy as np
import os
import joblib  # Thư viện lưu/tải mô hình ML
from sklearn.ensemble import RandomForestRegressor  
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu
from sklearn.model_selection import train_test_split  # Chia dữ liệu huấn luyện/kiểm tra

class EmissionModel:
    def __init__(self):
        # Khởi tạo mô hình rừng ngẫu nhiên với 100 cây và hạt giống cố định
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()  # Bộ chuẩn hóa dữ liệu
        self.features = [
            'Engine Size(L)',  # Kích thước động cơ (lít)
            'Cylinders',  # Số xi-lanh
            'Fuel Consumption Comb (L/100 km)',  # Mức tiêu thụ nhiên liệu kết hợp
            'Horsepower',  # Công suất động cơ (tính năng mới)
            'Weight (kg)',  # Trọng lượng xe (tính năng mới)
            'Year'  # Năm sản xuất (tính năng mới)
        ]
        self.target = 'CO2 Emissions(g/km)'  # Biến mục tiêu: lượng phát thải CO2
        self.trained = False  # Trạng thái huấn luyện
        self.model_path = 'models/trained_model.joblib'  # Đường dẫn lưu mô hình
        self.scaler_path = 'models/trained_scaler.joblib'  # Đường dẫn lưu bộ chuẩn hóa

    def load_and_preprocess_data(self, data_path):
        """Tải và tiền xử lý dữ liệu"""
        df = pd.read_csv(data_path)  # Đọc file CSV
        
        # Thêm các tính năng tổng hợp cho mục đích demo
        np.random.seed(42)  # Đặt hạt giống cho việc tạo số ngẫu nhiên
        # Tạo tính năng công suất dựa trên kích thước động cơ
        df['Horsepower'] = df['Engine Size(L)'] * 100 + np.random.normal(0, 10, len(df))
        # Tạo tính năng trọng lượng dựa trên kích thước động cơ
        df['Weight (kg)'] = df['Engine Size(L)'] * 500 + np.random.normal(0, 50, len(df))
        # Tạo tính năm sản xuất ngẫu nhiên
        df['Year'] = np.random.randint(2015, 2024, len(df))
        
        # Ánh xạ các loại nhiên liệu
        fuel_type_mapping = {
            "Z": "Premium Gasoline",  # Xăng cao cấp
            "X": "Regular Gasoline",  # Xăng thông thường
            "D": "Diesel",  # Dầu diesel
            "E": "Ethanol(E85)",  # Nhiên liệu ethanol
            "N": "Natural Gas"  # Khí tự nhiên
        }
        df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
        
        # Loại bỏ các phương tiện sử dụng khí tự nhiên (quá ít mẫu)
        df = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)
        
        return df

    def prepare_features(self, df):
        """Chuẩn bị các đặc trưng cho huấn luyện/dự đoán"""
        X = df[self.features].copy()  # Trích xuất các cột đặc trưng
        if self.target in df.columns:
            y = df[self.target]  # Trích xuất cột mục tiêu nếu có
        else:
            y = None
        return X, y

    def save_model(self):
        """Lưu mô hình đã huấn luyện và bộ chuẩn hóa vào đĩa"""
        # Tạo thư mục models nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Lưu mô hình và bộ chuẩn hóa
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
    def load_model(self):
        """Tải mô hình đã huấn luyện và bộ chuẩn hóa từ đĩa"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.trained = True
            return True
        return False

    def train(self, data_path):
        """Huấn luyện mô hình hoặc tải mô hình đã huấn luyện nếu có"""
        # Thử tải mô hình trước
        if self.load_model():
            print("Đã tải mô hình đã huấn luyện từ đĩa")
            # Vẫn cần tính toán điểm test cho các đánh giá
            df = self.load_and_preprocess_data(data_path)
            X, y = self.prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_test_scaled = self.scaler.transform(X_test)
            test_score = self.model.score(X_test_scaled, y_test)
            return test_score
            
        # Nếu không có mô hình đã huấn luyện, huấn luyện mô hình mới
        df = self.load_and_preprocess_data(data_path)
        X, y = self.prepare_features(df)
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Chuẩn hóa các đặc trưng
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Huấn luyện mô hình
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
        
        # Lưu mô hình đã huấn luyện
        self.save_model()
        
        # Tính toán và trả về các chỉ số
        X_test_scaled = self.scaler.transform(X_test)
        test_score = self.model.score(X_test_scaled, y_test)
        return test_score

    def predict(self, features_dict):
        """Thực hiện dự đoán"""
        if not self.trained:
            raise ValueError("Mô hình cần được huấn luyện trước!")
            
        # Chuyển đổi từ dictionary sang DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Chuẩn hóa các đặc trưng
        features_scaled = self.scaler.transform(features_df)
        
        # Thực hiện dự đoán
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction

    def get_feature_importance(self):
        """Lấy điểm quan trọng của các đặc trưng"""
        if not self.trained:
            raise ValueError("Mô hình cần được huấn luyện trước!")
            
        # Tạo dictionary ánh xạ tên đặc trưng với độ quan trọng tương ứng
        importance_dict = dict(zip(self.features, self.model.feature_importances_))
        return importance_dict 