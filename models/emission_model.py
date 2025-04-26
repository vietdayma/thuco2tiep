import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng
import numpy as np   # Thư viện tính toán số học
import os           # Thư viện tương tác với hệ điều hành
import joblib       # Thư viện lưu trữ và tải mô hình học máy
from sklearn.ensemble import RandomForestRegressor  # Thuật toán rừng ngẫu nhiên cho hồi quy
from sklearn.preprocessing import StandardScaler    # Chuẩn hóa dữ liệu
from sklearn.model_selection import train_test_split  # Chia tập dữ liệu thành tập huấn luyện và kiểm tra

class EmissionModel:
    def __init__(self):
        """
        Khởi tạo mô hình dự đoán phát thải CO2.
        Thiết lập mô hình RandomForest, bộ chuẩn hóa dữ liệu và các tham số cần thiết.
        RandomForest được chọn vì khả năng xử lý dữ liệu phi tuyến tính và độ chính xác cao.
        """
        # Khởi tạo mô hình RandomForest với 100 cây quyết định và giá trị ngẫu nhiên cố định
        # n_estimators: số lượng cây trong rừng, random_state: hạt giống ngẫu nhiên để kết quả có thể lặp lại
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Bộ chuẩn hóa để đưa các đặc trưng về cùng một thang đo
        # Giúp thuật toán học máy hoạt động tốt hơn và hội tụ nhanh hơn
        self.scaler = StandardScaler()
        
        # Danh sách các đặc trưng đầu vào cho mô hình - những yếu tố ảnh hưởng đến phát thải CO2
        self.features = [
            'Engine Size(L)',              # Dung tích động cơ (lít)
            'Cylinders',                   # Số xi-lanh
            'Fuel Consumption Comb (L/100 km)',  # Mức tiêu thụ nhiên liệu kết hợp
            'Horsepower',                  # Công suất động cơ (mã lực)
            'Weight (kg)',                 # Trọng lượng xe (kg)
            'Year'                         # Năm sản xuất
        ]
        
        # Tên cột chứa giá trị đầu ra (lượng CO2 thải ra) - biến mục tiêu cần dự đoán
        self.target = 'CO2 Emissions(g/km)'
        
        # Cờ đánh dấu trạng thái huấn luyện của mô hình (True: đã huấn luyện, False: chưa huấn luyện)
        self.trained = False
        
        # Đường dẫn lưu trữ mô hình và bộ chuẩn hóa
        # Cho phép tái sử dụng mô hình đã huấn luyện mà không cần huấn luyện lại
        self.model_path = 'models/trained_model.joblib'
        self.scaler_path = 'models/trained_scaler.joblib'

    def load_and_preprocess_data(self, data_path):
        """
        Tải và tiền xử lý dữ liệu từ file CSV.
        Tiền xử lý dữ liệu là bước quan trọng để chuẩn bị dữ liệu phù hợp cho mô hình.
        
        Tham số:
            data_path (str): Đường dẫn đến file dữ liệu CSV.
            
        Trả về:
            DataFrame: Dữ liệu đã được tiền xử lý.
        """
        # Đọc dữ liệu từ file CSV vào DataFrame pandas
        df = pd.read_csv(data_path)
        
        # Tạo thêm các đặc trưng tổng hợp cho mô hình (feature engineering)
        # Đặt seed ngẫu nhiên để kết quả có thể lặp lại
        np.random.seed(42)  # Đảm bảo tính nhất quán khi chạy nhiều lần
        
        # Tạo đặc trưng Horsepower dựa trên dung tích động cơ với nhiễu ngẫu nhiên
        # Nhiễu ngẫu nhiên giúp tăng tính đa dạng của dữ liệu mà vẫn giữ quan hệ với dung tích động cơ
        df['Horsepower'] = df['Engine Size(L)'] * 100 + np.random.normal(0, 10, len(df))
        
        # Tạo đặc trưng Weight dựa trên dung tích động cơ với nhiễu ngẫu nhiên
        # Xe có động cơ lớn hơn thường nặng hơn, cộng với nhiễu để tạo tính tự nhiên
        df['Weight (kg)'] = df['Engine Size(L)'] * 500 + np.random.normal(0, 50, len(df))
        
        # Tạo đặc trưng Year ngẫu nhiên từ 2015 đến 2023
        # Mô phỏng dữ liệu năm sản xuất của xe
        df['Year'] = np.random.randint(2015, 2024, len(df))
        
        # Chuyển đổi mã loại nhiên liệu thành tên đầy đủ dễ hiểu hơn
        # Tạo dictionary ánh xạ từ mã sang tên đầy đủ
        fuel_type_mapping = {
            "Z": "Premium Gasoline",   # Xăng cao cấp
            "X": "Regular Gasoline",   # Xăng thường
            "D": "Diesel",             # Dầu diesel
            "E": "Ethanol(E85)",       # Ethanol E85
            "N": "Natural Gas"         # Khí tự nhiên
        }
        df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)  # Áp dụng ánh xạ
        
        # Loại bỏ xe sử dụng khí tự nhiên (do ít mẫu)
        # Lọc bỏ dòng có giá trị "Natural Gas" và reset lại chỉ số của DataFrame
        df = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)
        
        return df

    def prepare_features(self, df):
        """
        Chuẩn bị dữ liệu đặc trưng đầu vào và nhãn đầu ra.
        Tách dữ liệu thành đặc trưng X và biến mục tiêu y.
        
        Tham số:
            df (DataFrame): DataFrame chứa dữ liệu đã tiền xử lý.
            
        Trả về:
            tuple: (X, y) - X là đặc trưng đầu vào, y là nhãn đầu ra.
        """
        # Tạo bản sao của các cột đặc trưng đã chọn (tránh thay đổi dữ liệu gốc)
        X = df[self.features].copy()
        
        # Kiểm tra xem cột mục tiêu có trong DataFrame không
        # Trong quá trình dự đoán thực tế, có thể không có cột mục tiêu
        if self.target in df.columns:
            y = df[self.target]  # Lấy cột mục tiêu
        else:
            y = None  # Không có dữ liệu mục tiêu (chỉ đặc trưng)
            
        return X, y

    def save_model(self):
        """
        Lưu mô hình đã huấn luyện và bộ chuẩn hóa vào file.
        Điều này giúp tái sử dụng mô hình mà không cần huấn luyện lại.
        Joblib được sử dụng để lưu trữ hiệu quả với các đối tượng Python lớn.
        """
        # Tạo thư mục chứa mô hình nếu chưa tồn tại
        # exist_ok=True: không báo lỗi nếu thư mục đã tồn tại
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Lưu mô hình và bộ chuẩn hóa sử dụng joblib
        # Joblib tối ưu hóa cho việc lưu trữ các mảng NumPy lớn
        joblib.dump(self.model, self.model_path)  # Lưu mô hình
        joblib.dump(self.scaler, self.scaler_path)  # Lưu bộ chuẩn hóa
        
    def load_model(self):
        """
        Tải mô hình đã huấn luyện và bộ chuẩn hóa từ file.
        Điều này giúp khôi phục mô hình mà không cần huấn luyện lại.
        
        Trả về:
            bool: True nếu tải thành công, False nếu không tìm thấy file.
        """
        # Kiểm tra xem file mô hình và bộ chuẩn hóa có tồn tại không
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            # Tải mô hình và bộ chuẩn hóa từ file
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.trained = True  # Đánh dấu mô hình đã được huấn luyện
            return True
        return False  # Không tìm thấy file mô hình hoặc bộ chuẩn hóa

    def train(self, data_path):
        """
        Huấn luyện mô hình hoặc tải mô hình đã huấn luyện trước đó.
        Điều này tối ưu hóa quá trình bằng cách ưu tiên sử dụng mô hình đã huấn luyện.
        
        Tham số:
            data_path (str): Đường dẫn đến file dữ liệu CSV.
            
        Trả về:
            float: Điểm đánh giá mô hình trên tập kiểm tra (R² score).
        """
        # Thử tải mô hình đã huấn luyện trước
        if self.load_model():
            print("Đã tải mô hình đã huấn luyện từ đĩa")
            # Vẫn cần tải dữ liệu để tính toán điểm đánh giá
            df = self.load_and_preprocess_data(data_path)
            X, y = self.prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_test_scaled = self.scaler.transform(X_test)  # Chuẩn hóa dữ liệu kiểm tra
            test_score = self.model.score(X_test_scaled, y_test)  # Tính điểm R²
            return test_score
            
        # Nếu không có mô hình đã huấn luyện, huấn luyện mô hình mới
        df = self.load_and_preprocess_data(data_path)  # Tải và tiền xử lý dữ liệu
        X, y = self.prepare_features(df)  # Tách đặc trưng và mục tiêu
        
        # Chia dữ liệu thành tập huấn luyện và kiểm tra (80% - 20%)
        # test_size=0.2: 20% dữ liệu dùng để kiểm tra, 80% để huấn luyện
        # random_state=42: đảm bảo kết quả nhất quán khi chạy nhiều lần
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Chuẩn hóa đặc trưng đầu vào để các đặc trưng có cùng thang đo
        # fit_transform: học từ dữ liệu huấn luyện và áp dụng biến đổi
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Huấn luyện mô hình RandomForest trên dữ liệu đã chuẩn hóa
        self.model.fit(X_train_scaled, y_train)
        self.trained = True  # Đánh dấu mô hình đã được huấn luyện
        
        # Lưu mô hình đã huấn luyện để sử dụng lại sau này
        self.save_model()
        
        # Tính điểm đánh giá trên tập kiểm tra (R² score - hệ số xác định)
        # Biến đổi tập kiểm tra với cùng bộ chuẩn hóa đã học từ tập huấn luyện
        X_test_scaled = self.scaler.transform(X_test)
        test_score = self.model.score(X_test_scaled, y_test)
        return test_score

    def predict(self, features_dict):
        """
        Dự đoán lượng phát thải CO2 dựa trên các đặc trưng đầu vào.
        Áp dụng mô hình đã huấn luyện để dự đoán kết quả cho dữ liệu mới.
        
        Tham số:
            features_dict (dict): Từ điển chứa các đặc trưng đầu vào
                                 của xe cần dự đoán phát thải.
            
        Trả về:
            float: Lượng phát thải CO2 dự đoán (g/km).
            
        Raises:
            ValueError: Nếu mô hình chưa được huấn luyện.
        """
        # Kiểm tra mô hình đã được huấn luyện chưa
        if not self.trained:
            raise ValueError("Mô hình cần phải được huấn luyện trước!")
            
        # Chuyển đổi từ điển đặc trưng thành DataFrame với một dòng
        # Để phù hợp với định dạng mà mô hình có thể xử lý
        features_df = pd.DataFrame([features_dict])
        
        # Chuẩn hóa đặc trưng đầu vào bằng bộ chuẩn hóa đã học
        features_scaled = self.scaler.transform(features_df)
        
        # Thực hiện dự đoán sử dụng mô hình đã huấn luyện
        # [0]: lấy phần tử đầu tiên vì kết quả là mảng với một phần tử
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction

    def get_feature_importance(self):
        """
        Lấy độ quan trọng của từng đặc trưng trong mô hình.
        RandomForest cung cấp thông tin về mức độ ảnh hưởng của từng đặc trưng
        đến kết quả dự đoán, giúp hiểu rõ hơn về mô hình.
        
        Trả về:
            dict: Từ điển với khóa là tên đặc trưng và giá trị là mức độ quan trọng.
            
        Raises:
            ValueError: Nếu mô hình chưa được huấn luyện.
        """
        # Kiểm tra mô hình đã được huấn luyện chưa
        if not self.trained:
            raise ValueError("Mô hình cần phải được huấn luyện trước!")
            
        # Tạo từ điển chứa tên đặc trưng và mức độ quan trọng tương ứng
        # feature_importances_: thuộc tính của RandomForestRegressor
        importance_dict = dict(zip(self.features, self.model.feature_importances_))
        return importance_dict 