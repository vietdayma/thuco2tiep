from flask import Flask, request, jsonify
from flask_cors import CORS
from controllers.emission_controller import EmissionController
import logging
import time
import os
import traceback
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import threading
import json
from functools import lru_cache

# Cấu hình logging - Thiết lập hệ thống ghi log để theo dõi hoạt động của server
logging.basicConfig(
    level=logging.INFO,  # Mức độ log: INFO - chỉ ghi lại thông tin quan trọng
    format='%(asctime)s - %(levelname)s - %(message)s'  # Định dạng: thời gian - mức độ - nội dung
)
logger = logging.getLogger(__name__)  # Tạo đối tượng logger cho module hiện tại

app = Flask(__name__)  # Khởi tạo ứng dụng Flask
CORS(app)  # Cho phép truy cập API từ các nguồn khác nhau (Cross-Origin Resource Sharing)

# Giảm rate limiting để cho phép nhiều request hơn - Cơ chế hạn chế số lượng request trong một khoảng thời gian
limiter = Limiter(
    get_remote_address,  # Sử dụng IP của client để theo dõi và giới hạn request
    app=app,
    default_limits=["200 per minute", "20 per second"],  # Giới hạn mặc định: 200 request/phút, 20 request/giây
    storage_uri="memory://",  # Lưu trữ thông tin giới hạn trong bộ nhớ
    strategy="fixed-window"  # Sử dụng chiến lược cửa sổ cố định để giới hạn
)

# Các biến toàn cục để quản lý trạng thái của server và mô hình
controller = None  # Đối tượng controller chính để xử lý dự đoán
model_initialized = False  # Cờ đánh dấu mô hình đã được khởi tạo hay chưa
initialization_in_progress = False  # Cờ đánh dấu quá trình khởi tạo đang diễn ra
prediction_lock = threading.RLock()  # Khóa đồng bộ hóa cho các thao tác dự đoán

# Cache đơn giản cho kết quả dự đoán - Giúp giảm thời gian xử lý cho các request lặp lại
prediction_cache = {}
MAX_CACHE_SIZE = 500  # Kích thước tối đa của cache - 500 kết quả

# Chuẩn bị cache function với lru_cache - Decorator để tự động lưu cache kết quả trả về
@lru_cache(maxsize=1000)
def cached_predict(engine_size, cylinders, fuel_consumption, horsepower, weight, year):
    """
    Hàm dự đoán có lưu cache - Sử dụng lru_cache để tối ưu hóa hiệu năng
    
    Lưu kết quả dự đoán dựa trên các tham số đầu vào, giúp trả về kết quả ngay lập tức
    nếu cùng một bộ tham số được sử dụng lại.
    
    Parameters:
        engine_size: Kích thước động cơ (L)
        cylinders: Số xi-lanh
        fuel_consumption: Mức tiêu thụ nhiên liệu (L/100km)
        horsepower: Công suất động cơ
        weight: Trọng lượng xe (kg)
        year: Năm sản xuất
        
    Returns:
        float: Giá trị dự đoán lượng khí thải CO2 (g/km)
    """
    features = {
        'Engine Size(L)': float(engine_size),
        'Cylinders': int(cylinders),
        'Fuel Consumption Comb (L/100 km)': float(fuel_consumption),
        'Horsepower': float(horsepower),
        'Weight (kg)': float(weight),
        'Year': int(year)
    }
    global controller
    return float(controller.predict_emission(features))

def initialize_model():
    """
    Khởi tạo mô hình nếu chưa được khởi tạo
    
    Đảm bảo mô hình chỉ được khởi tạo một lần duy nhất, tránh khởi tạo lại
    khi có nhiều request đồng thời. Hàm kiểm tra và xử lý các trường hợp:
    - Đang trong quá trình khởi tạo
    - Mô hình đã được khởi tạo
    - Cần khởi tạo mô hình mới
    
    Returns:
        bool: True nếu mô hình đã khởi tạo thành công, False nếu có lỗi
    """
    global controller, model_initialized, initialization_in_progress
    
    # Kiểm tra xem quá trình khởi tạo đã đang diễn ra hay chưa
    if initialization_in_progress:
        logger.info("Model initialization already in progress, waiting...")
        return False
        
    if not model_initialized:
        try:
            initialization_in_progress = True  # Đánh dấu bắt đầu quá trình khởi tạo
            logger.info("Starting model initialization...")
            start_time = time.perf_counter()  # Bắt đầu đo thời gian
            
            controller = EmissionController()  # Tạo đối tượng controller mới
            # Sử dụng đường dẫn tuyệt đối đến file dữ liệu
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, "co2 Emissions.csv")
            
            # Kiểm tra sự tồn tại của file dữ liệu
            if not os.path.exists(csv_path):
                logger.error(f"Could not find the file: {csv_path}")
                initialization_in_progress = False
                return False
                
            # Khởi tạo mô hình với dữ liệu từ file
            test_score = controller.initialize_model(csv_path)
            initialization_time = time.perf_counter() - start_time
            logger.info(f"Model initialized with test score: {test_score:.3f} in {initialization_time:.2f} seconds")
            
            # Đánh dấu hoàn thành khởi tạo
            model_initialized = True
            initialization_in_progress = False
            return True
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            logger.error(traceback.format_exc())
            initialization_in_progress = False
            return False
    return True

# Middleware để khởi tạo mô hình trước khi xử lý request
@app.before_request
def setup():
    """
    Khởi tạo mô hình trước khi xử lý các request
    
    Được gọi tự động trước mỗi request đến server.
    Đảm bảo mô hình đã sẵn sàng trước khi xử lý các yêu cầu dự đoán.
    """
    global model_initialized
    if not model_initialized:
        if not initialize_model():
            logger.error("Failed to initialize model")

def get_cache_key(data):
    """
    Tạo khóa cache từ dữ liệu đầu vào
    
    Chuyển đổi dữ liệu đầu vào thành chuỗi khóa duy nhất để lưu và truy xuất trong cache.
    
    Parameters:
        data: Dictionary chứa các thông số của xe
        
    Returns:
        str: Chuỗi khóa duy nhất đại diện cho bộ thông số, hoặc None nếu có lỗi
    """
    try:
        key_parts = []
        for field in ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 
                     'Horsepower', 'Weight (kg)', 'Year']:
            if field in data:
                key_parts.append(f"{field}:{data[field]}")
        return "|".join(key_parts)
    except:
        # Xử lý dự phòng trong trường hợp có lỗi
        return None

@app.route('/predict', methods=['POST'])
@limiter.limit("100 per second")  # Tăng giới hạn lên 100 request/giây cho endpoint này
def predict():
    """
    Endpoint chính của API để dự đoán lượng khí thải CO2
    
    Nhận thông số xe dưới dạng JSON, thực hiện dự đoán và trả về kết quả.
    Bao gồm các cơ chế:
    - Kiểm tra và xác thực đầu vào
    - Kiểm tra cache trước khi dự đoán
    - Xử lý lỗi và trả về giá trị mặc định nếu cần
    - Đo thời gian xử lý
    
    Returns:
        JSON: Kết quả dự đoán và thông tin liên quan
    """
    # Bắt đầu đo thời gian xử lý
    start_time = time.perf_counter()
    
    # Kiểm tra định dạng dữ liệu đầu vào
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON', 'status': 'error'}), 400
    
    try:    
        # Đảm bảo mô hình đã được khởi tạo - nếu không, trả về giá trị dự phòng
        if not model_initialized:
            if not initialize_model():
                logger.warning("Model not initialized, returning fallback value")
                return jsonify({
                    'prediction': 200.0,  # Giá trị mặc định: 200g/km
                    'process_time_ms': (time.perf_counter() - start_time) * 1000,
                    'status': 'fallback'
                }), 200
        
        # Lấy dữ liệu từ request
        data = request.json
        
        # Danh sách các trường bắt buộc
        required_fields = [
            'Engine Size(L)', 'Cylinders', 
            'Fuel Consumption Comb (L/100 km)',
            'Horsepower', 'Weight (kg)', 'Year'
        ]
        
        # Kiểm tra đầy đủ các trường - nếu thiếu, trả về giá trị dự phòng
        if not all(field in data for field in required_fields):
            logger.warning(f"Missing required fields, returning fallback. Received: {data}")
            return jsonify({
                'prediction': 200.0,  # Giá trị mặc định
                'process_time_ms': (time.perf_counter() - start_time) * 1000,
                'status': 'fallback',
                'message': 'Missing fields'
            }), 200
        
        # Kiểm tra cache trước khi thực hiện dự đoán - tối ưu hóa hiệu năng
        cache_key = get_cache_key(data)
        if cache_key and cache_key in prediction_cache:
            cached_result = prediction_cache[cache_key]
            process_time = (time.perf_counter() - start_time) * 1000
            return jsonify({
                'prediction': float(cached_result),
                'process_time_ms': process_time,
                'cached': True,
                'status': 'success'
            }), 200
        
        # Ghi log request (chỉ log 10% request để giảm tải I/O)
        if start_time % 10 < 1:
            logger.info(f"Received prediction request: {data}")
        
        # Thực hiện dự đoán với cache lru
        try:
            with prediction_lock:  # Khóa đồng bộ hóa để đảm bảo an toàn thread
                prediction = cached_predict(
                    data['Engine Size(L)'],
                    data['Cylinders'],
                    data['Fuel Consumption Comb (L/100 km)'],
                    data['Horsepower'],
                    data['Weight (kg)'],
                    data['Year']
                )
                
                # Lưu kết quả vào cache
                if cache_key and len(prediction_cache) < MAX_CACHE_SIZE:
                    prediction_cache[cache_key] = prediction
        except Exception as inner_e:
            # Xử lý lỗi khi dự đoán - trả về giá trị dự phòng
            logger.error(f"Error making prediction: {str(inner_e)}")
            return jsonify({
                'prediction': 200.0,
                'process_time_ms': (time.perf_counter() - start_time) * 1000,
                'status': 'fallback',
                'message': 'Prediction error'
            }), 200
        
        # Tính toán thời gian xử lý
        process_time = (time.perf_counter() - start_time) * 1000
        
        # Ghi log kết quả (chỉ log 10% request)
        if start_time % 10 < 1:
            logger.info(f"Prediction: {prediction:.2f}, Processing time: {process_time:.2f}ms")
        
        # Trả về kết quả dự đoán thành công
        return jsonify({
            'prediction': float(prediction),
            'process_time_ms': process_time,
            'cached': False,
            'status': 'success'
        }), 200
        
    except Exception as e:
        # Xử lý các lỗi không mong muốn
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Luôn trả về status 200 với giá trị dự phòng để cải thiện trải nghiệm người dùng
        return jsonify({
            'prediction': 200.0,  # Giá trị dự phòng
            'process_time_ms': (time.perf_counter() - start_time) * 1000,
            'status': 'error',
            'message': str(e)
        }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint kiểm tra trạng thái hoạt động của API
    
    Trả về thông tin về trạng thái khởi tạo và hoạt động của mô hình.
    Được sử dụng bởi hệ thống giám sát và frontend để kiểm tra kết nối.
    
    Returns:
        JSON: Thông tin trạng thái API và model
    """
    try:
        if not model_initialized:
            if initialization_in_progress:
                return jsonify({
                    "status": "initializing",
                    "message": "Model initialization in progress"
                }), 200  # Trả về 200 ngay cả khi đang khởi tạo
            else:
                return jsonify({
                    "status": "initializing",
                    "message": "Model not yet initialized"
                }), 200
        return jsonify({
            "status": "healthy",
            "message": "API is running and model is initialized",
            "stats": {
                "cache_size": len(prediction_cache)  # Thống kê kích thước cache hiện tại
            }
        }), 200
    except Exception as e:
        # Xử lý lỗi và vẫn trả về 200 để tránh báo lỗi giả
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 200

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """
    Endpoint xóa cache dự đoán
    
    Xóa toàn bộ cache để giải phóng bộ nhớ hoặc cập nhật kết quả dự đoán
    khi mô hình được cập nhật.
    
    Returns:
        JSON: Kết quả thực hiện xóa cache
    """
    global prediction_cache
    try:
        old_size = len(prediction_cache)  # Lưu kích thước cache cũ
        prediction_cache = {}  # Xóa dictionary cache
        cached_predict.cache_clear()  # Xóa lru_cache
        return jsonify({
            "status": "success",
            "message": f"Cache cleared. {old_size} entries removed."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 200

@app.route('/fallback', methods=['POST'])
def fallback_prediction():
    """
    Endpoint dự phòng luôn trả về giá trị cố định
    
    Được sử dụng trong trường hợp cần kiểm tra hiệu năng mạng mà không quan tâm
    đến logic xử lý thực tế. Cũng hữu ích khi mô hình chính gặp sự cố.
    
    Returns:
        JSON: Kết quả dự đoán giả với giá trị mặc định
    """
    return jsonify({
        'prediction': 200.0,  # Giá trị dự phòng cố định
        'process_time_ms': 5.0,  # Thời gian xử lý giả
        'status': 'fallback',
        'cached': False
    }), 200

# Chỉ thực thi khi chạy trực tiếp file này (không khi được import)
if __name__ == '__main__':
    # Lấy cổng từ biến môi trường (Render sets this)
    port = int(os.environ.get('PORT', 10000))
    
    # Khởi tạo mô hình khi khởi động
    logger.info(f"Starting server on port {port}...")
    logger.info("Initializing model at startup...")
    
    try:
        if not initialize_model():
            logger.error("Failed to initialize model at startup")
            # Vẫn tiếp tục chạy server ngay cả khi khởi tạo thất bại - sẽ sử dụng giá trị dự phòng
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Cấu hình máy chủ: sử dụng gunicorn trong môi trường sản xuất, Flask dev server cho phát triển local
    if os.environ.get('RENDER'):
        # Render sẽ chạy với gunicorn
        app.run(host='0.0.0.0', port=port)
    else:
        # Phát triển local với chế độ debug
        app.run(host='0.0.0.0', port=port, debug=True) 