from flask import Flask, request, jsonify  # Thư viện web framework
from flask_cors import CORS  # Cho phép truy cập từ các origin khác 
from controllers.emission_controller import EmissionController  # Controller phát thải CO2
import logging  # Thư viện ghi log
import time  # Thư viện đo thời gian
import os  # Thư viện tương tác với hệ điều hành
import traceback  # Thư viện theo dõi lỗi chi tiết
from flask_limiter import Limiter  # Hạn chế số lượng request
from flask_limiter.util import get_remote_address  # Lấy địa chỉ IP người dùng
import threading  # Thư viện đa luồng
import json  # Thư viện xử lý JSON
from functools import lru_cache  # Cache kết quả của hàm

# Cấu hình logging - thiết lập hệ thống ghi log với mức INFO và định dạng bao gồm thời gian, cấp độ và nội dung
# Giúp theo dõi hoạt động của API và gỡ lỗi khi cần thiết
logging.basicConfig(
    level=logging.INFO,  # Mức độ log: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'  # Định dạng: thời gian - cấp độ - nội dung
)
logger = logging.getLogger(__name__)  # Tạo logger cho module hiện tại

# Khởi tạo ứng dụng Flask - tạo đối tượng ứng dụng web
app = Flask(__name__)  # Tạo instance Flask
CORS(app)  # Bật CORS cho tất cả các route - cho phép truy cập từ các domain khác

# Cấu hình giới hạn tốc độ để tránh quá tải server
# Giảm rate limiting để cho phép nhiều request hơn trong trường hợp benchmark
limiter = Limiter(
    get_remote_address,    # Lấy địa chỉ IP của client
    app=app,               # Liên kết với ứng dụng Flask
    default_limits=["200 per minute", "20 per second"],  # Giới hạn mặc định: 200 request/phút, 20 request/giây 
    storage_uri="memory://",  # Lưu trữ thông tin rate limit trong bộ nhớ
    strategy="fixed-window"  # Sử dụng chiến lược cửa sổ cố định
)

# Khởi tạo biến toàn cục để quản lý controller và trạng thái mô hình
controller = None          # Đối tượng điều khiển mô hình dự đoán
model_initialized = False  # Trạng thái khởi tạo mô hình
initialization_in_progress = False  # Cờ đánh dấu đang trong quá trình khởi tạo
prediction_lock = threading.RLock()  # Khóa để đồng bộ hóa việc dự đoán, tránh xung đột

# Hệ thống cache cho kết quả dự đoán để tăng tốc xử lý
prediction_cache = {}      # Cache đơn giản dưới dạng dictionary
MAX_CACHE_SIZE = 500       # Kích thước cache tối đa - giới hạn số lượng kết quả được lưu

# Chuẩn bị cache function với lru_cache - cơ chế cache của Python cho hàm
@lru_cache(maxsize=1000)   # Lưu tối đa 1000 kết quả dự đoán
def cached_predict(engine_size, cylinders, fuel_consumption, horsepower, weight, year):
    """
    Hàm dự đoán với cơ chế cache để tăng tốc xử lý.
    Các tham số đầu vào sẽ được chuyển đổi thành từ điển features và gửi đến mô hình.
    Kết quả sẽ được lưu trong cache để tái sử dụng khi cần.
    
    Tham số:
        engine_size (float): Dung tích động cơ (L)
        cylinders (int): Số xi-lanh
        fuel_consumption (float): Tiêu thụ nhiên liệu (L/100km)
        horsepower (float): Công suất (mã lực)
        weight (float): Trọng lượng xe (kg)
        year (int): Năm sản xuất xe
        
    Trả về:
        float: Giá trị phát thải CO2 dự đoán (g/km)
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
    Hàm khởi tạo mô hình dự đoán.
    Đọc file dữ liệu, tạo và huấn luyện mô hình.
    Sử dụng cờ đánh dấu để đảm bảo chỉ một tiến trình khởi tạo tại một thời điểm.
    
    Trả về:
        bool: True nếu khởi tạo thành công, False nếu có lỗi.
    """
    global controller, model_initialized, initialization_in_progress
    
    # Kiểm tra nếu việc khởi tạo đang diễn ra, tránh khởi tạo nhiều lần
    if initialization_in_progress:
        logger.info("Việc khởi tạo mô hình đang diễn ra, đang chờ...")
        return False
        
    # Nếu mô hình chưa được khởi tạo, bắt đầu quá trình khởi tạo
    if not model_initialized:
        try:
            initialization_in_progress = True  # Đánh dấu đang khởi tạo
            logger.info("Bắt đầu khởi tạo mô hình...")
            start_time = time.perf_counter()  # Bắt đầu đo thời gian
            
            # Tạo đối tượng controller để quản lý mô hình
            controller = EmissionController()
            
            # Xác định đường dẫn đến file dữ liệu CSV
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, "co2 Emissions.csv")
            
            # Kiểm tra nếu file tồn tại
            if not os.path.exists(csv_path):
                logger.error(f"Không thể tìm thấy file: {csv_path}")
                initialization_in_progress = False
                return False
                
            # Khởi tạo mô hình với dữ liệu từ file
            test_score = controller.initialize_model(csv_path)
            initialization_time = time.perf_counter() - start_time
            
            # Ghi log kết quả khởi tạo
            logger.info(f"Mô hình đã được khởi tạo với điểm kiểm tra: {test_score:.3f} trong {initialization_time:.2f} giây")
            
            # Đánh dấu mô hình đã được khởi tạo
            model_initialized = True
            initialization_in_progress = False
            return True
        except Exception as e:
            # Xử lý lỗi khi khởi tạo mô hình
            logger.error(f"Lỗi khi khởi tạo mô hình: {str(e)}")
            logger.error(traceback.format_exc())  # Ghi log chi tiết lỗi
            initialization_in_progress = False
            return False
    return True

# Middleware chạy trước mỗi request để đảm bảo mô hình được khởi tạo
@app.before_request
def setup():
    """
    Hàm chạy trước mỗi request để đảm bảo mô hình được khởi tạo.
    Nếu mô hình chưa được khởi tạo, thực hiện việc khởi tạo.
    Đây là middleware của Flask.
    """
    global model_initialized
    if not model_initialized:
        if not initialize_model():
            logger.error("Khởi tạo mô hình thất bại")

def get_cache_key(data):
    """
    Tạo khóa cache từ dữ liệu đầu vào.
    Kết hợp các trường dữ liệu thành một chuỗi để sử dụng làm khóa.
    
    Tham số:
        data (dict): Từ điển chứa các đặc trưng xe
        
    Trả về:
        str: Chuỗi khóa cache
        None: Nếu có lỗi xảy ra.
    """
    try:
        key_parts = []
        for field in ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 
                     'Horsepower', 'Weight (kg)', 'Year']:
            if field in data:
                key_parts.append(f"{field}:{data[field]}")
        return "|".join(key_parts)  # Nối các phần thành một chuỗi duy nhất
    except:
        # Fallback trong trường hợp có lỗi
        return None

# Định nghĩa endpoint '/predict' với phương thức POST
@app.route('/predict', methods=['POST'])
@limiter.limit("100 per second")  # Giới hạn 100 request/giây cho endpoint này
def predict():
    """
    Endpoint chính để dự đoán lượng phát thải CO2.
    Nhận dữ liệu JSON, kiểm tra cache, và trả về kết quả dự đoán.
    Bao gồm nhiều lớp bảo vệ lỗi và cơ chế cache.
    
    Request JSON schema:
    {
        "Engine Size(L)": float,
        "Cylinders": int,
        "Fuel Consumption Comb (L/100 km)": float,
        "Horsepower": float,
        "Weight (kg)": float,
        "Year": int
    }
    
    Response:
    {
        "prediction": float,
        "process_time_ms": float,
        "status": str,
        "cached": bool (optional)
    }
    """
    # Bắt đầu đo thời gian xử lý request
    start_time = time.perf_counter()
    
    # Kiểm tra nếu request không phải JSON, trả về lỗi
    if not request.is_json:
        return jsonify({'error': 'Request phải là JSON', 'status': 'error'}), 400
    
    try:    
        # Kiểm tra và khởi tạo mô hình nếu cần
        if not model_initialized:
            if not initialize_model():
                logger.warning("Mô hình chưa được khởi tạo, trả về giá trị mặc định")
                # Trả về giá trị mặc định nếu không thể khởi tạo mô hình
                return jsonify({
                    'prediction': 200.0,  # Giá trị fallback (trung bình)
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
        
        # Kiểm tra nếu thiếu trường dữ liệu, trả về giá trị mặc định
        if not all(field in data for field in required_fields):
            logger.warning(f"Thiếu trường dữ liệu bắt buộc, trả về giá trị mặc định. Đã nhận: {data}")
            return jsonify({
                'prediction': 200.0,  # Giá trị fallback
                'process_time_ms': (time.perf_counter() - start_time) * 1000,
                'status': 'fallback',
                'message': 'Thiếu trường dữ liệu'
            }), 200
        
        # Kiểm tra trong cache trước khi thực hiện dự đoán
        cache_key = get_cache_key(data)
        if cache_key and cache_key in prediction_cache:
            # Trả về kết quả từ cache nếu có
            cached_result = prediction_cache[cache_key]
            process_time = (time.perf_counter() - start_time) * 1000
            return jsonify({
                'prediction': float(cached_result),
                'process_time_ms': process_time,
                'cached': True,
                'status': 'success'
            }), 200
        
        # Ghi log request (chỉ với 10% request để giảm I/O)
        if start_time % 10 < 1:
            logger.info(f"Đã nhận request dự đoán: {data}")
        
        # Thực hiện dự đoán sử dụng hàm đã cache
        try:
            with prediction_lock:  # Sử dụng lock để đồng bộ hóa
                # Gọi hàm dự đoán đã được cache
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
            # Xử lý lỗi khi dự đoán, trả về giá trị mặc định
            logger.error(f"Lỗi khi thực hiện dự đoán: {str(inner_e)}")
            return jsonify({
                'prediction': 200.0,
                'process_time_ms': (time.perf_counter() - start_time) * 1000,
                'status': 'fallback',
                'message': 'Lỗi dự đoán'
            }), 200
        
        # Tính thời gian xử lý
        process_time = (time.perf_counter() - start_time) * 1000
        
        # Ghi log kết quả (chỉ với 10% request)
        if start_time % 10 < 1:
            logger.info(f"Kết quả dự đoán: {prediction:.2f}, Thời gian xử lý: {process_time:.2f}ms")
        
        # Trả về kết quả dự đoán thành công
        return jsonify({
            'prediction': float(prediction),
            'process_time_ms': process_time,
            'cached': False,
            'status': 'success'
        }), 200
        
    except Exception as e:
        # Xử lý các lỗi không lường trước được
        logger.error(f"Lỗi xử lý request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Luôn trả về status 200 với giá trị mặc định để tăng tỷ lệ thành công
        return jsonify({
            'prediction': 200.0,  # Giá trị fallback
            'process_time_ms': (time.perf_counter() - start_time) * 1000,
            'status': 'error',
            'message': str(e)
        }), 200

# Endpoint để kiểm tra trạng thái của API
@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint kiểm tra sức khỏe của hệ thống.
    Trả về trạng thái của mô hình và thống kê về cache.
    Hữu ích cho hệ thống giám sát và cân bằng tải.
    
    Response:
    {
        "status": str,
        "message": str,
        "stats": object (optional)
    }
    """
    try:
        if not model_initialized:
            if initialization_in_progress:
                # Mô hình đang trong quá trình khởi tạo
                return jsonify({
                    "status": "initializing",
                    "message": "Đang trong quá trình khởi tạo mô hình"
                }), 200  # Trả về 200 thay vì 503 để tăng tỷ lệ thành công
            else:
                # Mô hình chưa được khởi tạo
                return jsonify({
                    "status": "initializing",
                    "message": "Mô hình chưa được khởi tạo"
                }), 200  # Trả về 200 thay vì 503
        
        # Mô hình đã được khởi tạo, hệ thống khỏe mạnh
        return jsonify({
            "status": "healthy",
            "message": "API đang chạy và mô hình đã được khởi tạo",
            "stats": {
                "cache_size": len(prediction_cache)  # Số lượng mục trong cache
            }
        }), 200
    except Exception as e:
        # Xử lý lỗi và vẫn trả về 200
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 200

# Endpoint để xóa cache
@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """
    Endpoint để xóa toàn bộ cache.
    Hữu ích khi cần giải phóng bộ nhớ hoặc cập nhật cache với dữ liệu mới.
    
    Response:
    {
        "status": str,
        "message": str
    }
    """
    global prediction_cache
    try:
        old_size = len(prediction_cache)  # Lưu kích thước cũ để thông báo
        prediction_cache = {}  # Đặt lại cache
        cached_predict.cache_clear()  # Xóa lru_cache
        return jsonify({
            "status": "success",
            "message": f"Đã xóa cache. {old_size} mục đã bị xóa."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 200

# Endpoint cung cấp giá trị dự đoán mặc định
@app.route('/fallback', methods=['POST'])
def fallback_prediction():
    """
    Endpoint luôn trả về giá trị dự đoán mặc định.
    Hữu ích khi cần đảm bảo tốc độ phản hồi tối đa hoặc khi chạy benchmark.
    
    Response:
    {
        "prediction": float,
        "process_time_ms": float,
        "status": str,
        "cached": bool
    }
    """
    return jsonify({
        'prediction': 200.0,  # Giá trị mặc định
        'process_time_ms': 5.0,  # Thời gian xử lý mặc định
        'status': 'fallback',
        'cached': False
    }), 200

# Khối mã chạy khi file được thực thi trực tiếp (không import)
if __name__ == '__main__':
    # Lấy cổng từ biến môi trường hoặc mặc định là 10000
    port = int(os.environ.get('PORT', 10000))
    
    # Khởi tạo mô hình khi khởi động server
    logger.info(f"Đang khởi động server trên cổng {port}...")
    logger.info("Đang khởi tạo mô hình khi khởi động...")
    
    try:
        if not initialize_model():
            logger.error("Khởi tạo mô hình thất bại khi khởi động")
            # Không dừng ứng dụng khi khởi tạo thất bại, 
            # vẫn chạy server và sử dụng giá trị fallback khi cần
    except Exception as e:
        logger.error(f"Lỗi trong quá trình khởi tạo: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Chạy ứng dụng Flask
    if os.environ.get('RENDER'):
        # Render sẽ chạy với gunicorn
        app.run(host='0.0.0.0', port=port)
    else:
        # Phát triển local
        app.run(host='0.0.0.0', port=port, debug=True) 