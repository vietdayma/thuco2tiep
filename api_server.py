from flask import Flask, request, jsonify
from flask_cors import CORS
from controllers.emission_controller import EmissionController
import logging
import time
import os
import traceback

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Khởi tạo controller global
controller = None
model_initialized = False
initialization_in_progress = False

def initialize_model():
    """Initialize the model if not already initialized"""
    global controller, model_initialized, initialization_in_progress
    
    if initialization_in_progress:
        logger.info("Model initialization already in progress, waiting...")
        return False
        
    if not model_initialized:
        try:
            initialization_in_progress = True
            logger.info("Starting model initialization...")
            start_time = time.perf_counter()
            
            controller = EmissionController()
            # Sử dụng đường dẫn tuyệt đối
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, "co2 Emissions.csv")
            
            if not os.path.exists(csv_path):
                logger.error(f"Could not find the file: {csv_path}")
                initialization_in_progress = False
                return False
                
            test_score = controller.initialize_model(csv_path)
            initialization_time = time.perf_counter() - start_time
            logger.info(f"Model initialized with test score: {test_score:.3f} in {initialization_time:.2f} seconds")
            model_initialized = True
            initialization_in_progress = False
            return True
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            logger.error(traceback.format_exc())
            initialization_in_progress = False
            return False
    return True

# Thay thế before_first_request bằng before_request
@app.before_request
def setup():
    """Initialize model before processing requests"""
    global model_initialized
    if not model_initialized:
        if not initialize_model():
            logger.error("Failed to initialize model")

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure model is initialized
    if not model_initialized and not initialize_model():
        return jsonify({'error': 'Model not initialized'}), 500
        
    # Bắt đầu đo thời gian xử lý
    start_time = time.perf_counter()
    
    try:
        # Validate input
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        required_fields = [
            'Engine Size(L)', 'Cylinders', 
            'Fuel Consumption Comb (L/100 km)',
            'Horsepower', 'Weight (kg)', 'Year'
        ]
        
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': f'Missing required fields. Required: {required_fields}'
            }), 400
        
        # Log request
        logger.info(f"Received prediction request: {data}")
        
        # Make prediction
        prediction = controller.predict_emission(data)
        
        # Tính thời gian xử lý
        process_time = (time.perf_counter() - start_time) * 1000
        
        # Log kết quả
        logger.info(f"Prediction: {prediction:.2f}, Processing time: {process_time:.2f}ms")
        
        return jsonify({
            'prediction': float(prediction),
            'process_time_ms': process_time
        })
        
    except Exception as e:
        # Log lỗi
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'process_time_ms': (time.perf_counter() - start_time) * 1000
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if not model_initialized:
        if initialization_in_progress:
            return jsonify({
                "status": "initializing",
                "message": "Model initialization in progress"
            }), 503
        else:
            return jsonify({
                "status": "initializing",
                "message": "Model not yet initialized"
            }), 503
    return jsonify({
        "status": "healthy",
        "message": "API is running and model is initialized"
    }), 200

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 10000))
    
    # Initialize model at startup
    logger.info(f"Starting server on port {port}...")
    logger.info("Initializing model at startup...")
    
    if not initialize_model():
        logger.error("Failed to initialize model at startup")
        exit(1)
        
    # Run with gunicorn in production, Flask dev server for local
    if os.environ.get('RENDER'):
        # Render will run with gunicorn
        app.run(host='0.0.0.0', port=port)
    else:
        # Local development
        app.run(host='0.0.0.0', port=port, debug=True) 