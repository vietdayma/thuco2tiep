from flask import Flask, request, jsonify
from controllers.emission_controller import EmissionController
from waitress import serve
import logging
import time

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
controller = EmissionController()

# Initialize model once khi khởi động server
try:
    test_score = controller.initialize_model('co2 Emissions.csv')
    logger.info(f"Model initialized with test score: {test_score:.3f}")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    # Bắt đầu đo thời gian xử lý
    start_time = time.perf_counter()
    
    try:
        # Log request
        logger.info(f"Received prediction request: {request.json}")
        
        # Make prediction
        prediction = controller.predict_emission(request.json)
        
        # Tính thời gian xử lý
        process_time = (time.perf_counter() - start_time) * 1000
        
        # Log kết quả
        logger.info(f"Prediction: {prediction:.2f}, Processing time: {process_time:.2f}ms")
        
        return jsonify({'prediction': float(prediction)})
        
    except Exception as e:
        # Log lỗi
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Log server startup
    logger.info("Starting production server with waitress...")
    logger.info("Server running at http://127.0.0.1:5000")
    serve(app, host='127.0.0.1', port=5000, threads=16) 