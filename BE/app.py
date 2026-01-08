from flask import Flask, jsonify, request
from ai_pipeline import start_pipeline

app = Flask(__name__)


@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify backend services are running."""
    return jsonify({"message": "Backend Services Running"})


@app.route('/count', methods=['POST'])
def count():
    """Endpoint to receive an image and return prediction."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    
    result = start_pipeline(image)
    
    return jsonify({"message": result})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
