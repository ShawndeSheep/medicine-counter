from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify backend services are running."""
    return jsonify({"message": "Backend Services Running"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
