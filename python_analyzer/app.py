from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from face_analyzer import FaceQualityAnalyzer

app = Flask(__name__)
CORS(app)

# Initialize face analyzer
analyzer = FaceQualityAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_face():
    try:
        # Get base64 image from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Analyze face quality
        result = analyzer.analyze_face(image)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)