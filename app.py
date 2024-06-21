from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load the YOLOv8 model
yolo = YOLO('best_weights.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image = request.files['image']

    # Convert the image to OpenCV format
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    results = yolo(img)

    # Convert the results to a JSON format
    output = []
    for result in results:
        for detection in result.detections:
            output.append({
                'class': detection.class_name,
                'confidence': detection.confidence,
                'x': detection.x,
                'y': detection.y,
                'w': detection.w,
                'h': detection.h
            })

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)