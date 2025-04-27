from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import time
import requests

app = Flask(__name__, template_folder="templates")

# Load YOLO model (optional, depends if you're using it somewhere)
model = YOLO('last.pt')  # If needed for other endpoints

# Store the latest prediction
current_prediction = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feed')
def feed():
    return render_template('feed_result.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/update-prediction', methods=['POST'])
def update_prediction():
    global current_prediction
    data = request.get_json()

    # Update current prediction
    current_prediction = data
    print(f"Received prediction: {data}")

    # Combine sensor data and prediction to generate a notification
    notification = generate_notification(data)
    if notification:
        print(f"Notification for farmer: {notification}")

    return jsonify({"status": "success", "message": "Prediction received successfully."})

def generate_notification(data):
    """
    Generates a farmer-friendly notification based on ripeness + (future) temperature/humidity
    """
    ripeness = data.get('ripeness', '')
    confidence = data.get('confidence', 0)

    if confidence < 0.5:
        return None  # Ignore low-confidence predictions

    if ripeness == "ripe":
        return "Bananas are ripe! Time to harvest soon! 🍌"
    elif ripeness == "overripe":
        return "Bananas are overripe! Harvest immediately to avoid loss! ⚠️"
    elif ripeness == "unripe":
        return "Bananas are still unripe. Please wait a few more days."
    else:
        return None

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
