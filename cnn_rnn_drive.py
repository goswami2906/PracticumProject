import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from tensorflow.keras.models import load_model
import tensorflow as tf  
import cv2

# Initialize Socket.IO server
sio = socketio.Server()
# Flask application
app = Flask(__name__)

# Variables for saving telemetry data
datetime_format = "%Y_%m_%d_%H_%M_%S"
model = None
prev_image_array = None

# Telemetry event handler
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle, throttle, and speed of the car
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])

        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image = image[60:135, :, :]  # Crop to focus on the road
        image = cv2.resize(image, (200, 66))  # Resize for model input
        image = image / 255.0  # Normalize image

        # Predict steering angle
        steering_angle = float(model.predict(np.array([image]), batch_size=1))

        # Set throttle value for maintaining speed
        throttle = 0.2 if float(speed) < 10 else 0.1

        print(f"Steering Angle: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed}")

        # Send control commands to the car
        send_control(steering_angle, throttle)

    else:
        # Edge case: if telemetry data is not received
        sio.emit('manual', data={}, skip_sid=True)

# Connect event handler
@sio.on('connect')
def connect(sid, environ):
    print("Connected: ", sid)
    send_control(0, 0)

# Function to send control commands to the car
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )

if __name__ == '__main__':
    # Path to the trained model
    model_path = r'C:\Users\RAHUL GOSWAMI\Documents\NCI\Practicum\Practicum Part 2_ Project\model\cnn_rnn_model.h5'

    # Load the trained model
    model = load_model(model_path)

    # Deploy the model using eventlet and Flask
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
