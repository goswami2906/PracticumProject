import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize SocketIO and Flask app
sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10  # Set a speed limit for throttle calculation

# Define the image preprocessing function
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop image to focus on road
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian Blur
    img = cv2.resize(img, (200, 66))  # Resize to the input shape for the model
    img = img / 255  # Normalize pixel values
    return img

# Handle telemetry data received from the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Extract speed and image from data
        speed = float(data['speed'])
        print(f"Received speed: {speed}")

        # Decode and preprocess the image
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])

        # Predict steering angle and calculate throttle
        steering_angle = float(model.predict(image))
        throttle = 1.0 - speed / speed_limit

        # Debug output for steering angle, throttle, and speed
        print(f"Predicted steering angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}")

        # Send control command to simulator
        send_control(steering_angle, throttle)
    else:
        print("No data received from the simulator")

# Event handler for connecting to the simulator
@sio.on('connect')
def connect(sid, environ):
    print("Connected to simulator")
    send_control(0, 0)  # Send initial command with zero steering and throttle

# Function to send steering and throttle to the simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Run the server with the loaded model
if __name__ == '__main__':
    # Load model
    model_path = r'C:\Users\RAHUL GOSWAMI\Documents\NCI\Practicum\Practicum Part 2_ Project\Project\NVIDIA\nvidia_model.h5'
    model = load_model(model_path, compile=False)
    print("Model loaded successfully.")
    
    # Wrap the app with SocketIO's WSGI server
    app = socketio.WSGIApp(sio, app)
    print("Starting server on port 4567...")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)

