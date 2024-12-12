import base64
import os
import socketio
import eventlet
import eventlet.wsgi
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, render_template
from keras.models import load_model
import cv2
from keras.losses import MeanSquaredError

# Create a Socket.IO server
sio = socketio.Server()

# Flask application
app = Flask(__name__)

# Path to the trained model
MODEL_PATH = r'C:\Users\RAHUL GOSWAMI\Documents\NCI\Practicum\Practicum Part 2_ Project\model\pilotnet_model.h5'

# Load the model with explicit loss registration
model = load_model(MODEL_PATH, custom_objects={"mse": MeanSquaredError()})

# Preprocessing function for images
def preprocess(image):
    # Crop the image (remove the sky and car hood)
    image = image[60:135, :, :]
    # Resize to the input shape expected by the model (66x200)
    image = cv2.resize(image, (200, 66))
    # Convert the image to YUV color space (as used in NVIDIA PilotNet)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    return image

# Global variable to store the previous steering angle
previous_steering_angle = 0.0

@sio.on('telemetry')
def telemetry(sid, data):
    global previous_steering_angle

    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the car's camera
        img_string = data["image"]
        # Decode the image
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        # Convert the image to a numpy array
        image_array = np.asarray(image)
        # Preprocess the image
        processed_image = preprocess(image_array)
        # Expand dimensions to match the input shape (batch size of 1)
        processed_image = np.expand_dims(processed_image, axis=0)

        # Predict the steering angle using the model
        predicted_steering_angle = float(model.predict(processed_image, batch_size=1))

        # Apply a low-pass filter for steering smoothness
        smoothed_steering_angle = 0.8 * previous_steering_angle + 0.2 * predicted_steering_angle
        previous_steering_angle = smoothed_steering_angle

        # Stop the vehicle if it tries to take a left turn steeper than -7 degrees
        if smoothed_steering_angle < -7.0:  # -7 degrees is the threshold
            print(f"Steering angle {smoothed_steering_angle} exceeds -7 degrees. Stopping the vehicle.")
            throttle = 0.0  # Stop the vehicle
        else:
            # Dynamically adjust throttle to maintain speed around 9-10 mph
            target_speed = 9.5
            if speed < target_speed:
                throttle = 0.3  # Speed up if below target speed
            else:
                throttle = 0.1  # Maintain low throttle

        print(f"Steering Angle: {smoothed_steering_angle}, Throttle: {throttle}, Speed: {speed}")

        # Send the predicted steering angle and throttle back to the simulator
        send_control(smoothed_steering_angle, throttle)

    else:
        # No data received
        sio.emit('manual', data={}, skip_sid=True)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__()
        },
        skip_sid=True
    )

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Wrap Flask application with Socket.IO
    app = socketio.Middleware(sio, app)

    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)
