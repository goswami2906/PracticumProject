README

Advancements in Steering Angle Prediction: Deep Learning Approaches for Self-Driving Cars

Project Overview

This project focuses on developing, training, and evaluating three deep learning models for predicting steering angles in self-driving cars. Using the Udacity Self-Driving Car Simulator, we collected data to train the following models:

PilotNet Model: A CNN-based architecture for end-to-end learning.

Enhanced Nvidia Model: An improved version of PilotNet with added layers and enhancements.

Hybrid CNN-RNN Model: Combines convolutional layers with LSTMs to account for spatial and temporal dependencies.

The primary goal is to determine the most effective model for steering angle prediction based on performance metrics such as MAE and RMSE.

Key Features

Dataset: Images from three camera perspectives (center, left, right) paired with metadata (steering angle, throttle, brake, speed).

Preprocessing: Includes resizing, cropping, color space conversion (YUV), and normalization.

Augmentation: Techniques such as brightness adjustments, flipping, zoom, and Gaussian blur to improve model robustness.

Evaluation Metrics: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

Requirements

Hardware:

Processor: Intel Core i5 (4 cores)

GPU: NVIDIA RTX 3050

RAM: 16 GB

Storage: 512 GB SSD

Software:

Operating System: Windows 10

Python Version: 3.7.12 (Anaconda Environment)

IDE: Visual Studio Code (VSCode)

Libraries:

Install the required libraries using the provided requirements.txt file:

pip install -r requirements.txt

Key libraries include:

opencv-contrib-python

numpy

matplotlib

tensorflow==2.17.0

keras==3.6.0

Flask, Eventlet

python-socketio, python-engineio

scikit-learn

pandas

Dataset

The dataset consists of:

Images from the Udacity Simulator (center, left, right cameras).

Metadata (steering angle, throttle, brake, speed) in driving_log.csv.

Preprocessing Steps:

Crop irrelevant portions (sky, car hood).

Resize to 66x200 pixels.

Normalize pixel values to [0, 1].

Augment images to improve generalization.

Split Ratio: 80% training, 20% validation.

Model Architectures

PilotNet: A lightweight CNN with five convolutional layers followed by fully connected layers.

Enhanced Nvidia: Adds more convolutional layers and advanced features such as batch normalization and dropout.

Hybrid CNN-RNN: Combines convolutional layers for spatial feature extraction with LSTM layers for temporal pattern recognition.

How to Run

Step 1: Set Up the Environment

conda activate car
pip install -r requirements.txt

Step 2: Launch the Udacity Simulator

Download and install the simulator.

Open the simulator in Autonomous Mode.

Step 3: Run the Drive Script

Navigate to the directory containing the drive.py script and the trained model (model.h5).

Run the script:

python drive.py

Step 4: Observe Vehicle Behavior

The vehicle's steering will be controlled by the model in real-time.

Results

PilotNet: Lowest inference time, suitable for simple driving tasks.

Enhanced Nvidia: Best overall performance (MAE: 0.010, RMSE: 0.020).

Hybrid CNN-RNN: Effective for dynamic scenarios but computationally intensive.

Limitations

Hardware Constraints: The CARLA simulator could not be used due to system limitations.

Dataset: Relies on simulated data, which may not fully replicate real-world conditions.

Future Work

Validate models on real-world driving datasets.

Optimize for real-time deployment through model quantization and pruning.

Incorporate sensor fusion (LIDAR, GPS) for enhanced decision-making.

Acknowledgments

This project was supervised by Arundev Vamadevan at the National College of Ireland.

For further assistance, refer to the documentation or contact Rahul Goswami (X23167572).