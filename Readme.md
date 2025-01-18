YOLO Object Detection Project
Overview
This project involves the development and deployment of a YOLO-based object detection model using a custom dataset. The project was undertaken as part of an internship at FLEX Pvt Ltd and aims to automate the detection and identification of objects in real-time video streams.

Project Structure
Files and Directories
data.yaml: Configuration file specifying the paths to the training and validation datasets, number of classes, and class names.
train.py: Script for training the YOLO model using the custom dataset.
check.py: Script for running real-time object detection using the trained YOLO model.
ListofCamera.py: Script for listing available camera devices on the system.
best.pt: The best-performing model weights obtained after training.
README.md: This README file.
runs/: Directory containing training runs and model weights.
Installation
Clone the repository:

Create a virtual environment and activate it:

Install the required packages:

Usage
Training the Model
Ensure that the data.yaml file is correctly configured with the paths to your training and validation datasets.
Run the train.py script to start training the YOLO model:
Real-Time Object Detection
Ensure that the trained model weights (best.pt) are available in the specified path.
Run the check.py script to start real-time object detection using the webcam:
Listing Available Camera Devices
Run the ListofCamera.py script to list available camera devices on your system:
Configuration
data.yaml
The data.yaml file should be configured as follows:

train.py
The train.py script is used to train the YOLO model:

check.py
The check.py script is used for real-time object detection:

ListofCamera.py
The ListofCamera.py script lists available camera devices:

Acknowledgments
I would like to thank FLEX Pvt Ltd for providing the opportunity to work on this project. Special thanks to my supervisor and the entire team for their support and guidance throughout the internship.

