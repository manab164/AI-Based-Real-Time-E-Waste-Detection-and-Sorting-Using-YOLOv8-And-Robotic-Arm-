# AI-Based-Real-Time-E-Waste-Detection-and-Sorting-Using-YOLOv8-And-Robotic-Arm-

## Overview

The **AI-Powered E-Waste Detection and Sorting System** is an intelligent waste management solution that uses **YOLOv8** and a **4-DOF robotic arm** to automatically detect, classify, and segregate electronic waste in real time. The system captures live video through a USB camera, identifies e-waste objects using a custom-trained YOLOv8 model, classifies them into **High**, **Moderate**, and **Low** hazard categories, and sends commands to an Arduino-controlled robotic arm for automated pick-and-place operations.

By eliminating manual sorting, the system improves recycling efficiency, reduces human exposure to hazardous materials, and supports sustainable e-waste management.

---

## Key Features

- Real-time e-waste detection using **YOLOv8**.
- Automated classification into **High**, **Moderate**, and **Low** hazard categories.
- Robotic arm-based automatic pick-and-place mechanism.
- Live camera-based object detection.
- Fast and accurate deep learning inference.
- Arduino-controlled robotic arm integration.
- Automated segregation with minimal human intervention.
- Scalable and modular system architecture.

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Computer Vision
- Deep Learning
- Arduino Uno
- 4-DOF Robotic Arm
- USB Camera
- NumPy
- PySerial
- Roboflow
- Google Colab / Local Training

---

## Dataset

The model was trained on a custom e-waste dataset containing multiple electronic components collected and annotated using **Roboflow**.

**Dataset Link:**  
https://app.roboflow.com/manab-jyoti-goswami-lqvan/e-waste-oeexv/8

### Classes Included

- Mobile Phone
- Laptop
- Mouse
- PCB
- Cable
- Bulb
- Battery
- Other E-Waste Components

Each detected object is classified into one of the following hazard levels:

- High Hazard
- Moderate Hazard
- Low Hazard

---

## Experimental Results

The proposed AI-powered e-waste sorting system demonstrated:

- High-accuracy real-time object detection using YOLOv8.
- Reliable hazard-level classification of electronic waste.
- Fast inference suitable for real-time applications.
- Automated pick-and-place operation using a robotic arm.
- Reduced manual handling of hazardous electronic waste.
- Improved sorting efficiency and operational safety.
- Scalable architecture suitable for smart recycling facilities.

---

## Project Workflow

1. Capture live video using a USB camera.
2. Process each frame using the YOLOv8 object detection model.
3. Detect and classify e-waste objects.
4. Assign the detected object to its corresponding hazard category.
5. Send classification results to the Arduino controller.
6. Control the robotic arm to pick the detected object.
7. Place the object into the appropriate hazard-specific bin.
8. Continue real-time detection and sorting for incoming e-waste.

---

## Applications

- Smart E-Waste Recycling Plants
- Automated Waste Segregation Systems
- Industrial Recycling Facilities
- Research and Educational Projects
- Smart Cities
- Sustainable Waste Management

---

## Future Improvements

- Expand the dataset with more e-waste categories.
- Improve model accuracy using larger and more diverse datasets.
- Deploy the system on embedded devices such as NVIDIA Jetson or Raspberry Pi.
- Integrate IoT for remote monitoring and control.
- Develop a web dashboard for real-time analytics and system monitoring.
- Add conveyor belt automation for large-scale industrial deployment.

---
