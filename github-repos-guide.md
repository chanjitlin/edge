# INF2009 GitHub Repositories - Detailed Guide

This document provides a comprehensive explanation of all GitHub repositories used in the INF2009 course on IoT and Edge Computing.

## Table of Contents
- [Sound Analytics (INF2009_SoundAnalytics)](#sound-analytics)
- [Image Analytics (INF2009_ImageAnalytics)](#image-analytics)
- [Video Analytics (INF2009_VideoAnalytics)](#video-analytics)
- [IoT Communications & MQTT (INF2009_MQTT)](#iot-communications--mqtt)
- [AWS IoT Core](#aws-iot-core)
- [Edge Impulse (INF2009_EdgeImpulse)](#edge-impulse)
- [Raspberry Pi Setup (INF2009_Setup)](#raspberry-pi-setup)

## Sound Analytics

**Repository**: [INF2009_SoundAnalytics](https://github.com/drfuzzi/INF2009_SoundAnalytics)

### Overview
This repository focuses on audio signal processing and analysis using Raspberry Pi and a microphone. It demonstrates how to capture, process, and analyze audio data for various applications like speech recognition and audio feature extraction.

### Key Components

#### Audio Capture and Processing
- **`microphone_streaming_with_spectrum.py`**: 
  - Demonstrates real-time audio capture using PyAudio
  - Performs Fast Fourier Transform (FFT) to display audio spectrum
  - Visualizes both time-domain waveform and frequency-domain spectrum
  - Uses matplotlib for real-time visualization

- **`microphone_streaming_with_spectrum_updated.py`**:
  - Updated version using SoundDevice library instead of PyAudio
  - Provides same functionality with more modern API

#### Audio Filtering
- **`filtering_audio.py`**:
  - Implements bandpass filtering using scipy's butter and sosfilt functions
  - Allows isolation of specific frequency ranges
  - Shows both original and filtered audio waveforms in real-time

- **`filtering_audio_updated.py`**:
  - Updated version using SoundDevice library

#### Speech Recognition
- **`microphone_recognition.py`**:
  - Demonstrates speech recognition using both Google Speech API and Sphinx
  - Compares online (Google) vs. offline (Sphinx) recognition accuracy
  - Shows timing differences between approaches

#### Audio Feature Extraction
- **`audio_features.py`**:
  - Extracts various audio features using librosa library:
    - Spectrogram: Visual representation of frequency spectrum over time
    - Chromagram: Representation of pitch class distribution (useful for music)
    - Mel-spectrogram: Perceptually-based spectrogram using Mel scale
    - MFCC (Mel-Frequency Cepstral Coefficients): Features commonly used in speech recognition

### Key Technologies
- **PyAudio/SoundDevice**: Audio capture libraries
- **scipy**: Signal processing functions (FFT, filtering)
- **librosa**: Audio feature extraction
- **speech_recognition**: Python library interfacing with speech recognition APIs

### Applications
- Speech recognition systems
- Audio classification
- Music analysis
- Noise detection and filtering
- Voice-controlled interfaces

## Image Analytics

**Repository**: [INF2009_ImageAnalytics](https://github.com/drfuzzi/INF2009_ImageAnalytics)

### Overview
This repository focuses on image processing and computer vision techniques using Raspberry Pi and a webcam. It covers fundamental techniques like color segmentation and feature extraction, as well as more advanced topics like face and landmark detection.

### Key Components

#### Basic Image Processing
- **`image_capture_display.py`**:
  - Captures images from webcam using OpenCV
  - Performs color segmentation to isolate red, green, and blue components
  - Shows both original and color-segmented images

#### Feature Extraction
- **`image_hog_feature.py`**:
  - Implements Histogram of Oriented Gradients (HOG) feature extraction
  - Visualizes HOG features alongside original image
  - Demonstrates how HOG captures shape information used in object detection

#### Face and Human Detection
- **`image_human_capture.py`**:
  - Uses OpenCV's HOGDescriptor for human detection
  - Demonstrates multi-scale detection approach
  - Includes bounding box visualization and center tracking

- **`image_human_capture_opencv.py`**:
  - Alternative implementation using Haar cascade classifier
  - Shows simpler approach to face detection

#### Facial Landmark Detection
- **`image_face_capture.py`**:
  - Uses MediaPipe's face_mesh model
  - Detects and visualizes detailed facial landmarks
  - Shows real-time face tracking and mesh overlay

- **`image_live_facial_landmarks.py`**:
  - Alternative implementation using face_recognition library
  - Extracts specific facial landmarks (eyes, nose, mouth)

### Key Technologies
- **OpenCV**: Computer vision library
- **MediaPipe**: Google's framework for perception tasks
- **scikit-image**: Scientific image processing library
- **face_recognition**: Simplified facial recognition library

### Applications
- Face detection and recognition
- Human detection and tracking
- Feature extraction for machine learning
- Gesture recognition
- Emotion detection

## Video Analytics

**Repository**: [INF2009_VideoAnalytics](https://github.com/drfuzzi/INF2009_VideoAnalytics)

### Overview
This repository extends image analytics to video streams, focusing on motion analysis, object tracking, and real-time inference on video data. It demonstrates advanced techniques for processing video streams on edge devices.

### Key Components

#### Optical Flow
- **`optical_flow.py`**:
  - Implements two approaches to optical flow:
    - Lucas-Kanade method: Sparse flow tracking specific points
    - Farneback method: Dense flow calculating movement for all pixels
  - Visualizes motion patterns with directional arrows/streamlines
  - Demonstrates motion tracking capabilities

#### Hand Detection and Tracking
- **`hand_landmark.py`**:
  - Uses MediaPipe's hand_landmarker model
  - Detects hand landmarks (21 points per hand)
  - Implements simple gesture detection (thumb up)
  - Shows real-time tracking of hand movements

#### Gesture Recognition
- **`hand_gesture.py`**:
  - Extends hand landmark detection to recognize specific gestures
  - Uses pretrained gesture_recognizer model
  - Classifies gestures like "Victory", "Thumbs up", etc.
  - Displays confidence scores for detected gestures

#### Object Detection
- **`obj_detection.py`**:
  - Implements EfficientDet object detection using MediaPipe
  - Draws bounding boxes around detected objects
  - Shows class labels and confidence scores
  - Demonstrates lightweight object detection suitable for edge devices

### Key Technologies
- **OpenCV**: For video capture and processing
- **MediaPipe**: For efficient ML-based perception tasks
- **TensorFlow Lite**: For running inference on edge devices

### Applications
- Motion detection and tracking
- Gesture-based interfaces
- Object detection and tracking
- Video summarization
- Surveillance systems

## IoT Communications & MQTT

**Repository**: [INF2009_MQTT](https://github.com/drfuzzi/INF2009_MQTT)

### Overview
This repository focuses on IoT communication protocols, specifically MQTT (Message Queuing Telemetry Transport). It demonstrates how to set up an MQTT broker, create publisher and subscriber clients, and implement basic IoT communication patterns.

### Key Components

#### MQTT Broker Setup
- Instructions for installing and configuring Mosquitto MQTT broker
- Configuration options for allowing anonymous clients
- Starting the broker manually or as a service

#### MQTT Client Implementation
- **MQTT Publisher**:
  - Python script using paho-mqtt
  - Publishes messages to specified topics
  - Demonstrates message formatting and QoS settings

- **MQTT Subscriber**:
  - Python script using paho-mqtt
  - Subscribes to specified topics
  - Implements message handling callback

#### Practical Application
- Instructions for implementing a webcam-based image capture system
- Using MQTT for triggering image capture
- Transmitting captured images via MQTT

### Key Technologies
- **Mosquitto**: Open-source MQTT broker
- **Paho MQTT**: Python client library for MQTT
- **MQTT Protocol**: Lightweight publish/subscribe messaging protocol

### Applications
- IoT sensor networks
- Remote monitoring systems
- Smart home automation
- Distributed control systems
- Low-bandwidth communication

## AWS IoT Core

**Repository**: AWS IoT Core examples

### Overview
This repository demonstrates how to connect edge devices (Raspberry Pi) to AWS IoT Core, enabling secure communication with the cloud and integration with other AWS services like DynamoDB.

### Key Components

#### AWS IoT Setup
- Creating a "thing" in AWS IoT Core
- Generating and downloading security certificates
- Configuring policies and permissions

#### Device Communication
- **`pipython.py`**:
  - Establishes secure MQTT connection to AWS IoT Core
  - Uses TLS/SSL with certificates for security
  - Publishes device data (e.g., CPU usage) to AWS IoT Core
  - Demonstrates real-time data transmission

#### AWS IoT Rules
- Creating rules for message routing
- Configuring DynamoDB as action destination
- Setting up table schema and mapping

#### Data Ingestion
- Real-time data flow from device to DynamoDB
- JSON message formatting
- Timestamp-based data organization

### Key Technologies
- **AWS IoT Core**: Managed cloud service for IoT
- **MQTT over TLS**: Secure communication protocol
- **AWS DynamoDB**: NoSQL database service
- **Paho MQTT**: Python client library

### Applications
- Cloud-based IoT solutions
- Remote device monitoring
- IoT data analytics
- Scalable IoT infrastructure
- Secure device communication

## Edge Impulse

**Repository**: [INF2009_EdgeImpulse](https://github.com/drfuzzi/INF2009_EdgeImpulse)

### Overview
This repository guides students through using Edge Impulse with Raspberry Pi and webcam to build, train, and deploy machine learning models on edge devices, with a focus on audio classification.

### Key Components

#### Platform Setup
- Creating Edge Impulse account
- Installing Edge Impulse CLI on Raspberry Pi
- Connecting Raspberry Pi to Edge Impulse platform

#### Data Collection
- Capturing audio samples using device microphone
- Labeling data (e.g., "noise", "faucet")
- Splitting data into training and testing sets

#### Model Development
- Creating an impulse (data processing pipeline)
- Configuring feature extraction (e.g., spectrogram)
- Training neural network classifier
- Testing and validating model performance

#### Model Deployment
- Deploying trained model back to Raspberry Pi
- Running inference on live audio input
- Viewing classification results and confidence scores

### Key Technologies
- **Edge Impulse**: End-to-end development platform for edge ML
- **TensorFlow Lite**: Lightweight ML framework for edge devices
- **DSP (Digital Signal Processing)**: Audio feature extraction

### Applications
- Sound classification systems
- Predictive maintenance
- Anomaly detection
- Voice command systems
- Environmental monitoring

## Raspberry Pi Setup

**Repository**: [INF2009_Setup](https://github.com/drfuzzi/INF2009_Setup)

### Overview
This repository provides instructions for setting up Raspberry Pi 400 with a webcam, establishing the foundation for all other labs in the course.

### Key Components

#### Hardware Setup
- Raspberry Pi 400 configuration
- Connecting Logitech C310 HD Webcam
- Configuring network connections

#### Software Setup
- Installing Raspberry Pi OS
- Enabling VNC for remote access
- Setting up static IP (optional)

#### Webcam Configuration
- Testing webcam functionality
- Capturing images and video
- Checking audio device recognition

#### Advanced Applications
- Python scripting for webcam control
- Creating a simple motion detection system
- Virtual environment setup for Python packages

### Applications
- Foundation for all other labs
- Remote monitoring systems
- Computer vision applications
- IoT device creation
- Edge computing experimentation

## Integration Between Repositories

The repositories in this course are designed to build upon each other, creating a comprehensive understanding of edge computing:

1. **INF2009_Setup** establishes the hardware foundation
2. **INF2009_SoundAnalytics**, **INF2009_ImageAnalytics**, and **INF2009_VideoAnalytics** develop expertise in different sensing modalities
3. **INF2009_MQTT** and **AWS IoT Core** enable communication between edge devices and cloud
4. **INF2009_EdgeImpulse** brings machine learning to the edge

Together, these repositories provide a complete toolkit for developing sophisticated IoT and edge computing applications that can:
- Sense the environment through audio and visual inputs
- Process data locally on edge devices
- Communicate with cloud services securely
- Deploy machine learning models for intelligent decision-making

## Key Technologies Across Repositories

| Repository | Key Technologies | Primary Languages |
|------------|------------------|-------------------|
| Sound Analytics | PyAudio, SoundDevice, librosa, scipy | Python |
| Image Analytics | OpenCV, MediaPipe, scikit-image | Python |
| Video Analytics | OpenCV, MediaPipe, TensorFlow Lite | Python |
| MQTT | Mosquitto, Paho MQTT | Python, Bash |
| AWS IoT Core | AWS IoT, MQTT, DynamoDB | Python, AWS Console |
| Edge Impulse | Edge Impulse SDK, TensorFlow Lite | JavaScript, Python |
| Raspberry Pi Setup | VNC, V4L2, SSH | Bash, Python |

## Common Skills Developed

Across all repositories, students develop several critical skills:
- **Edge Computing Fundamentals**: Resource constraints, optimization techniques
- **Real-time Processing**: Handling streaming data from sensors
- **IoT Communication**: Protocols and security considerations
- **Machine Learning on Edge**: Deployment and optimization of ML models
- **System Integration**: Combining hardware, software, and cloud services
- **Performance Optimization**: Profiling and improving edge application performance
