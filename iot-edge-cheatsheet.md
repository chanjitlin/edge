# IoT & Edge Computing - Comprehensive Exam Cheat Sheet

## Intro to Edge Computing

### Cloud vs. Edge Computing
- **Cloud Computing**:
  - Benefits: Lower maintenance costs, scalability, virtualization
  - Limitations: Latency, bandwidth requirements, privacy concerns
  - Processing happens in centralized data centers
  
- **Edge Computing**:
  - Benefits: Reduced latency, bandwidth efficiency, enhanced privacy, offline operation
  - Limitations: Limited computational resources, higher initial cost, complex deployment
  - Processing happens closer to data source

### Edge Computing Platforms
- **CPU-Centric**: Raspberry Pi, Arduino (general-purpose computing)
- **GPU-Centric**: NVIDIA Jetson (parallel processing, vision applications)
- **FPGA/ASIC**: Custom hardware accelerators (specialized workloads)

### AIoT (Artificial Intelligence of Things)
- **Definition**: Integration of AI capabilities with IoT devices
- **Benefits**:
  - Increased productivity through automation
  - Improved energy efficiency through optimization
  - Enhanced decision-making without cloud dependency
  - Real-time insights from local data processing
- **Limitations**:
  - Implementation costs
  - Limited on-device processing power
  - Technical expertise requirements
  - Power consumption challenges

### Edge Deployment Challenges
- **Power**: Battery life, energy harvesting, power efficiency
- **Thermal**: Heat dissipation, operating temperature range
- **Environmental**: Waterproofing, dust protection (IP ratings)
- **Connectivity**: Intermittent connections, protocol compatibility
- **Security**: Physical access vulnerabilities, limited encryption capacity

## Audio Analytics on Edge

### Audio Fundamentals
- **Sampling Rate**: 44.1 kHz (CD quality), 48 kHz (professional audio)
- **Human Hearing Range**: 20 Hz - 20 kHz
- **Adult Voice**: Fundamental frequency range 85-155 Hz (male), 165-255 Hz (female)
- **Buffer Size**: Larger buffers = better frequency resolution but more latency
- **Signal Types**: Stationary vs. Non-stationary (speech is non-stationary)

### Audio Signal Processing Concepts
- **Time Domain Features**:
  - Amplitude Envelope: Overall volume contour
  - Zero Crossing Rate: Frequency estimation, voiced/unvoiced detection
  - Root Mean Square Energy: Signal power measurement
  
- **Frequency Domain Features**:
  - Spectrum: Frequency components via FFT
  - Spectrogram: Spectrum over time (time-frequency representation)
  - Spectral Centroid: "Brightness" of sound
  - Spectral Flux: Frame-to-frame spectrum change
  
- **Perceptual Features**:
  - Mel Scale: Perceptually-based frequency scale (humans distinguish lower frequencies better)
  - MFCC (Mel-Frequency Cepstral Coefficients): Speech recognition gold standard
  - Chromagram: Music analysis, pitch class representation
  - Cepstrum: "Spectrum of a spectrum," useful for pitch detection

### Audio Filters
- **Low-Pass**: Attenuates high frequencies, keeps low frequencies
- **High-Pass**: Attenuates low frequencies, keeps high frequencies
- **Band-Pass**: Keeps frequencies within a range, attenuates others
- **Notch**: Removes specific frequency band

### Audio Models for Edge
- **YAMNet**: 
  - Lightweight sound classification model based on MobileNetV1
  - Pre-trained on AudioSet (600+ audio classes)
  - Suitable for edge devices due to efficiency
  
- **Speech Recognition**:
  - On-Device: CMUSphinx (offline, lightweight)
  - Cloud APIs: Google Speech API, AWS Transcribe (more accurate)
  - Wake word detection (e.g., "Hey Siri", "Alexa")

### Audio Libraries for Edge
- **PyAudio**: Audio I/O with Python
- **Librosa**: Feature extraction, analysis
- **SoundDevice**: Modern Python audio I/O
- **Speech_Recognition**: Python library for various speech APIs

## Image and Video Analytics on Edge

### Image Fundamentals
- **Representation**: 
  - Grayscale: 1 channel (intensity values)
  - RGB: 3 channels (red, green, blue values)
  - Matrix size: Height × Width × Channels
  
- **Characteristics**:
  - High Frequency: Edges, textures, noise (rapid intensity changes)
  - Low Frequency: Smooth regions, backgrounds (gradual changes)
  - Resolution: Higher = more detail but more processing required
  - Color Space: RGB, HSV, YCbCr (different representations for different tasks)

### Image Processing Techniques
- **Color Segmentation**: Isolating regions by color thresholds
- **Edge Detection**: 
  - Gradient operations (Sobel, Prewitt)
  - Process: Smoothing → Gradient → Thresholding
  - Identifies boundaries between regions
  
- **Feature Extraction**:
  - HOG (Histogram of Oriented Gradients): 
    - Captures local shape information
    - Grid of gradient histograms
    - Used for human/face detection
  - Local Binary Patterns: Texture description
  - SIFT/SURF: Scale-invariant features (patented)
  - ORB: Free alternative to SIFT/SURF

- **Face/Landmark Detection**:
  - Haar Cascades: Fast but less accurate
  - HOG + SVM: Better accuracy, still efficient
  - CNN-based: Most accurate but computationally intensive
  - MediaPipe: Efficient landmark tracking (face, hand, body)

### Video Processing
- **Optical Flow**: 
  - Tracks pixel movement between frames
  - Lucas-Kanade: Sparse tracking of specific points
  - Farneback: Dense flow calculation for all pixels
  - Applications: Motion detection, gesture recognition
  
- **Background Subtraction**:
  - Identifies moving objects by comparing to background model
  - Methods: Frame differencing, MOG2, KNN
  
- **Video Compression**:
  - Key frames (I-frames) and predicted frames (P/B-frames)
  - Reduces redundancy between frames
  - Critical for edge bandwidth management

### Computer Vision Libraries
- **OpenCV**: Comprehensive computer vision library
  - Key Functions: 
    - `cv2.VideoCapture()`: Capture video
    - `cv2.cvtColor()`: Convert color spaces
    - `cv2.Canny()`: Edge detection
    - `cv2.HoughCircles()`: Circle detection
    - `cv2.findContours()`: Shape detection
    - `cv2.calcOpticalFlowFarneback()`: Dense optical flow

- **MediaPipe**: Google's ML pipeline for vision tasks
  - Solutions: Face mesh, hand tracking, pose estimation
  - Optimized for mobile and edge deployment

## Deep Learning on the Edge

### Model Optimization Techniques

#### Quantization
- **Definition**: Reducing precision of weights/activations (FP32 → INT8/INT4)
- **Types**:
  - Post-Training Quantization: Convert after training
  - Quantization-Aware Training: Train with simulated quantization
- **Techniques**:
  - Symmetric: x_q = round(x/scale)
  - Asymmetric: x_q = round(x/scale) + zero_point
  - Per-tensor vs. Per-channel (more accurate)
- **Benefits**: 3-4× smaller models, faster inference, reduced memory usage
- **Scale Factor Calculation**: scale = (max_val - min_val) / (2^bits - 1)

#### Pruning
- **Definition**: Removing unnecessary weights/connections
- **Types**:
  - Unstructured: Individual weights (higher compression)
  - Structured: Entire channels/filters (better hardware acceleration)
- **Process**: Train → Prune → Fine-tune (iterative)
- **Criteria**: Magnitude-based, L1/L2 regularization
- **Optimal Weight Distribution**: Centered around zero (bimodal Gaussian)

#### Model Distillation
- **Concept**: Training smaller "student" model to mimic larger "teacher" model
- **Process**:
  - Train large teacher model
  - Use teacher's predictions as soft targets
  - Train smaller student model on combination of true labels and soft targets
- **Benefits**: Smaller models with better accuracy than training from scratch

#### Neural Architecture Search (NAS)
- **Purpose**: Automatically finding optimal model architecture
- **Approaches**: Reinforcement learning, evolutionary algorithms, gradient-based
- **Edge-specific NAS**: Optimizing for inference time, memory, and energy constraints

### Frameworks for Edge Deployment
- **TensorFlow Lite**:
  - Converter: `tflite_convert`
  - 8-bit quantization support
  - Micro version for MCUs
  
- **PyTorch Mobile**:
  - `torch.quantization` API
  - JIT tracing/scripting
  - Model optimization toolkit
  
- **ONNX Runtime**:
  - Cross-framework compatibility
  - Hardware-specific optimizations
  - Quantization and operator fusion

### Hardware Acceleration
- **CPU**: NEON (ARM), AVX (x86)
- **GPU**: CUDA (NVIDIA), OpenCL
- **NPU/TPU**: Purpose-built neural processors
- **DSP**: Digital Signal Processors for specific operations

## Edge AI Tools and Challenges

### Edge Development Frameworks
- **TensorFlow Lite**:
  - Converter: `tflite_convert --post_training_quantize`
  - Micro version for MCUs
  - Hardware acceleration support

- **PyTorch Mobile**:
  - Direct export from PyTorch
  - Android and iOS support
  - Quantization tools

- **ONNX Runtime**:
  - Cross-framework compatibility
  - Hardware acceleration
  - Model optimization tools

- **Edge Impulse**:
  - End-to-end development platform
  - Data collection to deployment
  - Supports TinyML devices
  - Command: `edge-impulse-linux`

- **Hardware-Specific SDKs**:
  - NVIDIA TensorRT: GPU optimization
  - Intel OpenVINO: Intel hardware acceleration
  - X-CUBE-AI: STM32 optimization

### Profiling and Benchmarking
- **Definition**: Analyzing performance to identify bottlenecks
- **Key Metrics**:
  - Inference time (latency)
  - Throughput (inferences per second)
  - Memory usage
  - Power consumption
  
- **Tools**:
  - `perf`: Linux performance analysis
  - TensorFlow Profiler
  - PyTorch Profiler
  - Hardware-specific tools (NVIDIA Nsight, Intel VTune)

### Federated Learning
- **Definition**: Training models across devices without sharing raw data
- **Process**:
  - Central server distributes model
  - Devices train on local data
  - Devices send model updates (not data)
  - Server aggregates updates
  
- **Types**:
  - Horizontal: Same features, different samples
  - Vertical: Different features, same samples
  - Hierarchical: Multi-level aggregation

### Edge Fleet Management
- **Definition**: Managing deployment and updates across multiple edge devices
- **Components**:
  - OTA updates
  - Monitoring and telemetry
  - Version control
  - Security management

### Common Challenges
- **Heterogeneous Hardware**: Diverse devices with different capabilities
- **Limited Resources**: Memory, computation, power constraints
- **Security**: Protecting models and data on exposed devices
- **Reliability**: Handling device failures, network issues
- **Model Drift**: Performance degradation over time
- **Debugging**: Limited visibility into deployed systems

## IoT Network Protocols

### MQTT (Message Queuing Telemetry Transport)
- **Architecture**: Publisher → Broker → Subscriber
- **QoS Levels**:
  - QoS 0: "At most once" - fire and forget
  - QoS 1: "At least once" - guaranteed delivery, possible duplicates
  - QoS 2: "Exactly once" - guaranteed delivery, no duplicates
  
- **Key Components**:
  - Topics: Hierarchical structure (e.g., `home/livingroom/temperature`)
  - Clients: Publishers and Subscribers
  - Broker: Central message handler
  
- **Features**:
  - Retained Messages: Last message stored for new subscribers
  - Last Will and Testament (LWT): Message sent if client disconnects abnormally
  - Clean/Persistent Sessions: State management
  
- **Technical Details**:
  - Port: 1883 (default), 8883 (TLS/SSL)
  - Low overhead: 2-byte minimum header
  - Designed for constrained networks
  
- **MQTT 5 Additions**:
  - Shared subscriptions
  - Topic aliases
  - Message expiry
  - User properties

### CoAP (Constrained Application Protocol)
- **Architecture**: REST-based client-server using UDP
- **RFC**: 7252
- **Message Types**:
  - Confirmable (CON): Requires ACK
  - Non-confirmable (NON): No ACK required
  - Acknowledgement (ACK): Confirms CON receipt
  - Reset (RST): Error response
  
- **Methods**: GET, POST, PUT, DELETE (HTTP-like)
- **Features**:
  - Observe option for subscriptions
  - Resource discovery (`/.well-known/core`)
  - Block transfer for large data
  - Multicast support
  
- **Technical Details**:
  - Default port: 5683 (UDP)
  - 4-byte header (compact)
  - Built-in content negotiation
  - Proxy capability to HTTP

### LoRa/LoRaWAN
- **LoRa**: Physical layer, Chirp Spread Spectrum modulation
- **LoRaWAN**: MAC protocol built on LoRa
- **Frequency Bands**:
  - 868 MHz (Europe)
  - 915 MHz (North America)
  - 923 MHz (Asia/Singapore)
  
- **Key Specifications**:
  - Range: 2-5km urban, up to 15km suburban
  - Data rate: 0.3-50 kbps
  - Very low power consumption
  
- **Device Classes**:
  - Class A: Lowest power, uplink-triggered downlink
  - Class B: Scheduled downlink slots (beacon synchronized)
  - Class C: Continuous listening, lowest latency downlink
  
- **Network Architecture**:
  - End devices
  - Gateways (transparent bridges)
  - Network server (manages network)
  - Application server (handles data)

### Bluetooth Low Energy (BLE)
- **Introduction**: Bluetooth 4.0 (2010)
- **Frequency**: 2.4 GHz ISM band
- **Range**: 
  - BLE 4.0/4.2: ~50m
  - BLE 5.0+: up to 400m (theoretical)
  
- **Key Concepts**:
  - Advertising: Device discovery on 3 dedicated channels
  - Connections: Point-to-point data exchange
  - GATT (Generic Attribute Profile): Data organization
  - Characteristics and Services: Data structure
  
- **Roles**:
  - Central: Initiates connections, typically more powerful
  - Peripheral: Advertises, accepts connections
  
- **Power Consumption**:
  - Sleep current: <1µA
  - Peak transmit: ~15mA
  - Designed for coin cell batteries (months/years)

### Zigbee
- **Based On**: IEEE 802.15.4
- **Frequency**: 2.4 GHz globally, 915 MHz (US), 868 MHz (Europe)
- **Range**: 10-100m typical
- **Device Types**:
  - Coordinator: One per network, initiates formation
  - Router: Extends network, relays messages
  - End Device: Limited functionality, low power
  
- **Topologies**:
  - Star: Single coordinator with end devices
  - Mesh: Multiple routers enabling multiple paths
  - Cluster Tree: Hierarchical routing structure
  
- **Features**:
  - Self-healing mesh
  - AES-128 encryption
  - Up to 65,000 nodes
  - Low power consumption
  - Interoperable with standardized profiles
  
- **Comparison with Z-Wave**:
  - Z-Wave: Sub-GHz bands (less interference), proprietary
  - Zigbee: 2.4 GHz (global standard), open specification
  - Both create mesh networks

### RESTful APIs
- **Architecture**: Representational State Transfer
- **HTTP Methods**:
  - GET: Retrieve resources
  - POST: Create new resources
  - PUT: Update entire resources
  - PATCH: Update part of resources
  - DELETE: Remove resources
  
- **Status Codes**:
  - 2xx: Success (200 OK, 201 Created)
  - 3xx: Redirection
  - 4xx: Client errors (404 Not Found)
  - 5xx: Server errors
  
- **Constraints**:
  - Client-server architecture
  - Statelessness
  - Cacheability
  - Uniform interface
  - Layered system
  
- **Resources**: Entities identified by URIs
- **Representations**: JSON, XML, etc.

### Wi-Fi HaLow (802.11ah)
- **Purpose**: Low-power, long-range Wi-Fi for IoT
- **Frequency**: Sub-1 GHz bands (902-928 MHz in US)
- **Range**: Up to 1km
- **Data Rate**: 150 Kbps to 347 Mbps
- **Power Efficiency**: Designed for battery-operated devices
- **Advantages**: Leverages existing Wi-Fi infrastructure

## Deployment on Edge Devices

### Profiling Techniques
- **Performance Analysis**:
  - Layer-wise execution time
  - Memory usage patterns
  - CPU/GPU utilization
  
- **Memory Analysis**:
  - Cache misses and hits
  - Memory access patterns
  - Heap allocations/deallocations
  
- **System-Level Profiling**:
  - Cycles, instructions, cache misses
  - Branch mispredictions (unpredictable control flow)
  - I/O operations and waiting time
  
- **Network Profiling**:
  - Latency, bandwidth usage
  - Packet loss patterns
  - Connection establishment time

- **Tools**:
  - `perf`: Linux performance analysis
    - `perf stat`: Basic metrics
    - `perf record`: Sampling data
    - `perf report`: Analysis visualization
  - Valgrind's Cachegrind: Cache performance
  - ARM Streamline: ARM-specific profiling
  - NVIDIA Nsight: GPU profiling

### Bottleneck Analysis
- **CPU Bottlenecks**:
  - High utilization
  - Low instructions per cycle (IPC)
  - High context switching
  
- **Memory Bottlenecks**:
  - High cache miss rates
  - Memory thrashing
  - Heap fragmentation
  
- **I/O Bottlenecks**:
  - Excessive disk operations
  - Network congestion
  - Bus contention
  
- **Case Studies**:
  - High cache miss rate → Improve data locality
  - Low IPC → Optimize branch prediction or loop unrolling
  - High `memcpy` overhead → Reduce data movement

### Optimization Techniques
- **Memory Management**:
  - Memory Pooling: Reuse buffers
  - Custom allocators for specific patterns
  - Zero-copy operations where possible
  
- **Computation Optimization**:
  - Layer Fusion: Combine operations
  - Loop unrolling for improved parallelism
  - SIMD instructions for vectorization
  
- **Scheduling**:
  - Dynamic Task Scheduling
  - Workload balancing across cores
  - Heterogeneous computing (CPU+GPU+NPU)
  
- **Model-specific**:
  - Operator fusion
  - Graph optimization
  - Hardware-specific kernel tuning

### Dynamic and Adaptive Techniques
- **Dynamic Task Scheduling**:
  - Work stealing algorithms
  - Priority-based scheduling
  - Adaptive resource allocation
  
- **Adaptive Model Partitioning**:
  - Edge-cloud offloading based on conditions
  - Network-aware computation splitting
  - Battery-aware task distribution
  
- **Adaptive Precision**:
  - Runtime precision switching
  - Power-based quantization levels
  - Importance-based mixed precision

### Real-world Case Studies
- **Smart Camera**:
  - Motion-activated object detection
  - Low-power standby, high-power analysis
  - Local filtering before cloud upload
  
- **Smart Home Camera**:
  - Face recognition partitioning
  - Simple detection on edge, complex matching in cloud
  - Privacy-preserving local processing
  
- **Smart Farm**:
  - Adaptive resource allocation based on growth stage
  - Prioritization of critical monitoring tasks
  - Dynamic sensor sampling rates

### Hardware Acceleration
- **Domain-Specific Accelerators**:
  - Neural Processing Units (NPUs)
  - Vision Processing Units (VPUs)
  - Tensor cores for matrix operations
  
- **Optimization for Accelerators**:
  - Operator fusion
  - Memory tiling for cache efficiency
  - Data layout optimization (NCHW vs NHWC)
  
- **Future Trends**:
  - AI-driven optimization
  - Cross-layer optimization
  - Federated and collaborative optimization

## Key Practical Skills and Commands

### Setting Up Edge Environments
```bash
# Virtual Environment
python3 -m venv venv_name
source venv_name/bin/activate

# Package Installation
pip install tensorflow-lite
pip install torch torchvision
pip install opencv-python
pip install paho-mqtt
```

### Audio Analysis
```bash
# Audio Recording/Playback
arecord --duration=10 test.wav
aplay test.wav

# Audio Libraries
pip install librosa pyaudio sounddevice speech_recognition

# Running Audio Analysis
python audio_features.py  # Feature extraction
python microphone_recognition.py  # Speech recognition
```

### Image/Video Processing
```bash
# Image Capture
fswebcam -r 1280x720 --no-banner image.jpg

# Video Recording
ffmpeg -f v4l2 -framerate 25 -video_size 640x480 -i /dev/video0 output.mp4

# OpenCV Basic Commands
python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); cv2.imwrite('frame.jpg', frame)"
```

### MQTT Commands
```bash
# Start Mosquitto Broker
mosquitto -c /etc/mosquitto/mosquitto.conf

# Subscribe to Topic
mosquitto_sub -h localhost -t "test/topic"

# Publish to Topic
mosquitto_pub -h localhost -t "test/topic" -m "Hello MQTT"
```

### Model Optimization
```bash
# TensorFlow Lite Conversion
tflite_convert --saved_model_dir=/tmp/model --output_file=/tmp/model.tflite --post_training_quantize

# PyTorch Quantization
python -c "import torch; model = torch.load('model.pth'); torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)"

# Edge Impulse
edge-impulse-linux  # Connect device
edge-impulse-linux-runner  # Run model
```

### Performance Analysis
```bash
# Basic Performance Stats
sudo perf stat python script.py

# Detailed Performance Recording
sudo perf record -g python script.py
sudo perf report

# Memory Analysis
valgrind --tool=massif python script.py
```

### AWS IoT Core
```bash
# Testing AWS IoT Connection
mosquitto_pub --cafile rootCA.pem --cert device.pem.crt --key private.pem.key -h endpoint.iot.region.amazonaws.com -p 8883 -t "device/data" -m "test" -d

# Running Python AWS IoT Client
python aws_iot_core/pipython.py
```

## Common Pitfalls and Quick Solutions

### Audio Analytics
- **Pitfall**: Incorrect sampling rate causing frequency distortion
  - **Solution**: Always specify and check sampling rate (44.1kHz standard)

- **Pitfall**: Buffer size too small causing poor frequency resolution
  - **Solution**: Increase buffer size but be aware of latency tradeoff

- **Pitfall**: Noisy audio reducing recognition accuracy
  - **Solution**: Apply bandpass filtering for the frequency range of interest

### Image/Video Analytics
- **Pitfall**: High resolution causing slow processing
  - **Solution**: Downsample images to appropriate size (e.g., 256x256)

- **Pitfall**: Lighting variations affecting detection
  - **Solution**: Use histogram equalization or normalization

- **Pitfall**: Memory issues with large video frames
  - **Solution**: Process frames sequentially, avoid storing all frames

### Deep Learning Models
- **Pitfall**: Model too large for device memory
  - **Solution**: Apply quantization, pruning, or distillation

- **Pitfall**: Slow inference due to complex architecture
  - **Solution**: Fuse operations, optimize graph, consider lighter architecture

- **Pitfall**: Accuracy drop after quantization
  - **Solution**: Use quantization-aware training or fine-tuning

### IoT Protocols
- **Pitfall**: High power consumption in battery-powered devices
  - **Solution**: Use correct sleep modes, optimize connection frequency

- **Pitfall**: Message loss in unreliable networks
  - **Solution**: Use MQTT QoS 1 or 2, or implement retry logic

- **Pitfall**: Security vulnerabilities in IoT communications
  - **Solution**: Always use TLS/SSL, certificate-based authentication

### Deployment
- **Pitfall**: Thermal throttling affecting performance
  - **Solution**: Implement proper cooling, thermal management

- **Pitfall**: Unexpected environmental conditions
  - **Solution**: Test in actual deployment environment, add appropriate casing

- **Pitfall**: Difficult debugging in deployed systems
  - **Solution**: Implement comprehensive logging and remote monitoring