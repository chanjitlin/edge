# Edge Computing Labs - Practical Cheat Sheet

## Sound Analytics Lab

### Setup & Prerequisites
- Create virtual environment: `python3 -m venv audio && source audio/bin/activate`
- Required packages: 
  ```bash
  sudo apt install portaudio19-dev
  pip3 install pyaudio sounddevice scipy matplotlib
  pip install librosa
  sudo apt-get install flac
  pip install pocketsphinx SpeechRecognition
  ```

### Audio Recording & Playback
- Record audio: `arecord --duration=10 test.wav`
- Playback audio: `aplay test.wav`

### PyAudio vs. SoundDevice
- **PyAudio**: 
  ```python
  import pyaudio
  audio = pyaudio.PyAudio()
  stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)
  data = stream.read(BUFFER)
  ```
- **SoundDevice**: 
  ```python
  import sounddevice as sd
  data = sd.rec(frames=BUFFER, samplerate=RATE, channels=CHANNELS, dtype='int16', blocking=True)
  ```

### Key Audio Processing Techniques
1. **Audio Waveform Visualization**:
   ```python
   plt.plot(audio_data)
   ```

2. **Spectrum Analysis** (FFT):
   ```python
   import numpy as np
   from scipy.fftpack import fft
   yf = fft(data_int)
   xf = np.fft.fftfreq(BUFFER, 1/RATE)
   # Plot frequencies up to Nyquist frequency
   plt.plot(xf[:BUFFER//2], np.abs(yf[:BUFFER//2]))
   ```

3. **Bandpass Filtering**:
   ```python
   from scipy.signal import butter, sosfilt
   def design_filter(lowfreq, highfreq, fs, order=3):
       nyq = 0.5*fs
       low = lowfreq/nyq
       high = highfreq/nyq
       sos = butter(order, [low, high], btype='band', output='sos')
       return sos
   # Apply filter
   sos = design_filter(500, 2000, 44100)
   filtered_data = sosfilt(sos, audio_data)
   ```

4. **Audio Feature Extraction**:
   ```python
   import librosa
   # Spectrogram
   S_full, phase = librosa.magphase(librosa.stft(y))
   # Mel Spectrogram
   S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
   # MFCC
   mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
   # Chroma
   chroma = librosa.feature.chroma_stft(S=S, sr=sr)
   ```

5. **Speech Recognition**:
   ```python
   import speech_recognition as sr
   r = sr.Recognizer()
   with sr.Microphone() as source:
       r.adjust_for_ambient_noise(source)
       audio = r.listen(source)
   # Google Speech Recognition (online)
   text = r.recognize_google(audio)
   # Sphinx (offline)
   text = r.recognize_sphinx(audio)
   ```

### Common Issues & Solutions
- **High Latency**: Reduce buffer size (trade-off with frequency resolution)
- **Poor Audio Quality**: Check sampling rate, increase bit depth
- **Recognition Errors**: Adjust for ambient noise, speak clearly
- **Microphone Access**: Use `sudo` if permission issues exist

## Image Analytics Lab

### Setup & Prerequisites
- Create virtual environment: `python3 -m venv image && source image/bin/activate`
- Required packages: 
  ```bash
  pip install opencv-python
  pip install scikit-image
  pip install mediapipe
  ```

### Camera Setup
- Test webcam: `fswebcam -r 1280x720 --no-banner image.jpg`
- OpenCV capture:
  ```python
  import cv2
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
      raise IOError("Cannot open webcam")
  ret, frame = cap.read()
  ```

### Key Image Processing Techniques
1. **Color Segmentation**:
   ```python
   # Define RGB boundaries
   boundaries = [([17, 15, 100], [50, 56, 200])] # Red range
   for (lower, upper) in boundaries:
       lower = np.array(lower, dtype="uint8")
       upper = np.array(upper, dtype="uint8")
       mask = cv2.inRange(frame, lower, upper)
       segmented = cv2.bitwise_and(frame, frame, mask=mask)
   ```

2. **HOG Feature Extraction**:
   ```python
   from skimage import feature
   (H, hogImage) = feature.hog(
       gray_image, 
       orientations=9, 
       pixels_per_cell=(8, 8),
       cells_per_block=(2, 2), 
       transform_sqrt=True, 
       visualize=True
   )
   ```

3. **Face Detection**:
   ```python
   # Using OpenCV Haar Cascades
   detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
   faces = detector.detectMultiScale(gray_image)
   
   # Using MediaPipe
   import mediapipe as mp
   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh()
   results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
   ```

4. **Edge Detection**:
   ```python
   edges = cv2.Canny(gray_image, 100, 200)  # Thresholds: min, max
   ```

### MediaPipe Implementation
```python
# Face mesh detection
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Process frame
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)

# Draw landmarks
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
```

### Common Issues & Solutions
- **Low Frame Rate**: Resize images to smaller dimensions
- **Memory Issues**: Process one frame at a time, don't store video
- **Detection Accuracy**: Adjust confidence thresholds
- **Camera Access**: Use `sudo` if permission issues exist

## Video Analytics Lab

### Setup & Prerequisites
- Create virtual environment: `python3 -m venv video && source video/bin/activate`
- Required packages: 
  ```bash
  pip install opencv-python mediapipe
  ```

### Key Video Processing Techniques
1. **Optical Flow**:
   ```python
   # Lucas-Kanade method (sparse)
   lk_params = dict(winSize=(15,15), maxLevel=2)
   # Detect features
   p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
   # Calculate optical flow
   p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
   
   # Farneback method (dense)
   flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
   ```

2. **Hand Landmark Detection**:
   ```python
   from mediapipe.tasks import python
   from mediapipe.tasks.python import vision
   
   # Initialize detector
   base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
   options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
   detector = vision.HandLandmarker.create_from_options(options)
   
   # Process frame
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
   detection_result = detector.detect(mp_image)
   
   # Access landmarks
   hand_landmarks_list = detection_result.hand_landmarks
   ```

3. **Hand Gesture Recognition**:
   ```python
   # Download model
   # wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
   
   # Initialize recognizer
   options = vision.GestureRecognizerOptions(
       base_options=base_options,
       running_mode=vision.RunningMode.LIVE_STREAM,
       num_hands=2,
       result_callback=save_result
   )
   recognizer = vision.GestureRecognizer.create_from_options(options)
   
   # Process frame
   recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)
   ```

4. **Object Detection**:
   ```python
   # Download EfficientDet model
   # wget -q -O efficientdet.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
   
   # Initialize detector
   options = vision.ObjectDetectorOptions(
       base_options=base_options,
       running_mode=vision.RunningMode.LIVE_STREAM,
       max_results=5, 
       score_threshold=0.5,
       result_callback=save_result
   )
   detector = vision.ObjectDetector.create_from_options(options)
   
   # Process frame
   detector.detect_async(mp_image, time.time_ns() // 1_000_000)
   ```

### Common Issues & Solutions
- **Slow Detection**: Lower resolution, decrease frame rate
- **Model Download Issues**: Verify internet connection, check file path
- **Hand Detection Problems**: Ensure good lighting, keep hands in frame

## MQTT Lab

### Setup & Prerequisites
- Create virtual environment: `python3 -m venv mqtt && source mqtt/bin/activate`
- Required packages: 
  ```bash
  pip install paho-mqtt
  ```

### MQTT Broker Setup
```bash
# Install Mosquitto
sudo apt install mosquitto

# Configure Mosquitto
sudo nano /etc/mosquitto/mosquitto.conf
# Add these lines:
# listener 1883
# allow_anonymous true

# Start broker manually
sudo mosquitto -c /etc/mosquitto/mosquitto.conf

# Enable on boot (optional)
sudo systemctl enable mosquitto
```

### MQTT Publisher
```python
import paho.mqtt.client as mqtt
import time

client = mqtt.Client("Publisher")
client.connect("localhost", 1883)  # Change localhost to broker IP

while True:
    client.publish("test/topic", "Hello, MQTT!")
    time.sleep(5)
```

### MQTT Subscriber
```python
import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    print(f"Received message '{message.payload.decode()}' on topic '{message.topic}'")

client = mqtt.Client("Subscriber")
client.on_message = on_message
client.connect("localhost", 1883)  # Change localhost to broker IP
client.subscribe("test/topic")
client.loop_forever()
```

### Common Issues & Solutions
- **Connection Refused**: Check broker IP and port
- **No Messages Received**: Verify topic names match exactly
- **Broker Not Responding**: Restart Mosquitto service
- **QoS Issues**: Start with QoS 0 and increase as needed

## AWS IoT Core Lab

### Setup & Prerequisites
- Create virtual environment: `python3 -m venv awsiotcore && source awsiotcore/bin/activate`
- Required packages: 
  ```bash
  pip install paho-mqtt psutil
  ```

### Required Files
- AWS IoT Core security files:
  - `aws-certificate.pem.crt`
  - `aws-private.pem.key`
  - `rootCA.pem`

### MQTT Connection to AWS IoT Core
```python
import paho.mqtt.client as mqtt
import ssl
import json
import time

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

client = mqtt.Client()
client.on_connect = on_connect
client.tls_set(
    ca_certs='./rootCA.pem', 
    certfile='./aws-certificate.pem.crt', 
    keyfile='./aws-private.pem.key', 
    tls_version=ssl.PROTOCOL_SSLv23
)
client.tls_insecure_set(True)
client.connect("your-endpoint.iot.region.amazonaws.com", 8883, 60)

# Send CPU usage data
message = json.dumps({
    "time": int(time.time()),
    "quality": "GOOD",
    "hostname": "rpiedge",
    "value": psutil.cpu_percent()
})
client.publish("device/data", payload=message, qos=0, retain=False)

client.loop_forever()
```

### Common Issues & Solutions
- **Certificate Issues**: Verify file names and paths
- **Connection Timeout**: Check internet connectivity
- **Authorization Error**: Confirm policy is attached to certificate
- **Topic Issues**: Ensure topic matches rule configuration

## Deep Learning on Edge Lab

### Setup & Prerequisites
- Create virtual environment: `python3 -m venv dlonedge && source dlonedge/bin/activate`
- Required packages: 
  ```bash
  pip install torch torchvision torchaudio
  pip install opencv-python
  pip install numpy --upgrade
  ```

### Running MobileNetV2 Model
```python
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import cv2

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
quantize = True  # Enable to use quantized model
if quantize:
    torch.backends.quantized.engine = 'qnnpack'
    
weights = MobileNet_V2_QuantizedWeights.DEFAULT
classes = weights.meta["categories"]
net = models.quantization.mobilenet_v2(pretrained=True, quantize=quantize)

# Inference loop
with torch.no_grad():
    while True:
        # Read frame
        ret, image = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        image = image[:, :, [2, 1, 0]]
        
        # Preprocess
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Run model
        output = net(input_batch)
        
        # Get top prediction
        _, predicted = torch.max(output, 1)
        label = classes[predicted[0]]
        
        # Display result
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Classification", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
```

### PyTorch Quantization
```python
# Post-training static quantization
import torch

# 1. Create a model instance
model = YourModelClass()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 2. Fuse modules for better quantization
torch.quantization.fuse_modules(model, [['conv1', 'relu1']], inplace=True)

# 3. Prepare model for quantization
model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)

# 4. Calibrate with representative data
with torch.no_grad():
    for data, _ in calibration_loader:
        model(data)

# 5. Convert to quantized model
torch.quantization.convert(model, inplace=True)

# 6. Save the quantized model
torch.save(model.state_dict(), 'quantized_model.pth')
```

### Common Issues & Solutions
- **Slow Inference**: Enable quantization, reduce input resolution
- **GPU Memory Issues**: Reduce batch size, use CPU for inference
- **Model Accuracy Drop**: Fine-tune on target domain
- **Framework Compatibility**: Use ONNX for cross-framework deployment

## Edge Impulse Lab

### Setup & Prerequisites
```bash
# Install dependencies
sudo apt update
curl -sL https://deb.nodesource.com/setup_18.x | sudo bash -
sudo apt install -y gcc g++ make build-essential nodejs sox gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps

# Install Edge Impulse CLI
sudo npm install edge-impulse-linux -g --unsafe-perm
```

### Connecting to Edge Impulse
```bash
# Connect device to Edge Impulse
sudo edge-impulse-linux

# Join another project
sudo edge-impulse-linux --clean
```

### Audio Data Collection
1. Create an Edge Impulse project
2. Connect your Raspberry Pi
3. Select Microphone as data source
4. Set appropriate sample length
5. Click "Start sampling"
6. Label your data (e.g., "noise", "faucet")
7. Split data into training and testing sets

### Model Training Workflow
1. Create impulse: Data → Feature extraction → ML model
2. Extract features (e.g., spectogram)
3. Train neural network classifier
4. Test model performance
5. Deploy to device

### Running Deployed Model
```bash
sudo edge-impulse-linux-runner
```

### Common Issues & Solutions
- **Connection Issues**: Check internet connectivity
- **Sampling Problems**: Ensure microphone is properly connected
- **Model Accuracy**: Collect more diverse training data
- **Resource Limitations**: Optimize model complexity for target device

## Common Lab Patterns and Best Practices

### Virtual Environment Management
```bash
# Create virtual environment
python3 -m venv env_name

# Activate virtual environment
source env_name/bin/activate

# Install requirements
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Debugging Tools
```bash
# Check USB devices
lsusb

# Check audio devices
arecord -l

# Test camera
v4l2-ctl --list-devices

# Check process status
ps aux | grep program_name

# Monitor resource usage
htop
```

### Performance Optimization Tips
1. **Image/Video Processing**:
   - Resize images to smaller dimensions
   - Convert to grayscale when color not needed
   - Use appropriate data types (uint8 vs. float32)
   
2. **Audio Processing**:
   - Balance buffer size with latency requirements
   - Pre-filter audio to target frequencies
   - Use efficient algorithms (FFT size as power of 2)
   
3. **Model Deployment**:
   - Quantize models where possible
   - Batch process when real-time not required
   - Profile before optimization
   
4. **IoT Communication**:
   - Choose appropriate QoS level
   - Minimize payload size
   - Use persistent connections
   
5. **General Edge Optimization**:
   - Minimize memory allocations
   - Reduce disk I/O operations
   - Balance CPU/GPU workloads

### Key GitHub Lab Repositories
- **Sound Analytics**: https://github.com/drfuzzi/INF2009_SoundAnalytics
- **Image Analytics**: https://github.com/drfuzzi/INF2009_ImageAnalytics  
- **Video Analytics**: https://github.com/drfuzzi/INF2009_VideoAnalytics
- **MQTT**: https://github.com/drfuzzi/INF2009_MQTT
- **Edge Impulse**: https://github.com/drfuzzi/INF2009_EdgeImpulse
- **AWS IoT Core**: Repo with AWS IoT Core samples
- **Deep Learning on Edge**: Repo with MobileNet examples
