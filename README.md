# Syntecxhub_Project - Hand Gesture Recognition
AI-based Hand Gesture Recognition System using TensorFlow, OpenCV, and Python for real-time gesture detection through webcam input.

# 🤖 AI Hand Gesture Recognition System

A real-time Hand Gesture Recognition System built using Python, TensorFlow, OpenCV, and NumPy. This project uses Computer Vision and Deep Learning techniques to recognize hand gestures through webcam input and display predictions in real time.

## 🚀 Features

- Real-time hand gesture detection
- AI-powered gesture recognition using TensorFlow
- Webcam-based live prediction
- Custom dataset creation and training
- OpenCV image processing
- Recognition of gestures such as:
  - ✊ Fist
  - ✋ Palm
  - 👍 Thumbs Up
  - 🖐 Open Hand

## 🛠️ Technologies Used

- Python
- TensorFlow
- OpenCV
- NumPy

## 📂 Project Structure

```text
HandGestureRecognition/
│
├── dataset/
│   ├── fist/
│   ├── palm/
│   ├── open_hand/
│   └── thumbs_up/
│
├── model/
│   └── gesture_model.h5
│
├── capture_data.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/HandGestureRecognition.git
cd HandGestureRecognition
```

### Install Dependencies

```bash
pip install tensorflow opencv-python numpy
```

## 📸 Dataset Collection

Run the dataset collection script:

```bash
py -3.10 capture_data.py
```

Enter the gesture name and capture multiple images for training.

Example:

```text
fist
palm
open_hand
thumbs_up
```

## 🧠 Train the Model

Train the gesture recognition model:

```bash
py -3.10 train_model.py
```

The trained model will be saved in:

```text
model/gesture_model.h5
```

## 🎯 Run Real-Time Prediction

Start live gesture recognition:

```bash
py -3.10 predict.py
```

The webcam will open and display real-time gesture predictions.

## 📈 Learning Outcomes

Through this project, I gained hands-on experience in:

- Computer Vision
- Deep Learning Fundamentals
- TensorFlow Model Training
- Dataset Collection & Preprocessing
- Real-Time AI Applications
- OpenCV Image Processing

## 🔮 Future Improvements

- Sign Language Recognition
- Virtual Mouse Control
- Volume Control with Gestures
- Gesture-Based Presentation Controller
- More Gesture Categories
- Improved Model Accuracy

## 👨‍💻 Author

Rohan Bhowmik

Passionate about Artificial Intelligence, Machine Learning, Computer Vision, and Software Development.

---

⭐ If you found this project interesting, consider giving it a star!
