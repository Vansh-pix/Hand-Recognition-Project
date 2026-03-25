# Hand Gesture Recognition (5 Classes)

This project is a real-time hand gesture recognition system built using OpenCV and TensorFlow.
It uses a Convolutional Neural Network (CNN) trained on frames extracted from gesture videos.

---

## Features

* Classifies 5 hand gestures:

  * hello
  * one
  * peace
  * thumbsup
  * yes
* Real-time webcam prediction
* Custom dataset created from videos
* Lightweight CNN model

---

## Model Details

* Input: 96 × 96 grayscale images
* Architecture:

  * 3 Convolutional layers + MaxPooling
  * Fully connected layer with Dropout
  * Softmax output (5 classes)

---

## Dataset

The dataset consists of videos grouped by gesture:

```
data/
  ├── hello/
  ├── one/
  ├── peace/
  ├── thumbsup/
  └── yes/
```

Each video is converted into 30 grayscale frames for training.

* Dataset link (Google Drive):
https://drive.google.com/drive/folders/1GxDVIY6IWtaV9YMPw_r_cJn1m1bLqc49?usp=sharing

---

## Installation

Install required libraries:

```
pip install -r requirements.txt
```

---

## Run the Demo

```
python src/demo.py
```

Make sure your webcam is connected.

---

## 📊 Results

* Training Accuracy: ~99%
* Validation Accuracy: ~99%

---

## ⚠️ Notes

* Performance may vary depending on lighting and background
* Ensure your hand is inside the detection box during live testing
* Dataset is not included in the repository (download separately)

---

## 👤 Author

Vansh Vaibhav
