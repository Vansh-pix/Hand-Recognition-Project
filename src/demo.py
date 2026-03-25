import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = r"model/sign_model_5class.h5"
IMG_SIZE = 96

CLASSES = ['hello', 'one', 'peace', 'thumbsup', 'yes']

model = load_model(MODEL_PATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    
    roi = frame[
        int(h*0.2):int(h*0.8),
        int(w*0.3):int(w*0.7)
    ]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    input_img = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    
    preds = model.predict(input_img, verbose=0)
    class_idx = np.argmax(preds)
    label = CLASSES[class_idx]
    confidence = np.max(preds)
    
    cv2.rectangle(
        frame,
        (int(w*0.3), int(h*0.2)),
        (int(w*0.7), int(h*0.8)),
        (0,255,0),
        2
    )
    
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )
    
    cv2.imshow("Sign Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()