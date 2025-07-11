import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model and class indices
model = load_model('resnet50_3dprint_defect4.h5')
with open('class_indices.pkl', 'rb') as f:
    inv_class_indices = pickle.load(f)

# Initialize video capture (use 0, 1, or your Iriun stream URL)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Pre-allocate buffer
img_buffer = np.zeros((1, 224, 224, 3), dtype=np.float32)

# Confidence threshold (adjust as needed)
CONFIDENCE_THRESHOLD = 70.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Focus on print area (ROI)
    height, width = frame.shape[:2]
    roi = frame[height//4:3*height//4, width//4:3*width//4]

    # Preprocessing
    resized = cv2.resize(roi, (224, 224))
    img_buffer[0] = resized
    img_buffer_pp = preprocess_input(img_buffer.copy())

    # Predict
    preds = model.predict(img_buffer_pp, verbose=0)
    print("Raw predictions:", preds[0])  # Debug output

    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id] * 100

    # Apply confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        label = inv_class_indices[class_id]
        color = (0, 255, 0)
        text = f"{label}: {confidence:.2f}%"
    else:
        label = "No printer detected"
        color = (0, 0, 255)
        text = label

    # Visual ROI
    cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (255, 0, 0), 2)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('3D Print Defect Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
