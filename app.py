import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import threading

# Load a highly accurate YOLOv8 Large model
model = YOLO("yolov8l.pt")  # Upgraded to YOLOv8 Large for superior detection

st.title("🔥 Ultra-Accurate Crowd Detection with YOLOv8")
st.write("Detect and count people in an image or live webcam stream.")

# Select mode
mode = st.radio("Choose Mode", ["📸 Image Upload", "🎥 Live Webcam"])

# ------------------- IMAGE UPLOAD PROCESSING -------------------
if mode == "📸 Image Upload":
    uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Apply CLAHE (Adaptive Histogram Equalization) for better contrast
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Resize for better YOLOv8 processing
        img_resized = cv2.resize(img_cv, (640, 640))

        # Run YOLOv8 detection with high confidence threshold
        results = model(img_resized, conf=0.7, iou=0.4)  

        people_count = 0
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()
                if class_id == 0 and confidence > 0.7:  
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_cv, f"Person {confidence:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert back to RGB for display
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # Display result
        st.image(img_cv, caption=f"👥 People Count: {people_count}", use_column_width=True)
        st.success(f"✅ Detected {people_count} people with maximum accuracy.")

# ------------------- LIVE WEBCAM DETECTION -------------------
elif mode == "🎥 Live Webcam":
    st.warning("Press 'Start Webcam' to begin real-time crowd detection.")

    stop_thread = False

    def run_webcam():
        global stop_thread
        cap = cv2.VideoCapture(0)  # Open webcam

        while cap.isOpened() and not stop_thread:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠ Webcam not accessible.")
                break

            # Preprocess the frame (CLAHE for better detection)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Run YOLOv8 detection in real-time
            results = model(frame, conf=0.7, iou=0.4)  

            people_count = 0
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = box.conf[0].item()
                    if class_id == 0 and confidence > 0.7:
                        people_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display result on Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption=f"Live Count: {people_count}", use_column_width=True)

        cap.release()

    if st.button("▶ Start Webcam"):
        stop_thread = False
        threading.Thread(target=run_webcam).start()

    if st.button("⏹ Stop Webcam"):
        stop_thread = True
