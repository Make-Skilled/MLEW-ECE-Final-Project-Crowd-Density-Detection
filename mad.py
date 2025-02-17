import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model correctly
try:
    model = YOLO("yolov8x.pt")  # Load YOLOv8 Extra Large model
    st.success("✅ YOLOv8 model loaded successfully!")
except Exception as e:
    st.error(f"⚠ Error loading YOLOv8 model: {e}")
    st.stop()

# Streamlit UI
st.title("🔥 Ultra-Accurate Crowd Detection with YOLOv8")
st.write("Detect and count people in an image or live webcam stream with maximum accuracy.")
st.markdown("---")

# Select mode
mode = st.radio("Choose Mode", ["📸 Image Upload", "🎥 Live Webcam"])

# Function to process images and detect people
def process_image(image):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Store original image dimensions
    orig_h, orig_w = img_cv.shape[:2]

    # Resize image for YOLOv8 processing
    img_resized = cv2.resize(img_cv, (640, 640))

    # Run YOLOv8 detection
    results = model.predict(img_resized, conf=0.7, iou=0.4)

    people_count = 0
    for r in results:
        for i in range(len(r.boxes)):
            box = r.boxes.xyxy[i].cpu().numpy()  # Convert tensor to NumPy array
            class_id = int(r.boxes.cls[i])       # Extract class ID
            confidence = r.boxes.conf[i].item()  # Extract confidence score

            if class_id == 0 and confidence > 0.7:  # Class 0 is 'person'
                people_count += 1

                # Scale bounding box to original image size
                x1, y1, x2, y2 = (box * [orig_w / 640, orig_h / 640, orig_w / 640, orig_h / 640]).astype(int)

                # Draw bounding box
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cv, f"Person {confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert back to RGB for Streamlit display
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_cv, people_count

# ------------------- IMAGE UPLOAD MODE -------------------
if mode == "📸 Image Upload":
    uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        with st.spinner("🔍 Processing image..."):
            image = Image.open(uploaded_file)
            processed_image, people_count = process_image(image)

            # Display result
            st.image(processed_image, caption=f"👥 People Count: {people_count}", use_column_width=True)
            st.success(f"✅ Detected {people_count} people with maximum accuracy!")
            st.balloons()

# ------------------- LIVE WEBCAM DETECTION -------------------
elif mode == "🎥 Live Webcam":
    st.warning("Press 'Start Webcam' to begin real-time crowd detection.")

    # Initialize session state for webcam control
    if "stop_webcam" not in st.session_state:
        st.session_state.stop_webcam = False

    def start_webcam():
        st.session_state.stop_webcam = False
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("⚠ Webcam not accessible. Please check your camera settings.")
            return

        st_frame = st.empty()  # Placeholder for live video feed

        while cap.isOpened() and not st.session_state.stop_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠ Failed to capture frame from webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            processed_frame, people_count = process_image(pil_image)

            st_frame.image(processed_frame, caption=f"Live Count: {people_count}", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()

    # Start webcam button
    if st.button("▶ Start Webcam"):
        start_webcam()

    # Stop webcam button
    if st.button("⏹ Stop Webcam"):
        st.session_state.stop_webcam = True
        st.success("Webcam stopped. You can start it again anytime!")

# Footer
st.markdown("---")
st.markdown("🚀 **Inspiring Message:** Every great project starts with a single step. Keep innovating and building amazing things!")
