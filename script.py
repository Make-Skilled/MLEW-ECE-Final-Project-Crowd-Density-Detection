import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load YOLO model
WEIGHTS_PATH = "yolov4.weights"
CONFIG_PATH = "yolov4.cfg"
NAMES_PATH = "coco.names"

# Load YOLO
net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

st.title("Crowd Density Estimation with YOLO + OpenCV")
uploaded_file = st.file_uploader("Upload a crowd image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    height, width, _ = img_cv.shape
    blob = cv2.dnn.blobFromImage(img_cv, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)
    
    class_ids, confidences, boxes = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    person_count = len(indices)
    
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_cv, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    st.image(img_cv, caption=f"People Count: {person_count}", use_column_width=True)
