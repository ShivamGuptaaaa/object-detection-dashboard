import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import pandas as pd
from collections import defaultdict
import time
import threading

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Global detection counts
detection_counts = defaultdict(int)
lock = threading.Lock()

# Streamlit page config
st.set_page_config(page_title="YOLOv8 Real-Time Dashboard", layout="wide")
st.title("🎯 YOLOv8 Real-Time Object Detection Dashboard")

st.markdown("""
This app uses YOLOv8 with Streamlit WebRTC for real-time object detection using your webcam.
All detected objects will appear with bounding boxes and the dashboard below will display live analytics.
""")

# Define video processor
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)[0]
        boxes = results.boxes

        with lock:
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                detection_counts[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                text = f"{label} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start WebRTC stream
ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Dashboard section
placeholder = st.empty()

# Function to update KPIs

def update_dashboard():
    while True:
        with lock:
            top_items = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(top_items, columns=["Object", "Count"])

        with placeholder.container():
            st.markdown("---")
            st.subheader("📊 Real-Time Detection Dashboard")

            kpi1, kpi2, kpi3 = st.columns(3)

            total = sum(detection_counts.values())
            most_common = top_items[0][0] if top_items else "N/A"
            top_count = top_items[0][1] if top_items else 0

            kpi1.metric("Total Detections", total)
            kpi2.metric("Top Object", most_common)
            kpi3.metric("Top Count", top_count)

            st.markdown("### 📈 Detection Distribution")
            st.bar_chart(df.set_index("Object"))

        time.sleep(3)

# Start dashboard update thread
thread = threading.Thread(target=update_dashboard, daemon=True)
thread.start()
