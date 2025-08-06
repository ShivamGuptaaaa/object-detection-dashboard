import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import pandas as pd
from collections import defaultdict
import time
import threading

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Global detection counter
detection_counts = defaultdict(int)
lock = threading.Lock()

st.set_page_config(page_title="YOLOv8 Real-Time Object Detection", layout="wide")
st.title("📸 YOLOv8 Real-Time Object Detection with Streamlit + Webcam")

st.markdown(
    """
This app uses [YOLOv8](https://github.com/ultralytics/ultralytics) for real-time object detection via your webcam.
"""
)

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


# Start webcam stream
ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Live KPIs and chart
placeholder = st.empty()

def display_kpi():
    while True:
        with lock:
            top_items = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(top_items, columns=["Object", "Count"])

        with placeholder.container():
            st.subheader("📊 Detection Summary")
            kpi1, kpi2, kpi3 = st.columns(3)

            total_detections = sum(detection_counts.values())
            top_object = top_items[0][0] if top_items else "N/A"
            top_count = top_items[0][1] if top_items else 0

            kpi1.metric("Total Detections", total_detections)
            kpi2.metric("Most Detected", top_object)
            kpi3.metric("Top Count", top_count)

            st.bar_chart(df.set_index("Object"))

        time.sleep(3)

# Launch KPI updater in background
thread = threading.Thread(target=display_kpi, daemon=True)
thread.start()
