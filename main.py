import cv2
import pandas as pd
from datetime import datetime
import os
from ultralytics import YOLO
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

warnings.filterwarnings("ignore", message="Examining the path of torch.classes raised")

st.set_page_config(page_title="Live Object Detection", layout="wide")
st.title("📸 Live Object Detection and Dashboard")

user_name = st.text_input("Enter your name:", "shivam")
filename = f"{user_name.lower().replace(' ', '_')}_detection_log.csv"

model = YOLO("yolov8n.pt")

if not os.path.exists(filename):
    pd.DataFrame(columns=["Timestamp", "Object", "Confidence"]).to_csv(filename, index=False)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, stream=True)

        for r in results:
            for box in r.boxes:
                cls = model.names[int(box.cls[0])]
                conf = round(float(box.conf[0]), 2)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Save to CSV
                new_data = pd.DataFrame([[timestamp, cls, conf]], columns=["Timestamp", "Object", "Confidence"])
                new_data.to_csv(filename, mode='a', header=False, index=False)

                # Draw box
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(img, f'{cls} {conf}', (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="object-detection", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})

# Dashboard Section
if os.path.exists(filename):
    df = pd.read_csv(filename)
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour

        st.subheader("📦 Object Detection Count")
        object_count = df["Object"].value_counts().reset_index()
        object_count.columns = ["Object", "Count"]

        fig1, ax1 = plt.subplots(figsize=(3, 2))
        sns.barplot(data=object_count, x="Object", y="Count", palette="viridis", ax=ax1)
        ax1.set_title("Detected Objects", fontsize=10)
        ax1.set_xlabel("Object", fontsize=8)
        ax1.set_ylabel("Count", fontsize=8)
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        st.subheader("🥧 Object Detection Distribution")
        fig2, ax2 = plt.subplots(figsize=(3, 2))
        colors = plt.cm.Pastel1.colors
        explode = [0.05] * len(object_count)

        ax2.pie(
            object_count["Count"],
            labels=object_count["Object"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=explode,
            wedgeprops={"edgecolor": "black"},
            textprops={'fontsize': 10}
        )
        ax2.set_title("Object Share", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2)

        st.subheader("🕒 Object Detection by Hour (Heatmap)")
        heat_data = df.groupby(['Hour', 'Object']).size().unstack(fill_value=0)

        fig3, ax3 = plt.subplots(figsize=(3, 2))
        sns.heatmap(heat_data, cmap='coolwarm', annot=True, fmt='d', linewidths=.5, ax=ax3)
        ax3.set_title("📊 Frequency of Objects by Hour", fontsize=10)
        ax3.set_xlabel("Object", fontsize=8)
        ax3.set_ylabel("Hour of Day", fontsize=8)
        st.pyplot(fig3)

        # ----------------------------
        # 📋 Final Summary Section
        # ----------------------------
        st.markdown("---")
        st.header("📋 Summary Report")

        most_detected = object_count.iloc[0]["Object"] if not object_count.empty else "N/A"
        peak_hour = df['Hour'].mode()[0] if not df.empty else "N/A"
        total_detections = len(df)
        unique_objects = df["Object"].nunique()

        summary_text = f"""
        - ✅ A total of **{total_detections} detections** were recorded.
        - 🔣 **{unique_objects} unique objects** were identified during this session.
        - 🥇 The most detected object is **'{most_detected}'**.
        - ⏰ The peak detection time was around **{peak_hour}:00 hrs**.
        - 📁 Data source file: `{os.path.basename(filename)}`
        """

        st.markdown(summary_text)
        st.success("📊 Dashboard generation complete — Great job, Detective! 🕵️")
        st.info("✨ Scroll up to explore the complete insights.")
