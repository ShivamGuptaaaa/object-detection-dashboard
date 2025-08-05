# main.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from ultralytics import YOLO
import pandas as pd
import datetime
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit configuration
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("🎯 Real-Time Object Detection using YOLOv8")

# Step 1: User input
user_name = st.text_input("Enter your name:", value="guest").strip().lower().replace(" ", "_")

if user_name:
    log_folder = "detection_logs"
    os.makedirs(log_folder, exist_ok=True)

    # Step 2: Load YOLOv8 model
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file `{model_path}` not found. Please upload it to the app directory.")
        st.stop()

    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    # Step 3: Start detection
    if st.button("🚀 Start Live Detection"):
        st.warning("📷 Detection started! A webcam window will open. Press 'Q' to quit.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access the webcam.")
            st.stop()

        df = pd.DataFrame(columns=["Timestamp", "Object", "Confidence"])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_name}_{timestamp}_log.csv"
        file_path = os.path.join(log_folder, filename)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                label = model.names[int(cls)]
                confidence = round(conf, 2)
                time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Append detection
                df = pd.concat([
                    df,
                    pd.DataFrame([[time_now, label, confidence]], columns=df.columns)
                ], ignore_index=True)

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("YOLOv8 Detection - Press Q to Quit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        df.to_csv(file_path, index=False)
        st.success(f"✅ Detection log saved: `{filename}`")

        latest_file = file_path  # ✅ Used in your summary section

        # ----------------------------
        # ✅ Your Perfect KPI and Dashboard Code
        # ----------------------------

        # Summary KPIs
        st.subheader("📌 Summary Metrics")
        col1, col2 = st.columns(2)
        col1.metric("🧾 Total Detections", len(df))
        col2.metric("🔍 Unique Objects", df["Object"].nunique())

        # Bar Chart - Object Count
        st.subheader("📦 Object Detection Count")
        object_count = df["Object"].value_counts().reset_index()
        object_count.columns = ["Object", "Count"]
        
        fig1, ax1 = plt.subplots(figsize=(4.5, 3))  # Compact bar chart
        sns.barplot(data=object_count, x="Object", y="Count", palette="viridis", ax=ax1)
        ax1.set_title("Detected Objects", fontsize=10)
        ax1.set_xlabel("Object", fontsize=8)
        ax1.set_ylabel("Count", fontsize=8)
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)
        
        # Pie Chart - Improved
        st.subheader("🥧 Object Detection Distribution")
        fig2, ax2 = plt.subplots(figsize=(4, 3))  # Compact pie chart
        colors = plt.cm.Pastel1.colors
        explode = [0.05] * len(object_count)
        
        wedges, texts, autotexts = ax2.pie(
            object_count["Count"],
            labels=object_count["Object"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=explode,
            wedgeprops={"edgecolor": "black"},
            textprops={'fontsize': 8}
        )
        ax2.set_title("Object Share", fontsize=10)
        plt.tight_layout()  # Prevent overlapping
        st.pyplot(fig2)
        
        # 🆕 Heatmap - Object Frequency per Hour
        st.subheader("🕒 Object Detection by Hour (Heatmap)")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        heat_data = df.groupby(['Hour', 'Object']).size().unstack(fill_value=0)
        
        fig3, ax3 = plt.subplots(figsize=(5, 3))  # Compact heatmap
        sns.heatmap(heat_data, cmap='coolwarm', annot=True, fmt='d', linewidths=.5, ax=ax3, cbar=False)
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
        - 🔢 **{unique_objects} unique objects** were identified during this session.
        - 🥇 The most detected object is **'{most_detected}'**.
        - ⏰ The peak detection time was around **{peak_hour}:00 hrs**.
        - 📁 Data source file: `{os.path.basename(latest_file)}`
        """

        st.markdown(summary_text)
        st.success("📊 Dashboard generation complete — Great job, Detective! 🕵️")
        st.info("✨ Scroll up to explore the complete insights.")
