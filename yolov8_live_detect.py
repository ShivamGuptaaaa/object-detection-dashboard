from ultralytics import YOLO
import cv2
import pandas as pd
import datetime
import os

# 🚀 Ask for user name
user_name = input("Enter your name: ").strip().lower().replace(" ", "_")

# 📁 Create logs folder if it doesn’t exist
log_dir = "detection_logs"
os.makedirs(log_dir, exist_ok=True)

# 📄 File name with timestamp to avoid overwriting
timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"{user_name}_{timestamp_now}.csv"
csv_path = os.path.join(log_dir, csv_filename)

# 🧠 Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 🎥 Start webcam
cap = cv2.VideoCapture(0)  # Change to 1 if external camera

# 📊 DataFrame to log detections
df = pd.DataFrame(columns=["Timestamp", "Object", "Confidence"])

print("📸 Camera started. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 📝 Log detection
        df = pd.concat(
            [df, pd.DataFrame([[timestamp, label, round(conf, 2)]], columns=df.columns)],
            ignore_index=True
        )

        # 🖼️ Draw detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 🪟 Show camera output
    cv2.imshow("YOLOv8 Live Detection", frame)

    # ❌ Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 🛑 Cleanup
cap.release()
cv2.destroyAllWindows()

# 💾 Save CSV
df.to_csv(csv_path, index=False)
print(f"✅ Data saved to {csv_path}")
