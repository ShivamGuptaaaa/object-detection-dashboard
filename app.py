# app.py

from ultralytics import YOLO
import cv2
import pandas as pd
import datetime
import os

# Step 1: Get user name
user_name = input("Enter your name: ").strip().lower().replace(" ", "_")

# Step 2: Create detection_logs folder if not exists
log_folder = "detection_logs"
os.makedirs(log_folder, exist_ok=True)

# Step 3: Generate timestamped filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{user_name}_{timestamp}_log.csv"
file_path = os.path.join(log_folder, filename)

# Step 4: Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt for better accuracy

# Step 5: Open webcam (0 for internal, 1 for external)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Webcam not accessible.")
    exit()

# Step 6: Create DataFrame for logging
df = pd.DataFrame(columns=["Timestamp", "Object", "Confidence"])

print("🚀 Camera started. Press 'q' to quit.")

# Step 7: Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    results = model(frame)[0]  # Detect objects

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        confidence = round(conf, 2)
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to DataFrame
        df = pd.concat(
            [df, pd.DataFrame([[time_now, label, confidence]], columns=df.columns)],
            ignore_index=True
        )

        # Draw bounding box & label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Cleanup
cap.release()
cv2.destroyAllWindows()

# Step 9: Save CSV
df.to_csv(file_path, index=False)
print(f"✅ Detection log saved as: {file_path}")
