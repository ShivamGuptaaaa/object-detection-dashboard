import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Streamlit config
st.set_page_config(page_title="📊 Detection Dashboard", page_icon="🧠", layout="centered")
st.title("🧠 Object Detection Dashboard")

# Load latest CSV from detection_logs
log_folder = "detection_logs"
if not os.path.exists(log_folder):
    st.error("❌ 'detection_logs' folder not found. Please run detection app first.")
    st.stop()

csv_files = glob.glob(os.path.join(log_folder, "*.csv"))
if not csv_files:
    st.warning("⚠️ No detection logs available yet.")
    st.stop()

latest_file = max(csv_files, key=os.path.getctime)
st.success(f"✅ Loaded file: `{os.path.basename(latest_file)}`")
df = pd.read_csv(latest_file)

# Raw Data
with st.expander("📂 View Raw Data"):
    st.dataframe(df)

# Summary KPIs
st.subheader("📌 Summary Metrics")
col1, col2 = st.columns(2)
col1.metric("🧾 Total Detections", len(df))
col2.metric("🔍 Unique Objects", df["Object"].nunique())

# Bar Chart - Object Count
st.subheader("📦 Object Detection Count")
object_count = df["Object"].value_counts().reset_index()
object_count.columns = ["Object", "Count"]

fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.barplot(data=object_count, x="Object", y="Count", palette="viridis", ax=ax1)
ax1.set_title("Detected Objects", fontsize=14)
ax1.set_xlabel("Object", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)

# Pie Chart - Improved
st.subheader("🥧 Object Detection Distribution")
fig2, ax2 = plt.subplots(figsize=(5, 4))
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
    textprops={'fontsize': 10}
)
ax2.set_title("Object Share", fontsize=14)
plt.tight_layout()  # Prevent overlapping
st.pyplot(fig2)

# 🆕 Heatmap - Object Frequency per Hour
st.subheader("🕒 Object Detection by Hour (Heatmap)")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
heat_data = df.groupby(['Hour', 'Object']).size().unstack(fill_value=0)

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.heatmap(heat_data, cmap='coolwarm', annot=True, fmt='d', linewidths=.5, ax=ax3)
ax3.set_title("📊 Frequency of Objects by Hour", fontsize=14)
ax3.set_xlabel("Object", fontsize=12)
ax3.set_ylabel("Hour of Day", fontsize=12)
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
