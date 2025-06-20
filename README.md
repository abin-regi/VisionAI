# VisionAI

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

> ⚡ AI-powered surveillance tool for detecting, tracking, and logging people and vehicles in video footage using facial recognition and license plate detection.

---

## ✨ Overview

**VisionAI** is an intelligent, privacy-friendly surveillance solution designed to analyze video footage in real-time or post-recording. It uses state-of-the-art **Buffalo facial recognition** and **license plate reading** to identify targets with precision — even in low-quality or obstructed frames.

🔐 **Local-first** design ensures data privacy  
🧠 Powered by **DeepFace + Buffalo_L**, **OpenCV**, and **OCR**  
🎯 Ideal for security, investigation, forensics, and traffic monitoring

---

## 🚀 Features

- 🧑‍💼 Facial recognition using **Buffalo_L** model
- 🚘 Vehicle tracking via license plate detection
- 🕒 Timestamped logs with frame snapshots
- 💡 Handles low-light and cluttered scenes
- 📁 Works with standard video formats (MP4, AVI, etc.)
- ⚙️ Offline processing with clean UI/CLI options

---

## 📂 Folder Structure

📦 VisionAI
├── app/ → Core modules (face, plate, video)
├── data/ → Input video & reference image
├── models/ → (Buffalo model goes here)
├── outputs/ → Snapshots and logs
├── run.py → Main script
└── README.md

---

## ⚙️ Getting Started

### 🧾 Prerequisites
> Python 3.8 or higher is recommended

### 📥 Setup
**Clone the Repository**  
git clone https://github.com/your-username/VisionAI.git
cd VisionAI
**Add Buffalo Model Files**  
Download from:  
🔗 [InsightFace Model Zoo – Buffalo_L](https://github.com/deepinsight/insightface/wiki/Model-Zoo)  

Place the downloaded folder like this:
models/
└── buffalo_l/
├── model-symbol.json
├── model-0000.params
└── ... (other files)

**Prepare Input**  
Place:  
- Reference image → `data/reference.jpg`  
- Video file → `data/sample_video.mp4`

### ▶️ Run the App
> python run.py
🖼️ Results will be saved in the `outputs/` folder, including:  
- Timestamp logs  
- Snapshot images  
- Matched identities/plates  

---

## 🧪 Example Use Cases
- 👮‍♂️ Law enforcement: Track suspects or stolen vehicles  
- 🏢 Corporate security: Identify unauthorized access  
- 🧾 Forensics: Build movement timelines  
- 🛣️ Traffic: Detect vehicles via license plates  
- 🎟️ Events/Retail: Spot VIPs or track crowd flow  

---

⚠️ **Disclaimer**  
This project is for educational use only. Ensure ethical and legal compliance before deploying in real-world scenarios involving surveillance or biometric analysis.
