# VisionAI 

**VisionAI : AI powered Video Surveillance Assistant** is an AI-powered tool that automatically identifies and tracks individuals and vehicles in CCTV footage. Using the **Buffalo model** for facial recognition and optical methods for license plate detection, it provides accurate results even from low-quality surveillance videos.

---

## 🔧 Features

- ✅ Face recognition using **Buffalo_L** model (via DeepFace)
- ✅ Vehicle detection through license plate recognition
- ✅ Extracts timestamps, snapshots, and presence durations
- ✅ Works with low- and high-resolution footage
- ✅ Local processing ensures data privacy

---

## ▶️ Usage
- Place a reference image in data/reference.jpg

- Place the target video in data/sample_video.mp4

Run the application:

- bash
- Copy
- Edit
- python run.py
- Results (timestamps, matched frames) will appear in the outputs/ folder.


## ⚠️ Disclaimer
This project is intended for educational use only. Ensure legal compliance before using facial recognition or surveillance technologies in real-world environments.

