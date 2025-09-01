# Gait-Based Streetlight Controller: AI & IoT Powered Smart Lighting

Welcome to the **Gait-Based Streetlight Controller**!
An innovative **AI-driven, context-aware smart streetlight system** that leverages **real-time human gait analysis** for adaptive, energy-efficient, and safer urban lighting.

---

## 💡 Overview

Traditional motion-sensor streetlights only detect movement, not *how* people move.
This project goes beyond by implementing an advanced computer vision system that:

* **Analyzes walking patterns, speed, and unsteadiness**
* **Adapts the intensity, color, and behavior of streetlights** in real-time (single pedestrians, groups, people with mobility issues, etc.)
* **Runs on edge devices** (Raspberry Pi + ESP32 + AI Accelerators) for privacy and low latency
* **Enhances both energy efficiency and public safety**—especially for vulnerable populations

---

## 🚀 Features

* 🧍 **Real-Time Gait Analysis** – Markerless pose estimation using YOLO/MediaPipe
* 🌈 **Adaptive Lighting Control** – Smart response to different walking styles
* ⚡ **Edge AI** – On-device inference for low latency
* 🔒 **Privacy by Design** – No raw video leaves the device
* 🌐 **IoT & Mesh Networking** – Coordinated multi-streetlight operation

---

## 🛠️ Hardware

* **Raspberry Pi** (AI processing & orchestration)
* **ESP32** (Local control, PWM for LEDs)
* **USB Camera / Pi Camera** (Real-time video feed)
* **Google Coral Edge TPU \[Optional]** (Accelerated AI)
* **RGB LED Streetlight Units**
* **Sensors** – Ambient light, PIR/motion, others as needed

---

## ⚙️ Software

* **Python** for control and analytics
* **Ultralytics YOLO, MediaPipe Pose** – Pose estimation
* **OpenCV, NumPy** – Vision & signal processing
* **Pandas** – Logging and analysis
* **MQTT / Lightweight IoT Protocols** – Networking (future)

---

## 📦 Setup & Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AJAY260505/gait-based-streetlight.git
   cd gait-based-streetlight
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights**
   Get YOLO pose estimation weights (e.g., `yolov8n-pose.pt`) and place them in the project folder.

4. **Connect Camera & ESP32/LED**
   Ensure your camera is connected and ESP32 is flashed to receive lighting instructions.

5. **Run the System**

   ```bash
   python main.py
   ```

6. **Visualize Gait Analysis**

   * System will display detected persons, gait classification (`SLOW`, `FAST`, `UNSTABLE`)
   * Lighting control signals streamed to ESP32

7. **Extend / Customize**

   * Adjust thresholds
   * Add new sensors/modules
   * Conduct user studies

---

## 📊 Project Structure

```
gait-based-streetlight/
├── main.py                   # Main application for detection & control
├── requirements.txt          # Python dependencies
├── multiperson_gait_log.csv  # Example output log
└── [other scripts/notebooks]
```

---

## 📝 Publications & Recognition

Prepared for submission to:

* **IEEE ISC2 (Smart Cities)**
* **CVPR**
* **International Journal of Sustainable Energy**
* Patent filing in India & other jurisdictions

👉 See our *[project report and presentation](...)* for technical details.

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repo and submit a PR
* Open issues for bugs, ideas, or enhancements

---

## 📄 License

This repository is licensed for **research and educational use**.
See [LICENSE](LICENSE) for details.

---

## 👥 Authors

* Ajay J ([@ajay260505](https://github.com/ajay260505))
* Darunesh R
* Jayanthmoulee C

---

## 🙏 Acknowledgements

Thanks to our mentors, institution, and the open-source community—especially contributors to **Ultralytics YOLO, MediaPipe, and OpenCV**.

---

✨ *For questions, collaboration, or demo requests, please open an issue or contact the maintainers.*

---
