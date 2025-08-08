FaceFAST App 🎯

A real-time face recognition attendance system developed using Streamlit, InsightFace, and Redis, designed to streamline and automate attendance in educational institutions, workplaces, or any environment that demands contactless tracking.

🔗 Live Demo: https://facefast-app.streamlit.app

📖 Introduction

FaceFAST App is a secure and efficient face recognition attendance tracking system. It leverages computer vision and machine learning to detect, recognize, and log attendance in real-time using webcam input.

Developed as part of an M.Sc.Artificial intelligence and Machine learning project, it supports:

•	Seamless user registration

•	Live face recognition and attendance logging

•	Auto-generated daily reports with in/out time and status

🚀 Features

•	👤 User Registration: Via webcam with face embeddings stored securely in Redis.

•	🎥 Real-Time Face Recognition: Uses streamlit-webrtc for live webcam streaming.

•	📝 Attendance Logging: Logs time, calculates presence duration, and assigns attendance status.

•	📊 Reports Generation: View registered users, raw logs, and smart summaries in one place.

•	🌐 Cloud Hosted: Deployed on Streamlit Cloud with HTTPS.

•	🔐 Role-Based Identification: Supports Student and Teacher roles.

🧰 Tech Stack

Frontend - Streamlit, Streamlit WebRTC

Backend - Python, OpenCV, InsightFace, ONNX Runtime, NumPy

Database - Redis (in-memory storage), Pandas (report generation)

ML/Recognition - InsightFace, Scikit-learn (cosine similarity)

Deployment - Streamlit Cloud

🏗️ System Architecture

<img width="773" height="194" alt="image" src="https://github.com/user-attachments/assets/9dca3b73-eabd-44fb-98b7-211a25e5677f" />

⚙️ Installation

🔧 Note: Redis server and Python 3.8+ are required.

1.	Clone the Repository

```python
git clone https://github.com/yourusername/FaceFAST-App.git
cd FaceFAST-App
```

2.	Install Dependencies

```python
pip install -r requirements.txt
```

3.	Configure Redis (Optional)

Set up a Redis instance (local or remote) and update the credentials in
```python
face_rec.py
```
4.	Run the App

```python
streamlit run Home.py
```

🧪 Usage

The application consists of four primary pages:

1.	Home – Loads models and initializes Redis

2.	Registration Form – Registers users via webcam.

3.	Real-Time Prediction – Detects and logs faces live.

4.	Reports – View all registered users, logs, and attendance reports.

Navigation is built-in using Streamlit’s multipage functionality.

📜 License

This project is licensed under the MIT License. Feel free to use and modify for educational or non-commercial purposes.

👩💻 Developed By

Rabiya

M.Sc. Computer Science

St. Ann’s College for Women

2024–2025

