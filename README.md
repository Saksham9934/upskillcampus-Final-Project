🚀 Skin Cancer Detection and Classification using YOLOv8
AI-Powered Medical Image Analysis using Deep Learning
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/> <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge"/> <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch"/> <img src="https://img.shields.io/badge/Dataset-ISIC%202020-success?style=for-the-badge"/> <img src="https://img.shields.io/github/license/Saksham9934/upskillcampus-Final-Project?style=for-the-badge"/> <img src="https://img.shields.io/github/stars/Saksham9934/upskillcampus-Final-Project?style=for-the-badge"/> <img src="https://img.shields.io/github/forks/Saksham9934/upskillcampus-Final-Project?style=for-the-badge"/> </p>
📖 Project Overview

Skin cancer is among the most common forms of cancer worldwide. Early diagnosis plays a vital role in improving survival rates and reducing treatment complexity.

This project presents an AI-powered Skin Cancer Detection and Classification System developed using the YOLOv8 deep learning model. The system analyzes dermoscopic images from the ISIC 2020 dataset and accurately classifies skin lesions as Benign or Malignant.

The model achieves excellent performance while maintaining high inference speed, making it suitable for future real-time clinical applications.

🎯 Objectives
Detect skin cancer lesions automatically
Classify lesions into Benign and Malignant
Assist dermatologists with AI-based diagnosis
Improve early detection accuracy
Reduce manual diagnostic effort
✨ Features
🧠 Deep Learning Based Detection
⚡ Real-Time Prediction
🎯 High Classification Accuracy
🏥 Medical Image Analysis
📊 Performance Evaluation
📈 Training Visualization
🔍 Binary Classification
📱 Scalable for Clinical Applications
🛠 Tech Stack
Category	Technology
Language	Python
Framework	PyTorch
Model	YOLOv8
Dataset	ISIC 2020
Notebook	Jupyter Notebook
Visualization	Matplotlib
Data Processing	NumPy, Pandas
Deployment Ready	Google Colab / Local
🧬 Dataset
ISIC 2020 Challenge Dataset

The project uses the International Skin Imaging Collaboration (ISIC 2020) dataset.

Dataset Statistics
Item	Value
Images	23,126
Classes	2
Benign	✓
Malignant	✓
Image Size	256 × 256
🔄 Workflow
ISIC Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Image Preprocessing
      │
      ▼
Data Augmentation
      │
      ▼
YOLOv8 Training
      │
      ▼
Validation
      │
      ▼
Testing
      │
      ▼
Prediction
      │
      ▼
Benign / Malignant
🖼 Model Architecture
Input Image
      │
      ▼
Image Resize (256x256)
      │
      ▼
YOLOv8 Backbone
      │
      ▼
Feature Extraction
      │
      ▼
Neck
      │
      ▼
Detection Head
      │
      ▼
Classification
      │
      ▼
Benign / Malignant
⚙ Data Preprocessing
Image Resizing (256 × 256)
Pixel Normalization
Horizontal Flip
Vertical Flip
Rotation
Zoom
Data Augmentation
🧠 Model

YOLOv8

YOLOv8 (You Only Look Once Version 8) is a state-of-the-art object detection and classification model developed by Ultralytics.

Advantages
High Accuracy
Faster Inference
Lightweight
Real-Time Detection
Better Generalization
⚙ Training Configuration
Parameter	Value
Epochs	20
Batch Size	32
Learning Rate	0.001
Framework	PyTorch
Optimizer	Adam
📈 Model Performance
Metric	Score
Accuracy	98.5%
Precision	0.98
Recall	1.00
F1 Score	0.99
📊 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
Support
📂 Project Structure
upskillcampus-Final-Project/

│── Dataset/
│── Training/
│── Testing/
│── Models/
│── Results/
│── notebooks/
│── requirements.txt
│── README.md
🚀 Installation
git clone https://github.com/Saksham9934/upskillcampus-Final-Project.git
cd upskillcampus-Final-Project
pip install -r requirements.txt
▶ Run Project
jupyter notebook

or

python predict.py
📷 Results

Add screenshots here.

Examples:

Dataset Samples
Training Curve
Confusion Matrix
Accuracy Graph
Prediction Output
📌 Future Improvements
Multi-Class Skin Cancer Detection
Deploy using Flask
Mobile Application
Streamlit Dashboard
Docker Deployment
Cloud Inference API
🤝 Contributing

Contributions are welcome.

Fork the repository
Create a new branch
Commit your changes
Push your branch
Open a Pull Request
🙌 Acknowledgements
UpSkill Campus
ISIC 2020 Challenge
Ultralytics YOLOv8
PyTorch Community
Open Source Contributors
📜 License

This project is licensed under the CC0 License.

👨‍💻 Author
Saksham Jha

📧 Email: sakshamjha3027@gmail.com

🐙 GitHub: https://github.com/Saksham9934

💼 LinkedIn: https://www.linkedin.com/in/saksham-jha-141623275/

<div align="center">
⭐ If you found this project useful, please consider giving it a Star!

Made with ❤️ using Python • PyTorch • YOLOv8

</div>
