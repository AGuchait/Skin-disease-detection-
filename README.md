# Skin-disease-detection-
# 🩺 DermaCare AI — Skin Disease Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit)
![EfficientNet](https://img.shields.io/badge/EfficientNet-B0-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered clinical skin disease detection system built using deep learning.**  
Detects 10 skin conditions entirely offline — no internet, no API, no data sharing.

</div>

---

## 📸 Screenshots

| Home Page | Analysis Results |
|-----------|-----------------|
| Upload skin image and run AI analysis | Detailed diagnosis with confidence scores, symptoms & treatment |

---

## 🧠 About The Project

**DermaCare AI** is a final year academic project that uses a deep learning model (EfficientNetB0) trained on thousands of dermatology images to detect and classify skin diseases from uploaded photos.

The application runs **100% locally on your machine** using Streamlit — no internet connection, no external API, and no data is ever sent to any server.

### Key Highlights

- 🔬 Detects **10 skin disease categories** with confidence scoring
- 📊 Shows **Top 3 predictions** with probability bars
- 🩺 Lists **clinical symptoms** detected for each condition
- 💊 Provides **treatment recommendations** with urgency rating
- 🔍 Shows **severity score** (1–10) with visual meter
- 🔒 **100% offline** — your images never leave your computer
- ⏹️ **Stop button** to cancel analysis at any time

---

## 🩻 Detectable Skin Conditions

| # | Condition | Severity |
|---|-----------|----------|
| 1 | Eczema | Moderate |
| 2 | Melanoma | Severe |
| 3 | Atopic Dermatitis | Moderate |
| 4 | Basal Cell Carcinoma | Severe |
| 5 | Melanocytic Nevi (Moles) | Mild |
| 6 | Benign Keratosis | Mild |
| 7 | Psoriasis | Moderate |
| 8 | Seborrheic Keratoses | Mild |
| 9 | Tinea Ringworm | Mild |
| 10 | Warts Molluscum | Mild |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **TensorFlow / Keras** | Deep learning framework |
| **EfficientNetB0** | CNN architecture for image classification |
| **Streamlit** | Web interface (runs locally) |
| **Pillow (PIL)** | Image loading and preprocessing |
| **NumPy** | Numerical operations and array processing |

---

## 📁 Project Structure

```
DermaCare-AI/
│
├── app_streamlit.py                 # Main application file
├── skin_disease_model_best.h5       # Trained EfficientNet model (download separately)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/dermacare-ai.git
cd dermacare-ai
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add the trained model

Download the trained model file `skin_disease_model_best.h5` and place it in the **root folder** of the project (same folder as `app_streamlit.py`).

```
dermacare-ai/
├── app_streamlit.py
└── skin_disease_model_best.h5    ← place it here
```

### Step 4 — Run the application

```bash
streamlit run app_streamlit.py
```

The app will automatically open in your browser at **http://localhost:8501** 🎉

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
streamlit>=1.28.0
tensorflow>=2.12.0
numpy>=1.24.0
Pillow>=10.0.0
```

Install all at once:

```bash
pip install streamlit tensorflow numpy Pillow
```

---

## 🚀 How To Use

1. **Open the app** by running `streamlit run app_streamlit.py`
2. **Go to Home page** using the navigation menu
3. **Upload a skin image** (JPG, PNG, or JPEG)
4. **Click "🔬 Run Analysis"** to start detection
5. **View results** across 4 tabs:
   - 📊 **Predictions** — Top 3 diagnoses with confidence bars
   - 🩺 **Symptoms** — Detected clinical symptoms + urgency
   - 💊 **Treatment** — Step-by-step treatment recommendations
   - 🔍 **Details** — Severity score and condition overview
6. Click **"⏹️ Stop"** at any time to cancel the analysis

---

## 🏗️ Model Architecture

- **Base Model:** EfficientNetB0 pretrained on ImageNet
- **Input Size:** 224 × 224 × 3 (RGB)
- **Preprocessing:** EfficientNet standard preprocessing (`preprocess_input`)
- **Output:** Softmax over 10 disease classes
- **Training:** Fine-tuned on the Kaggle Skin Diseases Image Dataset
- **Hardware Used:** NVIDIA GTX 1650 (4GB VRAM), Intel i5 10th Gen, 8GB RAM

---

## 📊 Dataset

This model was trained on the **Skin Diseases Image Dataset** from Kaggle.

🔗 [Kaggle Dataset — Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)

---

## ⚠️ Medical Disclaimer

> **This application is for informational and educational purposes only.**  
> It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.  
> Always consult a qualified and licensed **dermatologist** for any skin-related concerns.  
> Do not make medical decisions based solely on the output of this AI system.

---

## 👥 Team — Group B3

| Name | Role |
|------|------|
| **Arnab Guchait** | Python · TensorFlow · Deep Learning |
| **Atanu Gayen** | Python · NumPy · Data Processing |
| **Avijit Kushwaha** | Streamlit · UI Design · Python |
| **Aryan Maity** | EfficientNet · Keras · Model Training |
| **Hrithika Chakraborty** | Dataset Curation · PIL · Image Processing |
| **Hiranmay Patra** | Model Evaluation · TensorFlow · Testing |
| **Ishan Sanyal** | Documentation · Python · Deployment |

**Department:** Computer Science & Engineering  
**Academic Year:** 2025 – 2026  
**Project Type:** Final Year Project

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

## 🤝 Acknowledgements

- [Kaggle](https://www.kaggle.com) — for the skin disease image dataset
- [TensorFlow](https://www.tensorflow.org) — deep learning framework
- [Streamlit](https://streamlit.io) — for the amazing web app framework
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019

---

<div align="center">
  <strong>DermaCare AI</strong> · Group B3 · Final Year Project 2026<br/>
  Built with ❤️ using Python, TensorFlow & Streamlit<br/><br/>
  ⭐ Star this repo if you found it helpful!
</div>
