# 🌱 Agri-Intelligence Agent
### Multimodal Agentic AI Framework for Precision Agriculture

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

---

## 📌 Overview

**Agri-Intelligence Agent** is a **multimodal, agentic AI system** built for **precision agriculture**. It combines **computer vision, generative AI, and structured data reasoning** to deliver **real-time, actionable farming insights**.

The system simulates an **AI agronomist** capable of understanding plant health from images, analyzing soil nutrient data, and generating intelligent recommendations — all in a single unified pipeline.

---

## 🎯 Objectives

- Enable **data-driven farming decisions**
- Reduce crop loss through **early disease detection**
- Provide **AI-powered agronomy assistance**
- Build a **scalable multimodal AI system**

---

## 🚀 Key Features

### 🧠 Multimodal Perception
Processes leaf images (disease detection) alongside soil data (N, P, K values), combining visual and numerical inputs for richer, more context-aware decisions.

### 🤖 Agentic Reasoning Engine
Powered by **Gemini 1.5 Flash**, the engine performs context-aware reasoning, autonomous decision-making, and structured recommendation generation.

### 👁️ Computer Vision Module
A custom **CNN model** handles plant disease classification with real-time prediction capability.

### 📝 Generative AI Insights
Produces actionable outputs including crop recommendations, treatment suggestions, and preventive measures — all grounded in the multimodal inputs.

---

## 🏗️ System Architecture

```
User Input (Image + Soil Data)
         ↓
Computer Vision Model (CNN)
         ↓
Feature + Data Fusion Layer
         ↓
  Agentic AI (Gemini API)
         ↓
Decision + Recommendation Output
         ↓
      Streamlit UI
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SaishwariKatkam/Multimodal-Agentic-AI-for-Agri-Intelligence.git
cd Multimodal-Agentic-AI-for-Agri-Intelligence
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Multimodal-Agentic-AI-for-Agri-Intelligence/
├── app.py                  # Main Streamlit application
├── models/                 # Trained CNN models
├── datasets/               # Input datasets
├── utils/                  # Helper functions
├── requirements.txt        # Dependencies
├── .env                    # API keys (not pushed to GitHub)
└── README.md               # Project documentation
```

---

## 🧪 Sample Workflow

1. **Upload** a plant leaf image 🌿
2. **Enter** soil nutrient values (N, P, K)
3. **System analyzes** the combined inputs
4. **AI agent generates** structured insights

**Output includes:**
- 🔍 Disease prediction
- 🌾 Crop recommendation
- 💡 Actionable suggestions

---

## 📊 Tech Stack

| Category | Technology |
|---|---|
| Programming | Python 3.9+ |
| Frontend | Streamlit |
| Computer Vision | CNN (TensorFlow / PyTorch) |
| Generative AI | Gemini 1.5 Flash |
| Data Processing | Pandas, NumPy |

---

## 🔮 Future Roadmap

- [ ] Semantic Kernel integration for advanced agent orchestration
- [ ] Reinforcement Learning for adaptive irrigation policies
- [ ] Long-term memory using Vector Databases
- [ ] Real-time IoT sensor integration
- [ ] Mobile app deployment

---

## 🏆 Key Highlights

- ✅ Multimodal AI system (Vision + Structured Data)
- ✅ Agentic decision-making pipeline
- ✅ Real-world agriculture use case
- ✅ Scalable, modular architecture

---

<p align="center">Made with ❤️ for smarter farming 🌾</p>
