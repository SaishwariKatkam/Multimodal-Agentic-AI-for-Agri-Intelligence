import streamlit as st
import pandas as pd
import os
import sys
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

# Load the keys from .env
load_dotenv()

# Update your configuration line
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- 1. SYSTEM PATH SETUP ---
subfolder_path = os.path.join(os.getcwd(), "04_GenAI_Agentic")
if subfolder_path not in sys.path:
    sys.path.append(subfolder_path)

try:
    from agri_agent import AgriIntelligenceAgent
except ImportError:
    st.error("Missing agri_agent.py in the subfolder!")
    st.stop()

@st.cache_resource
def get_working_model():
    """Finds the first available model for your specific API key."""
    try:
        # Ask Google what we are allowed to use
        available_models = [m.name for m in genai.list_models() 
                           if 'generateContent' in m.supported_generation_methods]
        
        # Priority: 1.5-flash -> 1.5-pro -> Anything else
        for name in available_models:
            if "1.5-flash" in name: return name
        for name in available_models:
            if "1.5-pro" in name: return name
            
        return available_models[0] # Just take the first one if the above aren't found
    except Exception as e:
        # If discovery fails, return the standard production name
        return "gemini-1.5-flash"

model_to_use = get_working_model()
llm_expert = genai.GenerativeModel(model_to_use)

@st.cache_resource
def load_agri_tools():
    return AgriIntelligenceAgent()

agri_agent = load_agri_tools()

# --- 3. UI SETUP ---
st.set_page_config(page_title="Agri-Agent AI Hub", page_icon="👨‍🌾")
st.title("🚜 Smart Farmer's Diagnostic Hub")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🌱 Sensors")
    n = st.slider("Nitrogen (N)", 0, 150, 90)
    p = st.slider("Phosphorus (P)", 0, 150, 42)
    k = st.slider("Potassium (K)", 0, 150, 43)
    temp = st.number_input("Temp (°C)", value=25.0)
    hum = st.number_input("Humidity (%)", value=80.0)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
    rain = st.number_input("Rainfall (mm)", value=200.0)
    soil_inputs = [n, p, k, temp, hum, ph, rain]

# --- 5. MAIN CHAT ---
col1, col2 = st.columns([1, 1.3])

with col1:
    uploaded_file = st.file_uploader("Upload Leaf", type=["jpg", "png"])
    if uploaded_file:
        st.image(uploaded_file)
        with open("temp_diag.jpg", "wb") as f: f.write(uploaded_file.getbuffer())

with col2:
    chat_container = st.container(height=500)
    for msg in st.session_state.messages:
        with chat_container.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your farm..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user"): st.markdown(prompt)

        with chat_container.chat_message("assistant"):
            # Execute Tools
            disease = "No image"
            if uploaded_file:
                disease = agri_agent.predict_disease("temp_diag.jpg").replace('_', ' ')
            cluster, crop_rec = agri_agent.analyze_soil_and_crop(*soil_inputs)

            # Construct Expert Prompt
            context = f"""
            Role: Agronomist. 
            User says: {prompt}
            Data: Disease={disease}, Soil Zone={cluster}, Rec Crop={crop_rec}.
            Sensors: N={n}, P={p}, K={k}, Rain={rain}mm, pH={ph}.
            Instructions: Explain the NPK values and why drought/rain affects them. Use bullet points.
            """

            try:
                response = llm_expert.generate_content(context)
                ans = response.text
            except Exception as e:
                ans = f"Connection failed. Used Model: {model_to_use}. Error: {str(e)}"

            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})