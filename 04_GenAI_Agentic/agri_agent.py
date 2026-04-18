import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
import numpy as np
import os
import pandas as pd

# 1. CNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.conv(x)

class AgriIntelligenceAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define Classes
        self.disease_classes = [
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
            'Tomato_healthy'
        ]

        # Setup Paths
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_path, '..', 'models')

        # Load CNN
        cnn_path = os.path.join(models_dir, 'Simple_CNN_plant_model.pth')
        self.disease_model = SimpleCNN(len(self.disease_classes))
        self.disease_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.disease_model.to(self.device)
        self.disease_model.eval()

        # Load Pickle Models
        self.soil_cluster_model = self._safe_load(models_dir, 'soil_cluster_model.pkl')
        self.soil_scaler = self._safe_load(models_dir, 'soil_scaler.pkl')
        self.crop_model = self._safe_load(models_dir, 'crop_rf_model.pkl')
        self.crop_scaler = self._safe_load(models_dir, 'crop_scaler.pkl')

    def _safe_load(self, directory, filename):
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            print(f"--- WARNING: File not found: {filename} ---")
            return None
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            # This helps us see if we loaded a class or an instance
            print(f"INFO: Loaded {filename} ({type(obj).__name__})")
            return obj

    def predict_disease(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((128, 128)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.disease_model(image)
            _, predicted = torch.max(output, 1)
        return self.disease_classes[predicted.item()]

    def analyze_soil_and_crop(self, n, p, k, temp, hum, ph, rain):
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=feature_names)
        
        # 1. Soil Clustering
        try:
            scaled_soil = self.soil_scaler.transform(input_df)
            cluster = self.soil_cluster_model.predict(scaled_soil)[0]
        except Exception as e:
            print(f"Clustering Error: {e}")
            cluster = "N/A"
        
        # 2. Crop Recommendation
        try:
            if self.crop_scaler:
                X_input = self.crop_scaler.transform(input_df)
            else:
                X_input = input_df
            
            # Using the instance to predict
            recommendation = self.crop_model.predict(X_input)[0]
            
        except Exception as e:
            print(f"Crop Prediction Error: {e}")
            recommendation = "Unknown"
            
        return cluster, recommendation

    def generate_report(self, image_path, soil_inputs):
        disease = self.predict_disease(image_path)
        cluster, crop = self.analyze_soil_and_crop(*soil_inputs)
        print("\n" + "="*40 + "\n      AGRI-INTELLIGENCE REPORT\n" + "="*40)
        print(f"Leaf Diagnosis : {disease}")
        print(f"Soil Category  : Cluster {cluster}")
        print(f"Recommendation : Plant '{crop.upper()}'\n" + "="*40 + "\n")

if __name__ == "__main__":
    agent = AgriIntelligenceAgent()
    
    # Path setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_img = os.path.join(base_dir, '..', 'test_data', 'test_leaf.jpg')
    
    sample_soil = [120, 250, 43, 20.87, 82.00, 6.5, 202.93]
    
    if os.path.exists(test_img):
        agent.generate_report(test_img, sample_soil)
    else:
        print(f"Test image missing at: {test_img}")