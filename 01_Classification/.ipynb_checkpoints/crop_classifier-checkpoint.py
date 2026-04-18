import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Load Dataset ---
df = pd.read_csv('datasets/crop_recommendation/Crop_recommendation.csv')

print("Dataset Preview:")
print(df.head())

# --- 2. Data Preprocessing ---
# Features: N, P, K, temperature, humidity, ph, rainfall
X = df.drop('label', axis=1) 
y = df['label']

# Scaling is CRITICAL for KNN and SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 3. Define and Train Models ---
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', random_state=42),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5)
}

results = {}

print("\n--- Training Models ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

# --- 4. Visualization for Project Report ---

# Accuracy Comparison Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='magma')
plt.title('Comparison of Classification Algorithms')
plt.ylabel('Accuracy Score')
plt.ylim(0.8, 1.0) # Zoom in to see differences
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')
plt.show()

# --- 5. Best Model Deep-Dive (Random Forest usually wins here) ---
best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)

plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()