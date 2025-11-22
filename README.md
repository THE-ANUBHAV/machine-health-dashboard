Machine Health Monitoring & Failure Prediction Dashboard
This repository contains a Predictive Maintenance Dashboard built with Streamlit and Python. It leverages machine learning to monitor industrial equipment health in real-time, predicting potential failures based on the AI4I 2020 Predictive Maintenance Dataset.

🚀 Project Overview
The application serves as a proactive decision-support tool for plant operators, transitioning maintenance strategies from reactive to predictive. By analyzing sensor data, it calculates the probability of machine failure and suggests actionable maintenance windows.

🧠 Machine Learning Models
The dashboard integrates four distinct pre-trained algorithms, allowing users to toggle between them to compare sensitivity and accuracy:

Random Forest Classifier: Robust ensemble method for high-variance data.

Gradient Boosting Classifier: Optimized for predictive precision.

Support Vector Machine (SVM): Effective for high-dimensional boundary detection.

MLP Neural Network: Captures non-linear relationships in sensor readings.

📊 Key Features
Live Risk Assessment: Inputs 6 key parameters (Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear, and Machine Quality Type) to output a Failure Probability (%).

Dynamic Status Indicators: Automatically categorizes machines into risk levels:

🟢 Safe (Prob ≤ 40%)

🟡 Medium Risk (40% < Prob ≤ 70%)

🔴 High Risk (Prob > 70% - Immediate Action)

Interactive Visualizations: Powered by Plotly, the interface displays simulated real-time sensor trends, failure cause distributions, and model performance metrics (ROC Curves, Confusion Matrices).

Maintenance Scheduler: Estimates Remaining Useful Life (RUL) based on tool wear heuristics and calculates potential cost savings.

🛠️ Technical Stack
Frontend: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn (Inference & Preprocessing)

Visualization: Plotly Express, Plotly Graph Objects

📦 Installation & Usage
Clone the repository and ensure the ai4i2020.csv dataset and .pkl model files are in the root directory.

Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

streamlit run app.py
