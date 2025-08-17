Model for Predicting ADL Disability Risk in Patients with Low Back Pain
This is a machine learning-based Streamlit web application for predicting ADL (Activities of Daily Living) disability risk in patients with low back pain.

Features
🎯 Intelligent Prediction: Predicts ADL disability risk based on SVM model
📊 Visual Explanation: Uses SHAP waterfall charts to explain prediction results
🌐 English Interface: Fully English user interface
📱 Responsive Design: Supports access from various devices
🔍 Feature Importance: Detailed display of each variable's impact on predictions

Installation and Running
Method 1: Using Startup Script (Recommended)
# Navigate to streamlit directory
cd streamlit

# Run startup script
python run_app.py
Method 2: Direct Streamlit Command
# Navigate to streamlit directory
cd streamlit

# Install dependencies
pip install -r requirements.txt

# Start application
streamlit run app.py
Method 3: Using Virtual Environment
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Unix/Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start application
streamlit run app.py
System Requirements
Python 3.8+
Memory: At least 2GB available RAM
Storage: At least 100MB available space
Dependencies
streamlit >= 1.36
pandas >= 1.5
numpy >= 1.24
scikit-learn >= 1.2
joblib >= 1.2
shap >= 0.43
matplotlib >= 3.7
plotly >= 5.0
Usage
Start Application: Run any of the startup commands above
Input Data: Enter 17 clinical variables in the form
Get Prediction: Click "Start Prediction" button
View Results: Check prediction category, risk probability, and SHAP explanation chart
Understand Results: Use the instructions section to help understand prediction results
Input Variable Description
The application requires input of the following 17 variables:

Demographics
Gender: Female(0) / Male(1)
Age: 45-85 years
Education Level: Below high school(1) / High school and above(2)
Medical History
Drinking Status: No(0) / Yes(1)
Diabetes: No(0) / Yes(1)
Stroke: No(0) / Yes(1)
Kidney Disease: No(0) / Yes(1)
Emotional Problems: No(0) / Yes(1)
Memory-related Diseases: No(0) / Yes(1)
Asthma: No(0) / Yes(1)
Falls: No(0) / Yes(1)
Depression: No(0) / Yes(1)
Lifestyle
Life Satisfaction: Low(1) / Medium(2) / High(3)
Self-rated Health Status: Poor(1) / Fair(2) / Good(3)
Physical Activity Level: Low(1) / Medium(2) / High(3)
Number of Hospitalizations: ≥0
Internet Participation: No(0) / Uses internet(1)
Output Result Interpretation
Prediction Category
0: No ADL disability risk
1: ADL disability risk exists
Risk Probability
Numerical value between 0-1, closer to 1 indicates higher risk
Displayed as percentage
SHAP Explanation Chart
Shows each feature's contribution to prediction results
Positive values: Increase risk
Negative values: Decrease risk
Larger absolute values indicate more significant impact
Important Notes
⚠️ Important Reminder:

This tool provides data-driven estimates and cannot replace professional medical advice
The model is based on training data, and practical application requires clinical judgment
Age and number of hospitalizations are automatically standardized
All categorical variables are encoded as numerical values
Troubleshooting
Common Issues
Dependency Package Installation Failure

pip install --upgrade pip
pip install -r requirements.txt
Port Occupied

streamlit run app.py --server.port 8502
Insufficient Memory

Close other applications
Reduce SHAP background sample count (modify in code)
File Path Error

Ensure running in streamlit directory
Check if data and model folders exist
Getting Help
If you encounter problems, please check:

Whether Python version is compatible
Whether all dependency packages are correctly installed
Whether necessary files exist
Console error messages
Technical Architecture
Frontend: Streamlit
Backend: Python + scikit-learn
Model: SVM classifier
Explainability: SHAP (SHapley Additive exPlanations)
Data Processing: pandas + numpy
Visualization: matplotlib + plotly
License
This project is for learning and research purposes only.
