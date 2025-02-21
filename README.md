# python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template

# Sample Dataset (Replace this with a real dataset)
data = {
    'annual_income': [50000, 120000, 30000, 80000, 45000, 70000],
    'credit_score': [700, 750, 650, 800, 600, 720],
    'loan_amount': [20000, 50000, 10000, 30000, 15000, 25000],
    'years_in_business': [5, 10, 2, 7, 3, 8],
    'employees': [3, 10, 2, 5, 4, 6],
    'approved': [1, 1, 0, 1, 0, 1]  # 1 = Approved, 0 = Rejected
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Splitting Data
X = df.drop(columns=['approved'])
y = df['approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating Model Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train Model
pipeline.fit(X_train, y_train)

# Save Model
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Flask App
app = Flask(_name_)

@app.route('/')
def home():
    return "Business Loan Eligibility Predictor API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['annual_income'], data['credit_score'], data['loan_amount'],
                          data['years_in_business'], data['employees']]])
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(features)
    return {'loan_approval': int(prediction[0])}

if _name_ == '_main_':
    app.run(debug=True)
