import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List
import nest_asyncio


# Applying nest_asyncio
nest_asyncio.apply()


# Loading my model and scalers
with open('RandomForest_model.sav', 'rb') as f:
    random_forest_model = pickle.load(f)

# Initialize label encoders (should be fitted with training data)
label_encoders = {
    'gender': LabelEncoder(),
    'job': LabelEncoder(),
    'location': LabelEncoder(),
    'marital_status': LabelEncoder()
}

# Fitting the encoders with actual data (placeholder)
def fit_label_encoders():
    for col in label_encoders.keys():
        label_encoders[col].fit([
            'male', 'female', 'other',
            'Teacher', 'Nurse', 'Doctor', 'Data Analyst', 'Software Developer', 'Accountant', 'Lawyer', 'Engineer', 'Data Scientist', 
            'Beitbridge', 'Harare', 'Gweru', 'Rusape', 'Chipinge', 'Chimanimani', 'Marondera', 'Kadoma', 'Mutare', 'Masvingo', 'Bulawayo', 
            'Kariba', 'Plumtree', 'Chiredzi', 'Shurugwi', 'Chivhu', 'Zvishavane', 'Nyanga', 'Karoi', 'Redcliff', 'Kwekwe', 'Gokwe', 
            'Victoria Falls', 'Hwange',
            'married','single', 'divorced'
        ])

fit_label_encoders()

# Initializing StandardScaler
scaler = StandardScaler()

app = FastAPI()

# Requesting model for training
class TrainRandomForestRequest(BaseModel):
    gender: str
    is_employed: bool
    job: str
    location: str
    loan_amount: float
    number_of_defaults: int
    outstanding_balance: float
    interest_rate: float
    age: int
    remaining_term: int
    salary: float
    marital_status: str
    loan_status: int  # Target variable

@app.post('/train_random_forest')
def train_random_forest(request: List[TrainRandomForestRequest]):
    data = [
        [
            label_encoders['gender'].transform([item.gender])[0],
            item.is_employed,
            label_encoders['job'].transform([item.job])[0],
            label_encoders['location'].transform([item.location])[0],
            item.loan_amount,
            item.number_of_defaults,
            item.outstanding_balance,
            item.interest_rate,
            item.age,
            item.remaining_term,
            item.salary,
            label_encoders['marital_status'].transform([item.marital_status])[0]
        ]
        for item in request
    ]
    
    X_train = np.array(data)
    y_train = np.array([item.loan_status for item in request])
    
    # Standardize numerical features
    X_train[:, 4:] = scaler.fit_transform(X_train[:, 4:])  # Fit and transform the numerical columns
    
    random_forest_model.fit(X_train, y_train)
    return {'message': 'Model trained successfully'}

# Request model for inference
class InferenceRandomForestRequest(BaseModel):
    gender: str
    is_employed: bool
    job: str
    location: str
    loan_amount: float
    number_of_defaults: int
    outstanding_balance: float
    interest_rate: float
    age: int
    remaining_term: int
    salary: float
    marital_status: str

@app.post('/predict_random_forest_proba')
def predict_random_forest_proba(request: InferenceRandomForestRequest):
    X_test = np.array([[
        label_encoders['gender'].transform([request.gender])[0],
        request.is_employed,
        label_encoders['job'].transform([request.job])[0],
        label_encoders['location'].transform([request.location])[0],
        request.loan_amount,
        request.number_of_defaults,
        request.outstanding_balance,
        request.interest_rate,
        request.age,
        request.remaining_term,
        request.salary,
        label_encoders['marital_status'].transform([request.marital_status])[0]
    ]])
    
    # Standardize numerical features
    X_test[:, 4:] = scaler.transform(X_test[:, 4:])  # Use transform to standardize

  

    y_proba = random_forest_model.predict_proba(X_test)
    
    if y_proba.shape[1] == 1:
        # Handling single-class prediction (probability of the single class)
        positive_class_probability = y_proba[:, 0]
    else:
        # Handling multi-class prediction
        positive_class_probability = y_proba[:, 1]
    
    return {'probability of default': positive_class_probability.tolist()}

# Running the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


