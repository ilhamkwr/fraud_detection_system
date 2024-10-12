from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

import os

# Load model from file if it exists
def load_saved_model():
    global model
    model_path = 'models/model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded from 'models/model.pkl'")
    else:
        print("No saved model found.")

# Initialize FastAPI
app = FastAPI()

load_saved_model()

# Global variables for storing datasets and models
X_train, X_test, y_train, y_test = None, None, None, None
model = None

# Model options
models_dict = {
    "random_forest": RandomForestClassifier(n_estimators=100),
    "logistic_regression": LogisticRegression(),
    "decision_tree": DecisionTreeClassifier()
}

@app.post("/preprocess/")
async def preprocess_data(file_path: str, target_column: str):
    global X_train, X_test, y_train, y_test
    
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop irrelevant columns
    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    
    # One-hot encode 'type' column
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # Split data into features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return {"message": "Data preprocessing complete.", "X_train_shape": X_train.shape, "X_test_shape": X_test.shape}

@app.post("/train_model/")
async def train_model(algorithm: str, retrain: Optional[bool] = False):
    global model

    # Check if model already exists and if retraining is not required
    if model is not None and not retrain:
        return {"message": "Model already exists. Use 'retrain=true' if you want to retrain the model."}

    # Check if the selected algorithm is valid
    if algorithm not in models_dict:
        return {"error": f"Algorithm '{algorithm}' is not available. Choose from {list(models_dict.keys())}"}
    
    # Train the selected model
    model = models_dict[algorithm]
    model.fit(X_train, y_train)

    # Save the model to a pickle file
    joblib.dump(model, 'models/model.pkl')

    return {"message": f"Model trained with {algorithm} and saved as model.pkl"}


@app.get("/test_accuracy/")
async def test_accuracy():
    global model

    if model is None:
        return {"error": "No model has been trained yet."}

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return {"accuracy": accuracy}

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int

@app.post("/predict_fraud/")
async def predict_fraud(transaction: Transaction):
    global model, X_train
    
    if model is None:
        return {"error": "No model has been trained or loaded yet."}

    # Convert the input transaction to a DataFrame
    transaction_df = pd.DataFrame([transaction.dict()])

    # One-hot encode the 'type' column
    if 'type' in transaction_df.columns:
        transaction_df = pd.get_dummies(transaction_df, columns=['type'], drop_first=True)
    else:
        return {"error": "Transaction type is missing in the input"}

    # Align with the training data (adding missing columns)
    transaction_df, _ = transaction_df.align(X_train, join='left', axis=1, fill_value=0)

    # Make sure all necessary columns are present
    if set(transaction_df.columns) != set(X_train.columns):
        return {"error": "Mismatch in input features and training features"}

    # Make a prediction
    try:
        prediction = model.predict(transaction_df)
        return {"is_fraud": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}


