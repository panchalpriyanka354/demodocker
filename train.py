import boto3
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# S3 bucket and data location
BUCKET_NAME = 'my-churn-model-bucket'
DATA_KEY = 'data/churn.csv'
MODEL_KEY = 'model/churn-model.pkl'

# Initialize S3 client
s3 = boto3.client('s3')

def download_data(bucket, key):
    """Download data from S3."""
    s3.download_file(bucket, key, 'churn.csv')
    return pd.read_csv('churn.csv')

def preprocess_data(df):
    """Preprocess the data."""
    # Assuming 'Churn' is the target column and others are features
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler, bucket, key):
    """Save the trained model and scaler to S3."""
    with open('churn-model.pkl', 'wb') as f:
        joblib.dump({'model': model, 'scaler': scaler}, f)
    s3.upload_file('churn-model.pkl', bucket, key)

def main():
    # Download the data
    df = download_data(BUCKET_NAME, DATA_KEY)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    
    # Save the model to S3
    save_model(model, scaler, BUCKET_NAME, MODEL_KEY)

if __name__ == '__main__':
    main()
