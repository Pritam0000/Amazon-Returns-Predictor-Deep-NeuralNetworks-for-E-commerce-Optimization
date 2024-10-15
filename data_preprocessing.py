import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder


def load_and_preprocess_data(filepath='Amazon_Returns_Dataset.csv'):
    # Load data
    df = pd.read_csv(filepath)
    
    # Split features and target
    X = df.drop(columns=['ID', 'Returned'])
    y = df['Returned']
    
    # Train-test-validation split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Target encoding
    cat_features = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
    encoder = TargetEncoder(cols=cat_features)
    X_train = encoder.fit_transform(X_train, y_train)
    X_val = encoder.transform(X_val)
    X_test = encoder.transform(X_test)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test