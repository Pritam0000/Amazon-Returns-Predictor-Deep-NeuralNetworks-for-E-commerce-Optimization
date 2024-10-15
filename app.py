import streamlit as st
from data_preprocessing import preprocess_data
from model_creation import create_model
from model_training import train_model
from model_evaluation import evaluate_model

st.title("Neural Network Spiral Classifier")
st.write("A neural network trained from scratch to classify spiral patterns.")

# Step 1: Preprocess Data
st.header("1. Data Preprocessing")
if st.button("Preprocess Data"):
    X_train, X_test, y_train, y_test = preprocess_data()
    st.success("Data preprocessed successfully!")
    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Test data shape: {X_test.shape}")

# Step 2: Create Model
st.header("2. Model Creation")
if st.button("Create Model"):
    model = create_model()
    st.success("Model created successfully!")
    st.write("Model summary:")
    st.text(model.summary())

# Step 3: Train Model
st.header("3. Train the Model")
epochs = st.number_input("Enter the number of epochs:", min_value=1, value=10)
if st.button("Train Model"):
    history = train_model(model, X_train, y_train, epochs)
    st.success("Model trained successfully!")
    st.write("Training history:")
    st.line_chart(history.history['accuracy'])

# Step 4: Evaluate Model
st.header("4. Evaluate the Model")
if st.button("Evaluate Model"):
    accuracy, loss = evaluate_model(model, X_test, y_test)
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")
    st.write(f"Model Loss: {loss:.4f}")

st.write("---")
st.caption("Project by Khush. Powered by Streamlit.")
