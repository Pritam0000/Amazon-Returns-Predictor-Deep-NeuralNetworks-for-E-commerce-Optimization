import streamlit as st
from data_preprocessing import load_and_preprocess_data
from model_creation import create_baseline_model
from model_training import train_model
from model_evaluation import evaluate_model
from model_prediction import prepare_input, make_prediction
import numpy as np
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(
    page_title="Neural Network for customer classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌀 Neural Network for customer classification")
st.write("A neural network trained for customer classification .")

# Initialize session state for storing data and model
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False  # To track preprocessing
    st.session_state.X_train, st.session_state.X_val, st.session_state.X_test = None, None, None
    st.session_state.y_train, st.session_state.y_val, st.session_state.y_test = None, None, None

if 'model' not in st.session_state:
    st.session_state.model = None  # Placeholder for the model

# Step 1: Preprocess Data
st.header("1. Data Preprocessing")
if st.button("Preprocess Data"):
    (st.session_state.X_train, st.session_state.X_val, 
     st.session_state.X_test, st.session_state.y_train, 
     st.session_state.y_val, st.session_state.y_test) = load_and_preprocess_data()

    st.session_state.data_preprocessed = True  # Mark as preprocessed
    st.success("✅ Data preprocessed successfully!")
    st.write(f"📊 **Training data shape:** {st.session_state.X_train.shape}")
    st.write(f"📊 **Test data shape:** {st.session_state.X_test.shape}")

# Step 2: Create Model
st.header("2. Model Creation")
if st.button("Create Model"):
    if not st.session_state.data_preprocessed:
        st.error("⚠️ Please preprocess the data first!")
    else:
        input_shape = (st.session_state.X_train.shape[1],)  # Match input shape with data
        st.session_state.model = create_baseline_model(input_shape)
        st.success("✅ Model created successfully!")
        st.write("### Model Summary:")
        model_summary = []
        st.session_state.model.summary(print_fn=lambda x: model_summary.append(x))
        st.text("\n".join(model_summary))  # Display model summary

# Step 3: Train Model
st.header("3. Train the Model")
epochs = st.number_input("Enter the number of epochs:", min_value=1, value=10, step=1)
if st.button("Train Model"):
    if st.session_state.model is None or not st.session_state.data_preprocessed:
        st.error("⚠️ Please create the model and preprocess data first!")
    else:
        history, model = train_model(
            st.session_state.model,
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.X_val,
            st.session_state.y_val,
            epochs
        )
        st.session_state.model = model  # Update the model in session state
        st.success("Model trained successfully!")
        st.write("Training history:")
        st.line_chart(history.history['accuracy'])

import streamlit as st
from data_preprocessing import load_and_preprocess_data
from model_creation import create_baseline_model
from model_training import train_model
from model_evaluation import evaluate_model
from model_prediction import prepare_input, make_prediction
import numpy as np
import seaborn as sns


# Step 4: Evaluate the Model
st.header("4. Evaluate the Model")
if st.button("Evaluate Model"):
    if st.session_state.model is None or not st.session_state.data_preprocessed:
        st.error("⚠️ Please create the model and preprocess data first!")
    else:
        try:
            # Print data shapes before evaluation
            st.write("Data shapes:")
            st.write(f"X_test shape: {st.session_state.X_test.shape if hasattr(st.session_state.X_test, 'shape') else 'Unknown'}")
            st.write(f"y_test shape: {st.session_state.y_test.shape if hasattr(st.session_state.y_test, 'shape') else 'Unknown'}")
            
            # Print model input shape
            st.write(f"Model input shape: {st.session_state.model.input_shape}")
            st.write(f"Model output shape: {st.session_state.model.output_shape}")
            
            accuracy, loss = evaluate_model(
                st.session_state.model, 
                st.session_state.X_test, 
                st.session_state.y_test
            )
            if accuracy is not None and loss is not None:
                st.write(f"🎯 **Model Accuracy:** {accuracy * 100:.2f}%")
                st.write(f"📉 **Model Loss:** {loss:.4f}")
            else:
                st.error("⚠️ Model evaluation failed. Please check the console for error messages.")
        except Exception as e:
            st.error(f"⚠️ An error occurred during model evaluation: {str(e)}")
            st.error("Please check the console for more detailed error messages.")
            import traceback
            st.text(traceback.format_exc())

