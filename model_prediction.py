import numpy as np

def prepare_input(input_string, num_features):
    """
    Prepare the input string for prediction.
    """
    try:
        # Split the input string and convert to float
        values = [float(x.strip()) for x in input_string.split(',')]
        
        # Ensure the correct number of features
        if len(values) != num_features:
            raise ValueError(f"Expected {num_features} features, but got {len(values)}")
        
        # Reshape for model input
        return np.array(values).reshape(1, -1)
    except ValueError as e:
        raise ValueError(f"Invalid input: {str(e)}")

def make_prediction(model, input_data):
    """
    Make a prediction using the trained model.
    """
    try:
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get the predicted class (0 or 1)
        predicted_class = (prediction > 0.5).astype(int)
        
        # Get the probability
        probability = prediction[0][0]
        
        return predicted_class[0][0], probability
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")