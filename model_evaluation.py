import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import tensorflow as tf

def evaluate_model(model, X_test, y_test):
    try:
        # Print shapes and types for debugging
        print(f"X_test type: {type(X_test)}, shape: {X_test.shape if hasattr(X_test, 'shape') else 'Unknown'}")
        print(f"y_test type: {type(y_test)}, shape: {y_test.shape if hasattr(y_test, 'shape') else 'Unknown'}")
        
        # Print model input shape
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")

        # Ensure X_test is a numpy array or tensor
        if not isinstance(X_test, (np.ndarray, tf.Tensor)):
            X_test = np.array(X_test)
        
        # Ensure y_test is a 1D numpy array
        if not isinstance(y_test, (np.ndarray, tf.Tensor)):
            y_test = np.array(y_test)
        if y_test.ndim > 1:
            y_test = y_test.ravel()
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.close()

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print metrics
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Evaluate the model
        loss = model.evaluate(X_test, y_test, verbose=0)[0]
        
        return accuracy, loss
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None