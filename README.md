# Amazon Returns Predictor: Deep Learning for E-commerce Optimization

This project aims to predict product returns for an e-commerce platform using deep learning techniques. It demonstrates the application of various neural network architectures and regularization techniques to improve model performance.

## Project Structure

- `data/`: Contains the dataset
- `notebooks/`: Jupyter notebooks for analysis
- `src/`: Source code for data preprocessing, model creation, training, and evaluation
- `models/`: Saved model weights
- `requirements.txt`: List of required packages
- `concepts_explained.md`: Detailed explanation of concepts used in the project

## Setup

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook in the `notebooks/` directory

## Usage

The main analysis is conducted in the `amazon_returns_analysis.ipynb` notebook. This notebook loads the data, preprocesses it, creates and trains various models, and evaluates their performance.

## Models

The project implements and compares four different neural network models:
1. Baseline Model
2. L2 Regularized Model
3. Dropout Model
4. Batch Normalization Model

## Results

The notebook provides visualizations of model performance, including accuracy plots, confusion matrices, and ROC curves for each model.

## Contributing

Feel free to fork this project and submit pull requests with improvements or additional features.

## License

This project is open source and available under the [MIT License](LICENSE).