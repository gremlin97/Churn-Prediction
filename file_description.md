# Churn Prediction Model

This repository contains the code and resources for building a customer churn prediction model using PySpark and MLflow. The model predicts whether a customer will churn based on various features such as credit score, age, balance, and more.

## Data Description

The dataset used for training the model has the following columns:

- `CustomerID`: A unique identifier for each customer
- `Surname`: The customer's surname or last name
- `CreditScore`: A numerical value representing the customer's credit score
- `Geography`: The country where the customer resides
- `Gender`: The customer's gender
- `Age`: The customer's age
- `Tenure`: The number of years the customer has been with the bank
- `Balance`: The customer's account balance
- `NumOfProducts`: The number of bank products the customer uses (e.g., savings account, credit card)
- `HasCrCard`: Whether the customer has a credit card
- `IsActiveMember`: Whether the customer is an active member
- `EstimatedSalary`: The estimated salary of the customer
- `Exited`: Whether the customer has churned (Target Variable)

## Files and Directories

- `preprocess.py`: Contains functions for reading and preprocessing the data.
- `main.py`: Contains the main workflow for training, testing, and logging the model.
- `bentoml_service.py`: Contains the BentoML service for deploying the churn prediction model.
- `data/`: Directory containing the dataset files.
- `pipeline/`: Directory containing the saved pipeline model for inference.
- `models/`: Directory containing the saved GBT model.

## Dependencies

- PySpark
- MLflow
- BentoML
- Pandas
- NumPy

## Setup

1. Install the required dependencies:
    ```sh
    pip install pyspark mlflow bentoml pandas numpy
    ```

2. Start the MLflow tracking server:
    ```sh
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
    ```

## Usage

### Training the Model

1. Preprocess the data and train the model by running `main.py`:
    ```sh
    python main.py
    ```

2. The script performs the following steps:
    - Reads the training data from `data/train.csv`.
    - Preprocesses the data.
    - Splits the data into training and testing sets.
    - Trains a Gradient Boosted Tree (GBT) classifier.
    - Logs the model and metrics to MLflow.

### Model Inference

1. Deploy the model as a BentoML service by running `bentoml_service.py`:
    ```sh
    bentoml serve bentoml_service:Churner
    ```

2. Make a prediction using the deployed BentoML service:
    ```sh
    import bentoml

    data_dict = {
        "CreditScore": [668, 664],
        "Geography": ["France", "France"],
        "Gender": ["Male", "Male"],
        "Age": [33.0, 34.0],
        "Tenure": [3, 4],
        "Balance": [0.0, 1.0],
        "NumOfProducts": [2, 3],
        "HasCrCard": [1.0, 3],
        "IsActiveMember": [0.0, 0.0],
        "EstimatedSalary": [181449.97, 32242.46],
    }

    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        result = client.churn(data=data_dict)

    print(result)
    ```

### Hyperparameter Tuning

1. Perform hyperparameter tuning using grid search:
    ```sh
    python grid_search.py
    ```

2. The script performs the following steps:
    - Reads the test data from `./data/test`.
    - Defines the GBT classifier and the parameter grid.
    - Uses cross-validation to find the best hyperparameters.
    - Logs the best model and metrics to MLflow.

## Logging and Tracking

- All training runs, models, and metrics are logged to MLflow.
- Use the MLflow UI to track experiments and compare model performance.

## Feature Importance

To retrieve and display feature importance, use the `feature_imp` function in `main.py`.

## Notes

- Ensure the data file paths and model save paths are correctly set according to your directory structure.
- Customize the hyperparameters, evaluation metrics, and other configurations as needed.
