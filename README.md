## README

* ## Project Summary: Customer Churn Prediction

### Objective

The primary goal of this project is to build a machine learning model that can predict whether a bank customer will churn (i.e., leave the bank) based on various features such as credit score, age, account balance, and other relevant factors. By identifying potential churners, the bank can take proactive measures to retain these customers.

### Approach

1. **Data Preprocessing**:
    - **Reading Data**: The dataset containing customer information is read from a CSV file.
    - **Feature Engineering**: Irrelevant features (e.g., `CustomerID`, `Surname`) are dropped. Categorical variables (`Geography` and `Gender`) are encoded using One-Hot Encoding.
    - **Vector Assembling**: All features are assembled into a single feature vector.

2. **Model Training**:
    - **Train-Test Split**: The dataset is split into training and testing sets.
    - **Model Selection**: A Gradient Boosted Tree (GBT) classifier is chosen for its ability to handle various data types and its effectiveness in classification tasks.
    - **Training**: The GBT model is trained on the training set with specified hyperparameters (`maxIter`, `maxDepth`, `stepSize`, and `seed`).

3. **Model Evaluation**:
    - **Prediction**: The trained model makes predictions on the test set.
    - **Evaluation**: The model's performance is evaluated using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

4. **Hyperparameter Tuning**:
    - **Grid Search**: To find the optimal hyperparameters, grid search with cross-validation is performed.
    - **Model Selection**: The best model is selected based on the highest AUC-ROC score.

5. **Model Logging and Tracking**:
    - **MLflow Integration**: All experiments, models, and metrics are logged using MLflow for easy tracking and comparison.
    - **Model Registry**: The best-performing model is registered in the MLflow model registry for future use.

6. **Deployment**:
    - **BentoML Service**: The trained model is deployed as a BentoML service, enabling predictions to be made via API requests.

7. **Inference**:
    - **Real-Time Predictions**: The deployed model can be used to make real-time predictions on new customer data.

### Data Description

* **Customer ID**: A unique identifier for each customer
* **Surname**: The customer's surname or last name
* **Credit Score**: A numerical value representing the customer's credit score
* **Geography**: The country where the customer resides
* **Gender**: The customer's gender
* **Age**: The customer's age.
* **Tenure**: The number of years the customer has been with the bank
* **Balance**: The customer's account balance
* **NumOfProducts**: The number of bank products the customer uses (e.g., savings account, credit card)
* **HasCrCard**: Whether the customer has a credit card
* **IsActiveMember**: Whether the customer is an active member
* **EstimatedSalary**: The estimated salary of the customer
* **Exited**: Whether the customer has churned (Target Variable)

### Tools and Libraries

- **PySpark**: Used for data processing and model training.
- **MLflow**: Used for experiment tracking, model logging, and deployment.
- **BentoML**: Used for deploying the model as a service.
- **Pandas**: Used for data manipulation.
- **NumPy**: Used for numerical operations.

### Summary of Steps

1. **Data Preparation**:
    - Read and preprocess the data.
    - Encode categorical variables and assemble features into a single vector.

2. **Model Training and Evaluation**:
    - Split the data into training and testing sets.
    - Train a GBT classifier on the training set.
    - Evaluate the model on the test set using AUC-ROC.

3. **Hyperparameter Tuning**:
    - Perform grid search with cross-validation to find the best hyperparameters.
    - Select the best model based on evaluation metrics.

4. **Logging and Tracking**:
    - Log experiments, models, and metrics using MLflow.

5. **Deployment and Inference**:
    - Deploy the model as a BentoML service.
    - Use the deployed service to make predictions on new data.

By following this approach, we aim to create a robust and reliable churn prediction model that can help the bank identify and retain customers at risk of churning.
