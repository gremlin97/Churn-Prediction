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
    result = client.churn(
        data = data_dict
    )

print(result)

'''
Json Request:

{
        "data": {
            "CreditScore": [668, 664],
            "Geography": ["France", "France"],
            "Gender": ["Male", "Male"],
            "Age": [33.0, 34.0],
            "Tenure": [3, 4],
            "Balance": [0.0, 1.0],
            "NumOfProducts": [2, 3],
            "HasCrCard": [1.0, 3],
            "IsActiveMember": [0.0, 0.0],
            "EstimatedSalary": [181449.97, 32242.46]
        }
}

'''