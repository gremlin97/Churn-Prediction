import bentoml
import mlflow
import pandas as pd
import sys
import pyspark
from pyspark.sql import SparkSession

sys.path.append('../')

from preprocess import preprocess_df, preprocess_df_inference

mlflow.set_tracking_uri('http://127.0.0.1:5000')
spark = SparkSession.builder.appName('Churn').getOrCreate()

@bentoml.service(
    resources = {'cpu':'2'},
    traffic = {'timeout':10},
)

class Churner:
    def __init__(self) -> None:
        self.logged_model = 'runs:/661036cb6f9648b3a1d75e0540948042/best_model'
        self.model =  mlflow.spark.load_model(self.logged_model)
        #self.infer_pipe = mlflow.spark.load_model('/workspaces/Churn-Prediction/pipeline/inference_pipe')

    @bentoml.api
    def churn(self, data) -> pd.DataFrame:
        print('T1',data)
        df = pd.DataFrame([data], index=[0])
        df = pd.DataFrame(data)
        print('T2',data)
        sdf = spark.createDataFrame(df)
        print('DataFrame after preprocessing:', sdf.show())
        sdf = preprocess_df_inference(sdf)
        print('DataFrame after preprocessing:', sdf.show())
        result = self.model.transform(sdf)
        # predicted_value = int(result.select("prediction").collect()[0][0])
        # print('Result is:',predicted_value, type(predicted_value))
        result = result.select("prediction")
        print('Result is:', result, type(result))
        result = result.toPandas()
        return result
        # if predicted_value == 0:
        #     return 'No'
        # else:
        #     return 'Yes'



# CreditScore|Geography|Gender| Age|Tenure|  Balance|NumOfProducts|HasCrCard|IsActiveMember|EstimatedSalary|Churned|Geography_se|Gender_se|
# +-----------+---------+------+----+------+---------+-------------+---------+--------------+---------------+-------+------------+---------+
# |        668|   France|  Male|33.0|     3|      0.0|            2|      1.0|           0.0|      181449.97|      0|         0.0|      0.0

# data_dict = {
#     "CreditScore": 668,
#     "Geography": "France",
#     "Gender": "Male",
#     "Age": 33.0,
#     "Tenure": 3,
#     "Balance": 0.0,
#     "NumOfProducts": 2,
#     "HasCrCard": 1.0,
#     "IsActiveMember": 0.0,
#     "EstimateSalary": 181449.97,
#     "Churned": 0,
#     "Geography_se": 0.0,
#     "Gender_se": 0.0
# }d