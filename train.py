from preprocess import read_dataframe, preprocess_df
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow.spark
import pyspark

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Churn Prediction')

# Hyperparams
mxiter = 5
mxdepth = 3
stepsz = 0.1
seed = 0

def train_test_split(df: pyspark.sql.dataframe.DataFrame) ->  tuple:
    train, test = df.randomSplit([0.9,0.1])
    return train, test

def train_model(train: pyspark.sql.dataframe.DataFrame, params) -> pyspark.ml.classification.GBTClassificationModel:

    gbt = GBTClassifier(featuresCol='IndependentFeatures', labelCol='Churned', **params)
    gbt = gbt.fit(train)
    return gbt

def prediction(test: pyspark.sql.dataframe.DataFrame, model: pyspark.ml.classification.GBTClassificationModel) ->  pyspark.sql.dataframe.DataFrame:
    preds = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol = 'Churned', metricName = 'areaUnderROC')
    roc = evaluator.evaluate(preds)
    print(f'Area under the curve for the model for the current run is {roc}')
    return roc

def feature_imp(model: pyspark.ml.classification.GBTClassificationModel) -> None:
    # Lets get the feature importance for the model
    importance = model.featureImportances
    features = ['CreditScore', 'Geography_ohe', 'Gender_ohe', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    print("#========Feature Importance========#")
    for feature, importance in zip(features, importance):
        print(f"{feature}: {importance:.4f}")

if __name__ == '__main__':
    with mlflow.start_run():
        df = read_dataframe('data/train.csv')
        df = preprocess_df(df)
        train, test = train_test_split(df)
        params = {'maxIter' : mxiter, 'maxDepth' : mxdepth, 'stepSize' : stepsz, 'seed' : seed}
        mlflow.log_params(params)
        model = train_model(train, params)
        roc = prediction(test, model)
        mlflow.spark.log_model(model,'Churn Predictor')
        mlflow.log_metric('Area under the ROC curve', roc)