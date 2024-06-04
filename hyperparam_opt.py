from preprocess import read_dataframe, preprocess_df, spark
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow.spark
import pyspark
import mlflow
import pyspark.ml.tuning as tune
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import numpy as np

TRAIN_EXPERIMENT = 'Churn Prediction'
OPT_EXPERIMENT = 'Hyperparam Optimization'
mlflow.autolog()
# Start server: #mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./mlruns
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment(TRAIN_EXPERIMENT)

def load_model(logged_model) -> pyspark.ml.classification.GBTClassificationModel:
    model = mlflow.spark.load_model(logged_model)
    return model

def grid_search(df_path: str) -> pyspark.ml.classification.GBTClassificationModel:
    # Model instantiation
    gbt = GBTClassifier()
    # Param Grid
    grid = tune.ParamGridBuilder(gbt.maxIter, np.arrange(1,10,1))
    grid = tune.ParamGridBuilder(gbt.maxDepth, np.arrange(2,6,1))
    grid = tune.ParamGridBuilder(gbt.stepSize, np.arrange(0.01,0.3,0.01))
    # Adding Hyperparams
    grid = grid.build()
    # Evaluator
    evaluator = BinaryClassificationEvaluator(labelCol = 'Churned', metricName = 'areaUnderROC')
    # Cross-Validation
    cv = tune.CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=evaluator)
    opt_model = cv.fit(test)
    return model

def retrieve_runs(top_n: int) -> str:
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(TRAIN_EXPERIMENT)
    best_run = client.search_runs(experiment_ids = experiment.experiment_id, run_view_type = ViewType.ACTIVE_ONLY, max_results = top_n, order_by=["metrics.test_roc ASC"])[0]
    run_id = best_run.info.run_id
    model_uri = f'runs:/{run_id}/models'
    return model_uri

if __name__ == '__main__':
    test = spark.read.parquet('./data/test')
    with mlflow.start_run():
        model = grid_search(test)
        model_uri = retrieve_runs(5)
        mlflow.register_model(model_uri = model_uri, name=OPT_EXPERIMENT)




