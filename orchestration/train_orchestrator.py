import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull, when, count, col
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow.spark
import pyspark

mlflow.autolog()

@task(retries=3, retry_delay_seconds=1)
def read_dataframe(f: str) -> pyspark.sql.dataframe.DataFrame:
    df = spark.read.csv(f, header=True, inferSchema=True)
    return df

@task
def preprocess_df(df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    # Lets rename exited, our target Exited to Churned
    df = df.withColumnRenamed('Exited','Churned')
    # Lets drop id, CustomerId and Surname as they are irrelevant features that dont add any value
    df = df.drop('id','CustomerId','Surname')
    df = encoder(df)
    # lets select our features and target to create our final df
    df_final = df.select('IndependentFeatures','Churned')   
    return df_final

def encoder(df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    # Lets use one-hot-encoding (Not ordinal as there is no ordered relation between the feature values)
    # However the columns are made up of strings hence we need to encode them with the string indexer then follow up with a ohe
    se = StringIndexer(inputCols = ['Geography','Gender'], outputCols = ['Geography_se', 'Gender_se'])
    ohe = OneHotEncoder(inputCols = ['Geography_se','Gender_se'], outputCols = ['Geography_ohe', 'Gender_ohe'])
    encode_pipe = Pipeline(stages = [se, ohe])
    df = encode_pipe.fit(df).transform(df)
    # We will not normalize the data as we will make use of a gradient boosted tree. For GBT branching is done on basis of actual values, hence we dont require
    # any data normalization
    # We need to use the vector assembler to create the feature set and traget variable
    feature_assembler = VectorAssembler(inputCols = ['CreditScore', 'Geography_ohe', 'Gender_ohe', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'], outputCol = 'IndependentFeatures')
    df = feature_assembler.transform(df)
    return df

@task
def train_test_split(df: pyspark.sql.dataframe.DataFrame) ->  tuple:
    train, test = df.randomSplit([0.9,0.1])
    return train, test

@task(log_prints=True)
def grid_search(df_path: str) -> pyspark.ml.classification.GBTClassificationModel:
    # Model instantiation
    gbt = GBTClassifier(featuresCol='IndependentFeatures', labelCol='Churned')
    # Param Grid
    grid = tune.ParamGridBuilder()
    # Add Params
    grid = grid.addGrid(gbt.maxIter, np.arange(1,10,1))
    grid = grid.addGrid(gbt.maxDepth, np.arange(2,5,1))
    grid = grid.addGrid(gbt.stepSize, np.arange(0.1,0.3,0.1))
    # Adding Hyperparams
    grid = grid.build()
    # Evaluator
    evaluator = BinaryClassificationEvaluator(labelCol = 'Churned', metricName = 'areaUnderROC')
    # Cross-Validation
    cv = tune.CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=evaluator)
    opt_model = cv.fit(test)
    return opt_model

@task(log_prints=True)
def retrieve_runs(top_n: int) -> str:
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(TRAIN_EXPERIMENT)
    best_run = client.search_runs(experiment_ids = experiment.experiment_id, run_view_type = ViewType.ACTIVE_ONLY, max_results = top_n, order_by=["metrics.test_roc ASC"])[0]
    run_id = best_run.info.run_id
    model_uri = f'runs:/{run_id}/models'
    return model_uri

@flow(name='Best Model', description='Get the best Model after grid search')
def register_best_model(test_df: pyspark.sql.dataframe.DataFrame) -> None:
    with mlflow.start_run():
        model = grid_search(test)
        model_uri = retrieve_runs(5)
        # Steps: Staging -> Production -> Archived
        mlflow.register_model(model_uri = model_uri, name=OPT_EXPERIMENT)

@flow(name='Main Flow', description='Main Pipeline Flow')
def main_flow(train_file: str) -> None:
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('Churn Prediction Orchestrator')

    spark = SparkSession.builder.appName('ChurnPipe').getOrCreate()
    df = read_dataframe('data/train.csv')
    df = preprocess_df(df)  
    save_pipe(df)  

    train, test = train_test_split(df)
    register_best_model(train)
    # To-Do: Add test validation


if __name__ == '__main__':
    main_flow()

