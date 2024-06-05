import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull, when, count, col
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName('Churn').getOrCreate()

def read_dataframe(f: str) -> pyspark.sql.dataframe.DataFrame:
    df = spark.read.csv(f, header=True, inferSchema=True)
    return df

def preprocess_df(df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    # Lets rename exited, our target Exited to Churned
    df = df.withColumnRenamed('Exited','Churned')
    # Lets drop id, CustomerId and Surname as they are irrelevant features that dont add any value
    df = df.drop('id','CustomerId','Surname')
    df = encoder(df)
    # lets select our features and target to create our final df
    df_final = df.select('IndependentFeatures','Churned')   
    return df_final

def preprocess_df_inference(df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
    # Lets rename exited, our target Exited to Churned
    # df = df.withColumnRenamed('Exited','Churned')
    # Lets drop id, CustomerId and Surname as they are irrelevant features that dont add any value
    df = df.drop('id','CustomerId','Surname', 'Exited')
    im = PipelineModel.load('/workspaces/Churn-Prediction/pipeline/inference_pipe')
    df = im.transform(df)
    # lets select our features and target to create our final df
    df_final = df.select('IndependentFeatures')   
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

def save_pipe(df: pyspark.sql.dataframe.DataFrame) -> None:
    se = StringIndexer(inputCols = ['Geography','Gender'], outputCols = ['Geography_se', 'Gender_se'])
    ohe = OneHotEncoder(inputCols = ['Geography_se','Gender_se'], outputCols = ['Geography_ohe', 'Gender_ohe'])    
    fa = VectorAssembler(inputCols = ['CreditScore', 'Geography_ohe', 'Gender_ohe', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'], outputCol = 'IndependentFeatures')
    inference_pipe = Pipeline(stages = [se, ohe, fa])
    # Fit the pipeline to some data before saving
    pipeline_model = inference_pipe.fit(df)
    pipeline_model.write().overwrite().save('/workspaces/Churn-Prediction/pipeline/inference_pipe')


# df = read_dataframe('data/train.csv')
# df = preprocess_df(df)
# df.show(5)
# ===== Save Pipeline ==== #
# save_pipe(df)
