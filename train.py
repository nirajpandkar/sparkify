# import libraries
import re

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, LongType
from pyspark.sql.functions import col, sum, countDistinct, udf, max, min, isnull, datediff
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

def get_metrics(res):
    """
    Provides accuracy, precision, recall and F1 score given prediction results.
    
    Arguments:
        res: Test prediction results
    Returns:
        None.
        Prints the accuracy, precision, recall and f1 score.
    """
    total = res.count()
    
    tp = res.where((res.label==1) & (res.prediction==1)).count()
    tn = res.where((res.label==0) & (res.prediction==0)).count()
    
    fp = res.where((res.label==0) & (res.prediction==1)).count()
    fn = res.where((res.label==1) & (res.prediction==0)).count()
        
    accuracy = (1.0*tp + tn) / total
    precision = 1.0*tp / (tp + fp)
    recall = 1.0*tp / (tp + fn)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    
    print('Accuracy: ', round(accuracy, 2))
    print('Precision: ', round(precision, 2))
    print('Recall: ', round(recall, 2))
    print('F1-Score: ', round(f1, 2))

def extract_features(df):
    """
    Create a vector Assembler of the features.
    
    Arguments:
        df: Dataframe consisting the relevant data columns.
    Returns:
        Dataframe with extracted features in the column "features".
    """
    feature_df = df.select("userId").distinct()
    col_names = []
    
    ts_dt_udf = udf(lambda x: x//1000, LongType())
    df = df.withColumn("registration_dt", ts_dt_udf(df.registration).cast("timestamp"))
    df = df.withColumn("timestamp_dt", ts_dt_udf(df.ts).cast("timestamp"))
    
    # Session Counts
    session_counts = df.groupby('userId').agg(countDistinct('sessionId').alias('session_count'))
    
    feature_df = feature_df.join(session_counts, on="userId")
    col_names.append("session_count")

    
    # Page Counts
    pages = df.select('page').distinct().sort('page')
    pages_list = [r.page for r in pages.collect()]
    page_counts = df.groupby('userId').pivot('page', pages_list).count()
    
    # Drop the "Cancel" page column
    # Fill NaNs with 0 - This will inherently transform "Cancellation Confirmation" column into "label" 
    # with 1 as churned and 0 as non churned
    page_counts = page_counts.drop("Cancel")
    page_counts = page_counts.fillna(value=0)
    page_counts = page_counts.withColumnRenamed("Cancellation Confirmation", "label")
    
    # Join these feature columns to our feature dataframe
    feature_df = feature_df.join(page_counts, on="userId")
    
    # Normalize by session counts
    cut_columns = {'userId', 'session_count', 'label'}
    remaining_cols = sorted(list(set(feature_df.columns) - cut_columns))
    for column in remaining_cols:
        feature_df = feature_df.withColumn(column, col(column) / feature_df.session_count)
    col_names.extend(remaining_cols)
    
    # Time since registration
    user_ages = df.select(["userId", datediff("timestamp_dt", "registration_dt")]).groupBy("userId").max().select("userId", col("max(datediff(timestamp_dt, registration_dt))").alias("age"))    
    feature_df = feature_df.join(user_ages, on="userId")
    col_names.append("age")
    
    # Total number of events 
    user_number_events = df.groupBy("userId").count().select("userId", col("count").alias("num_events"))
    feature_df = feature_df.join(user_number_events, on="userId")
    col_names.append("num_events")
    
    # Include device categorical variable
    device_udf = udf(lambda x: str(re.findall(r'\((.*?)\)', x)[0].split(";")[0].split()[0]) if x is not None else None, StringType())
    df = df.withColumn("device", device_udf(df.userAgent))

    df_device = df.select(["userId", "device"]).distinct()
    df_device = StringIndexer(inputCol = "device", outputCol="device_index").fit(df_device).transform(df_device)
    df_device = OneHotEncoderEstimator(inputCols=["device_index"], outputCols=["device_classVec"]).fit(df_device).transform(df_device)
    feature_df = feature_df.join(df_device.select("userId", "device_classVec"), on="userId")
    col_names.append("device_classVec")

    print(col_names)
    # Assemble the vector
    assembler = VectorAssembler(inputCols=col_names, outputCol='features')
    
    return assembler.transform(feature_df)

def preprocess(df):
    """
    Cleans the dataset of unwanted rows.
    """
    df = df.filter(df.userId != "")
    return df

def split_train_test(df_features):
    """
    Splits the dataset into training and testing subsets.
    """
    train, test = df_features.randomSplit([0.8, 0.2], 42)
    return train, test

def trainLR(train):
    """
    Trains the model given train dataset
    """
    lr =  LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.0, elasticNetParam=0)
    model = lr.fit(train)
    return model

def crossValidationLR(train):
    """
    Performs grid search using cross validation and returns a CV model.
    """
    paramGrid = ParamGridBuilder()\
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.fitIntercept, [False, True])\
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
        .build()

    crossval = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=3)
    model = crossval.fit(train)
    return model

def evaluate(model, test):
    """
    Evaluates the trained model on the test set and prints out evaluation metrics.
    """
    predictions = model.transform(test)
    get_metrics(predictions)
    
if __name__ == "__main__":
    # create a Spark session
    spark = SparkSession.builder.appName("sparkify").getOrCreate()

    path = "medium-sparkify-event-data.json"
    df = spark.read.json(path)
    
    df = preprocess(df)
    df_features = extract_features(df)
    
    train, test = split_train_test(df_features)
    model = trainLR(train)
    evaluate(model, test)