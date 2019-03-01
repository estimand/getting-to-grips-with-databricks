# Databricks notebook source
# MAGIC %md
# MAGIC # Modelling
# MAGIC ## *Getting to grips with Databricks*
# MAGIC
# MAGIC By Gianluca Campanella (<g.campanella@estimand.com>)
# MAGIC
# MAGIC [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

# COMMAND ----------

import numpy as np

from datetime import datetime
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import functions as F, types as T
from pyspark.storagelevel import StorageLevel

# COMMAND ----------

movies = spark.table('movies')
ratings = spark.table('ratings')

# COMMAND ----------

# MAGIC %md
# MAGIC Split data temporally into training and test sets.

# COMMAND ----------

testSetStart = F.lit(datetime(2018, 1, 1))

ratings = ratings.withColumn('isTestSet', F.col('ts') >= testSetStart)

# COMMAND ----------

# MAGIC %md
# MAGIC Extract training set and join with `movies`.

# COMMAND ----------

trainingSet = (ratings
               .filter(~F.col('isTestSet'))
               .drop('isTestSet')
               .join(movies.hint('broadcast'), on='movieId', how='inner'))

# COMMAND ----------

display(trainingSet)

# COMMAND ----------

# MAGIC %md
# MAGIC Compute tf-idf representations of the `genres` of movies rated by each user (during the training period).

# COMMAND ----------

def computeGenreTFIDFs(df):
    genreTFs = (df
                .select(F.col('userId'),
                        F.explode(F.col('genres')).alias('genre'))
                .groupBy(F.col('userId'), F.col('genre'))
                .agg(F.count(F.col('*')).alias('tf')))
    nUsers = genreTFs.count()
    genreIDFs = (genreTFs
                 .groupBy(F.col('genre'))
                 .agg(F.log(F.lit(nUsers) /
                            F.sum((F.col('tf') > 0).cast(T.IntegerType())))
                      .alias('idf')))
    genreTFIDFs = (genreTFs
                   .join(genreIDFs, on='genre', how='inner')
                   .withColumn('tfidf', F.col('tf') * F.col('idf'))
                   .drop('tf', 'idf')
                   .groupBy(F.col('userId'))
                   .agg(F.collect_list(F.struct(F.col('genre'), F.col('tfidf')))
                        .alias('genreTFIDFs'))
                   .withColumn('genreTFIDFs',
                               F.map_from_entries(F.col('genreTFIDFs'))))
    return genreTFIDFs

# COMMAND ----------

trainingGenreTFIDFs = computeGenreTFIDFs(trainingSet)

# COMMAND ----------

trainingSet = (trainingSet
               .join(trainingGenreTFIDFs, on='userId', how='inner'))

# COMMAND ----------

display(trainingSet)

# COMMAND ----------

# MAGIC %md
# MAGIC Extract test set and join with `movies` and `trainingGenreTFIDFs`.

# COMMAND ----------

testSet = (ratings
           .filter(F.col('isTestSet'))
           .drop('isTestSet')
           .join(movies.hint('broadcast'), on='movieId', how='inner')
           .join(trainingGenreTFIDFs, on='userId', how='inner'))

# COMMAND ----------

display(testSet)

# COMMAND ----------

# MAGIC %md
# MAGIC Load custom transformers and featurisation pipeline.

# COMMAND ----------

# MAGIC %run ./featurizers

# COMMAND ----------

# MAGIC %md
# MAGIC Featurise training and test sets.

# COMMAND ----------

fittedPipeline = featurizationPipeline.fit(trainingSet)

featurizedTrainingSet = (fittedPipeline
                         .transform(trainingSet)
                         .select(F.col('userId'), F.col('movieId'),
                                 F.col('rating'), F.col('features'))
                         .persist(StorageLevel.MEMORY_AND_DISK_SER))

featurizedTestSet = (fittedPipeline
                     .transform(testSet)
                     .select(F.col('userId'), F.col('movieId'),
                             F.col('rating'), F.col('features'))
                     .persist(StorageLevel.MEMORY_AND_DISK_SER))

# COMMAND ----------

display(featurizedTrainingSet)

# COMMAND ----------

display(featurizedTestSet)

# COMMAND ----------

# MAGIC %md
# MAGIC Initialise `LinearRegression` and define hyperparameter grid to search over.

# COMMAND ----------

lr = LinearRegression(labelCol='rating')

paramMaps = (ParamGridBuilder()
             .addGrid(lr.elasticNetParam, [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.])
             .addGrid(lr.regParam, [10.**x for x in range(-4, 5)])
             .build())

# COMMAND ----------

# MAGIC %md
# MAGIC Perform grid search using RMSE as metric to minimise.

# COMMAND ----------

gridSearchResults = [None] * len(paramMaps)

evaluator = RegressionEvaluator(metricName='rmse',
                                labelCol='rating',
                                predictionCol='prediction')

for i, model in lr.fitMultiple(featurizedTrainingSet, paramMaps):
    predictions = model.transform(featurizedTestSet, paramMaps[i])
    gridSearchResults[i] = evaluator.evaluate(predictions)
    print('α = {elasticNetParam:.2f}, λ = {regParam:.0e}, '
          'RMSE = {rmse:.3f}'
          .format(**{k.name: v for k, v in paramMaps[i].items()},
                  rmse=gridSearchResults[i]))

bestModelIndex = np.argmin(gridSearchResults)
bestParams = {k.name: v for k, v in paramMaps[bestModelIndex].items()}

# COMMAND ----------

print('Best hyperparameters found:')
print('α = {elasticNetParam:.2f}, λ = {regParam:.0e}, '
      'RMSE = {rmse:.3f}'
      .format(**bestParams,
              rmse=gridSearchResults[bestModelIndex]))

# COMMAND ----------

featurizedTrainingSet.unpersist()
featurizedTestSet.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC Re-fit model on entire dataset.

# COMMAND ----------

all_ = (ratings
        .drop('isTestSet')
        .join(movies.hint('broadcast'), on='movieId', how='inner'))

genreTFIDFs = computeGenreTFIDFs(all_)

all_ = (all_
        .join(genreTFIDFs, on='userId', how='inner'))

featurizedAll = (featurizationPipeline.fit(all_)
                 .transform(all_)
                 .select(F.col('userId'), F.col('movieId'),
                         F.col('rating'), F.col('features'))
                 .persist(StorageLevel.MEMORY_AND_DISK_SER))

# COMMAND ----------

lr = LinearRegression(**bestParams)

model = lr.fit(featurizedAll)

# COMMAND ----------

# MAGIC %md
# MAGIC Persist model to DBFS.

# COMMAND ----------

model.write().overwrite().save('fitted-model')
