# Databricks notebook source
# MAGIC %md
# MAGIC # ETL
# MAGIC ## *Getting to grips with Databricks*
# MAGIC
# MAGIC By Gianluca Campanella (<g.campanella@estimand.com>)
# MAGIC
# MAGIC [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

# COMMAND ----------

from pyspark.sql import functions as F, types as T

# COMMAND ----------

movielensLocation = 'dbfs:/movielens/'

# COMMAND ----------

# MAGIC %md
# MAGIC Print the contents of `README.txt`.

# COMMAND ----------

print(dbutils.fs.head('dbfs:/movielens/README.txt'))

# COMMAND ----------

# MAGIC %md
# MAGIC Load movies from `movies.csv`.

# COMMAND ----------

MovieType = T.StructType([
    T.StructField('movieId', T.IntegerType()),
    T.StructField('title', T.StringType()),
    T.StructField('genres', T.StringType()),
])

movies = (spark.read
          .option('header', True)
          .csv(movielensLocation + 'movies.csv', schema=MovieType))

# COMMAND ----------

display(movies)

# COMMAND ----------

# MAGIC %md
# MAGIC Set `genres` to missing when equal to '(no genres listed)', otherwise split into a string array.

# COMMAND ----------

movies = (movies
          .withColumn('genres',
                      F.when(F.col('genres') == '(no genres listed)', None)
                       .otherwise(F.split(F.col('genres'), r'\|'))))

# COMMAND ----------

display(movies)

# COMMAND ----------

# MAGIC %md
# MAGIC Extract the release year into `releaseYear`.

# COMMAND ----------

movies = (movies
          .withColumn('releaseYear',
                      F.regexp_extract(F.col('title'), r'\((\d{4})\)', 1)
                       .cast(T.IntegerType())))

# COMMAND ----------

display(movies)

# COMMAND ----------

# MAGIC %md
# MAGIC Save `movies` as a managed table in Hive.

# COMMAND ----------

(movies
 .coalesce(1)
 .write
 .mode('overwrite')
 .saveAsTable('movies'))

# COMMAND ----------

# MAGIC %md
# MAGIC Load ratings from `ratings.csv`.

# COMMAND ----------

RatingType = T.StructType([
    T.StructField('userId', T.IntegerType()),
    T.StructField('movieId', T.IntegerType()),
    T.StructField('rating', T.FloatType()),
    T.StructField('timestamp', T.LongType()),
])

ratings = (spark.read
           .option('header', True)
           .csv(movielensLocation + 'ratings.csv', schema=RatingType)
           .withColumn('ts', F.col('timestamp').cast('timestamp'))
           .drop('timestamp'))

# COMMAND ----------

display(ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC Save `ratings` as a managed table in Hive.

# COMMAND ----------

(ratings
 .repartition(100, 'movieId', 'userId')
 .sortWithinPartitions('movieId', 'userId')
 .write
 .bucketBy(100, 'movieId', 'userId')
 .mode('overwrite')
 .saveAsTable('ratings'))
