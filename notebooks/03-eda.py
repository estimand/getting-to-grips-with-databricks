# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory data analysis
# MAGIC ## *Getting to grips with Databricks*
# MAGIC
# MAGIC By Gianluca Campanella (<g.campanella@estimand.com>)
# MAGIC
# MAGIC [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

# COMMAND ----------

from pyspark.sql import functions as F, types as T

# COMMAND ----------

movies = spark.table('movies')
ratings = spark.table('ratings')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Movies

# COMMAND ----------

display(movies)

# COMMAND ----------

# MAGIC %md
# MAGIC Number of movies by release year.

# COMMAND ----------

display(
    movies
    .groupBy(F.col('releaseYear'))
    .agg(F.count(F.col('*')).alias('nMovies'))
    .orderBy(F.col('releaseYear'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Number of movies by genre.

# COMMAND ----------

display(
    movies
    .select(F.explode(F.col('genres')).alias('genre'))
    .groupBy(F.col('genre'))
    .agg(F.count(F.col('*')).alias('nMovies'))
    .orderBy(F.col('nMovies').desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC Number of movies by word in the title.

# COMMAND ----------

display(
    movies
    .withColumn('title',
                F.trim(F.regexp_extract(F.col('title'), r'(.+?)\(', 1)))
    .select(F.explode(F.split(F.lower(F.col('title')), r'\s+')).alias('word'))
    .groupBy(F.col('word'))
    .agg(F.count(F.col('*')).alias('nMovies'))
    .orderBy(F.col('nMovies').desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC Create an [init script](https://docs.azuredatabricks.net/user-guide/clusters/init-scripts.html) that installs [NLTK](https://www.nltk.org/) and its data.
# MAGIC **Note**: this script needs to be [added to the cluster configuration](https://docs.azuredatabricks.net/user-guide/clusters/init-scripts.html#configure-a-cluster-scoped-init-script).

# COMMAND ----------

dbutils.fs.mkdirs('dbfs:/databricks/init-scripts/')

dbutils.fs.put('dbfs:/databricks/init-scripts/install-nltk.sh',
'''#!/bin/bash

/databricks/python/bin/pip install nltk
/databricks/python/bin/python -m nltk.downloader all''')

# COMMAND ----------

# MAGIC %md
# MAGIC Define a UDF (user-defined function) that tokenises and extracts nouns from a given string.

# COMMAND ----------

import nltk

@F.udf(T.ArrayType(T.StringType()))
def extractNouns(x):
    return [word
            for word, tag in nltk.pos_tag(nltk.word_tokenize(x.lower()))
            if tag.startswith('NN')]

# COMMAND ----------

display(
    movies
    .withColumn('titleNouns', extractNouns(F.col('title')))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Number of movies by noun in the title.

# COMMAND ----------

display(
    movies
    .withColumn('titleNouns', extractNouns(F.col('title')))
    .select(F.explode(F.col('titleNouns')).alias('noun'))
    .groupBy(F.col('noun'))
    .agg(F.count(F.col('*')).alias('nMovies'))
    .orderBy(F.col('nMovies').desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ratings

# COMMAND ----------

display(ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC Time range of ratings.

# COMMAND ----------

display(
    ratings
    .agg(F.min(F.col('ts')), F.max(F.col('ts')))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Maximum number of ratings per movie and user.

# COMMAND ----------

display(
    ratings
    .groupBy(F.col('movieId'), F.col('userId'))
    .agg(F.count(F.col('*')).alias('nRatings'))
    .agg(F.max(F.col('nRatings')))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Inner join with `movies`.

# COMMAND ----------

display(
    ratings
    .join(movies.hint('broadcast'), on='movieId', how='inner')
)

# COMMAND ----------

# MAGIC %md
# MAGIC Movies with 100 or more ratings sorted by average rating.

# COMMAND ----------

display(
    ratings
    .join(movies.hint('broadcast'), on='movieId', how='inner')
    .groupBy(F.col('movieId'), F.col('title'))
    .agg(F.avg(F.col('rating')).alias('avgRating'),
         F.count(F.col('*')).alias('nRatings'))
    .filter(F.col('nRatings') >= 100)
    .orderBy(F.col('avgRating').desc())
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT m.movieId, m.title, AVG(r.rating) AS avgRating, COUNT(*) AS nRatings
# MAGIC FROM ratings r INNER JOIN movies m ON m.movieId = r.movieId
# MAGIC GROUP BY m.movieId, m.title
# MAGIC HAVING nRatings >= 100
# MAGIC ORDER BY avgRating DESC
