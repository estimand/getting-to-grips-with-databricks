# Databricks notebook source
# MAGIC %md
# MAGIC # Data download
# MAGIC ## *Getting to grips with Databricks*
# MAGIC
# MAGIC By Gianluca Campanella (<g.campanella@estimand.com>)
# MAGIC
# MAGIC [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

# COMMAND ----------

# MAGIC %md
# MAGIC Download the [latest MovieLens dataset](https://grouplens.org/datasets/movielens/latest/) using the magic command `%sh`.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget http://files.grouplens.org/datasets/movielens/ml-latest.zip

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip `ml-latest.zip` on the driver node and see what it contains.

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip ml-latest.zip
# MAGIC rm ml-latest.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lah ml-latest/

# COMMAND ----------

# MAGIC %md
# MAGIC Move these files to DBFS using the functions in `dbutils.fs`.

# COMMAND ----------

driverLocation = 'file:/databricks/driver/ml-latest/'
dbfsLocation = 'dbfs:/movielens/'

dbutils.fs.mkdirs(dbfsLocation)

for file in dbutils.fs.ls(driverLocation):
    dbutils.fs.mv(file.path, dbfsLocation + file.name)

dbutils.fs.rm(driverLocation, True)

# COMMAND ----------

# MAGIC %md
# MAGIC Confirm that the files are not on the driver any more.

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lah

# COMMAND ----------

# MAGIC %md
# MAGIC Check that the files are on DBFS using the magic command `%fs`.

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /movielens
