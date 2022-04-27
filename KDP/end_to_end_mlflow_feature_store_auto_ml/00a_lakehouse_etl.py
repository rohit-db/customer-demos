# Databricks notebook source
# MAGIC %md
# MAGIC ## ETL
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/1_data_prep.png?raw=true">

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup
# MAGIC 
# MAGIC Grab the CSV from the web (DO ONCE)

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -P /tmp https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# DBTITLE 1,Verify that file was downloaded
display(dbutils.fs.ls("file:/tmp/Telco-Customer-Churn.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load into Delta Lake

# COMMAND ----------

# MAGIC %md
# MAGIC #### Path configs

# COMMAND ----------

# Load libraries
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType, FloatType

# Copy file from driver to DBFS
driver_to_dbfs_path = f'dbfs:{get_default_path()}/Telco-Customer-Churn.csv'
dbutils.fs.cp('file:/tmp/Telco-Customer-Churn.csv', driver_to_dbfs_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read and display

# COMMAND ----------

# Define schema
schema = StructType([
  StructField('customerID', StringType()),
  StructField('gender', StringType()),
  StructField('seniorCitizen', DoubleType()),
  StructField('partner', StringType()),
  StructField('dependents', StringType()),
  StructField('tenure', DoubleType()),
  StructField('phoneService', StringType()),
  StructField('multipleLines', StringType()),
  StructField('internetService', StringType()), 
  StructField('onlineSecurity', StringType()),
  StructField('onlineBackup', StringType()),
  StructField('deviceProtection', StringType()),
  StructField('techSupport', StringType()),
  StructField('streamingTV', StringType()),
  StructField('streamingMovies', StringType()),
  StructField('contract', StringType()),
  StructField('paperlessBilling', StringType()),
  StructField('paymentMethod', StringType()),
  StructField('monthlyCharges', DoubleType()),
  StructField('totalCharges', DoubleType()),
  StructField('churnString', StringType())
  ])

# Read CSV, write to Delta and take a look
bronze_df = spark.read.format('csv').schema(schema).option('header','true')\
               .load(driver_to_dbfs_path)

bronze_df.write.format('delta').mode('overwrite').save(bronze_tbl_path)

display(bronze_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create bronze DELTA table

# COMMAND ----------

# Create bronze table

(spark.sql(f''' CREATE TABLE {database_name}.{bronze_tbl_name} 
USING DELTA LOCATION '{bronze_tbl_path}' '''))

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from rohit_bhagwat_churn_demo.bronze_customers

# COMMAND ----------


