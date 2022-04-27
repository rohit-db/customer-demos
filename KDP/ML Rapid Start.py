# Databricks notebook source
# MAGIC %md
# MAGIC ## ETL
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/1_data_prep.png?raw=true">

# COMMAND ----------

# MAGIC %md 
# MAGIC # Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning 10.0 ML or above.

# COMMAND ----------

# MAGIC %md 
# MAGIC # Setup

# COMMAND ----------

# MAGIC %run ./end_to_end_mlflow_feature_store_auto_ml/Shared_Include

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -P /tmp https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# Verify that the file was downloaded 
display(dbutils.fs.ls("file:/tmp/Telco-Customer-Churn.csv"))

# COMMAND ----------

# Load libraries
import shutil
import numpy as np # linear algebra
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType, FloatType

# Copy file from driver to DBFS
driver_to_dbfs_path = f'dbfs:{get_default_path()}/Telco-Customer-Churn.csv'
dbutils.fs.cp('file:/tmp/Telco-Customer-Churn.csv', driver_to_dbfs_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Data
# MAGIC 
# MAGIC First we'll import the data. 
# MAGIC We'll look at how to read the data natively using Spark but also using Pandas API on Spark. 

# COMMAND ----------

# Import Data using Spark 

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
display(bronze_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pandas API on Spark
# MAGIC Starting in Spark 3.2, the pandas API is available in PySpark. This allows data scientists to leverage the pandas API along with Spark to analyze large datasets with the familiar Pandas syntax.

# COMMAND ----------

import pyspark.pandas as ps

ps_bronze_df = ps.read_csv(driver_to_dbfs_path)

# Familiar pandas dataframe and its methods
# ps_bronze_df.columns, ps_bronze_df.describe()

display(ps_bronze_df)
display(ps_bronze_df.describe())

# COMMAND ----------

# Run SQL on Pandas Dataframe
from pyspark.pandas import sql
sql("""SELECT gender, count(1) FROM {ps_bronze_df} group by gender""")

# COMMAND ----------

# Convert back to Spark Dataframe 
ps_bronze_df.to_spark()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load data into Bronze table 

# COMMAND ----------

bronze_df.write.format('delta').mode('overwrite').save(bronze_tbl_path)

# COMMAND ----------

# MAGIC %sql 
# MAGIC create table rb_kdp_demo.bronze_customers
# MAGIC USING DELTA LOCATION '/tmp/rohit_bhagwat/ibm-telco-churn/bronze/raw'

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from rohit_bhagwat_churn_demo.bronze_customers

# COMMAND ----------

# MAGIC %md 
# MAGIC # Exploratory Data Analysis & Data Visualization
# MAGIC Before a data scientist can write a report on analytics or train a machine learning (ML) model, they need to understand the shape and content of their data. This exploratory data analysis often involves the same basic analyses: visualizing data distributions and computing summary statistics like row count, null count, mean, item frequencies, etc.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Profile Tab
# MAGIC Data teams working on a cluster running **DBR 9.1 or newer** can generate data profiles in the Notebook via the cell output UI. When viewing the contents of a data frame using the Databricks display function, users will see a “Data Profile” tab to the right of the “Table” tab in the cell output. Clicking on this tab will automatically execute a new command that generates a profile of the data in the data frame. The profile will include summary statistics for numeric, string, and date columns as well as histograms of the value distributions for each column. Note that this command will profile the entire data set in the data frame or SQL query results, not just the portion displayed in the table (which can be truncated).

# COMMAND ----------

display(bronze_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Or use Pandas Profiling if it fits in memory

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(bronze_df.toPandas(), title="Telco-Churn Raw Data Exploration", progress_bar=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------

# MAGIC %md ## Data Visualizations
# MAGIC Lets explore the dataset using Seaborn and Matplotlib, which are included in the Databricks ML Runtime.
# MAGIC 
# MAGIC We are going to convert our Pyspark on Pandas dataframe to a dataframe for use with single node libraries like Seaborn and scikit-learn. For large datasets it is best to continue with a Spark Dataframe and use a distributed training option such as  [Spark ML](https://spark.apache.org/docs/latest/ml-guide.html).

# COMMAND ----------

# MAGIC %md # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization Logic
# MAGIC 
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

# Read into Spark
raw_df = spark.table(f"{database_name}.bronze_customers")

# COMMAND ----------

# DBTITLE 1,Define featurization function
import pyspark.pandas as ps
def compute_churn_features(data):
  
  # Convert to koalas (CONSIDER UPDATING THIS PIECE TO USE NATIVE PANDAS API STARTING DBR10.0+)
  data = data.to_pandas_on_spark()
  
  # One-Hot-Encoding while dropping 1 category to avoid colinearities
  data = ps.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phoneService', 'multipleLines', 'internetService',
                                 'onlineSecurity', 'onlineBackup', 'deviceProtection',
                                 'techSupport', 'streamingTV', 'streamingMovies',
                                 'contract', 'paperlessBilling', 'paymentMethod'],
                        drop_first=True,
                        dtype='int64')
  
  # Convert label to int and rename column
  data['churnString'] = data['churnString'].map({'Yes': 1, 'No': 0})
  data = data.astype({'churnString': 'int32'})
  data = data.rename(columns = {'churnString': 'churn'})
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '')
  data.columns = data.columns.str.replace('(', '-')
  data.columns = data.columns.str.replace(')', '')
  
  # Drop missing values
  data = data.dropna()
  
  return data

data = compute_churn_features(raw_df).drop('customerID')

display(data)

# COMMAND ----------

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train, test = train_test_split(data.to_pandas(), test_size=0.30, random_state=206)
colLabel = 'churn'

# The predicted column is colLabel which is a scalar from [3, 9]
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

display(train_x)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit model and log with MLflow
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png" height = 1200 width = 800>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Wrappers around your training code

# COMMAND ----------

# Set experiment
mlflow.set_experiment(f"/Users/{current_user_name}/first_churn_experiment")

# Begin training run
with mlflow.start_run(run_name="sklearn") as mlflow_run:
    run_id = mlflow_run.info.run_uuid
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",mlflow_run.info.experiment_id)
    
    # Fit model
    model = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=32)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    
    # Get metrics
    acc = accuracy_score(predictions, test_y)
    print("Metrics:")
    print("  mean accuracy:",acc)
    
    # Log
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("max_leaf_nodes", 32)
    mlflow.log_metric("mean_acc", acc)
        
    mlflow.sklearn.log_model(model, "sklearn-model")

# COMMAND ----------

# MAGIC %md 
# MAGIC **Databricks Autologging is a no-code solution that extends MLflow automatic logging to deliver automatic experiment tracking for machine learning training sessions on Databricks.** 
# MAGIC 
# MAGIC With Databricks Autologging, model parameters, metrics, files, and lineage information are automatically captured when you train models from a variety of popular machine learning libraries. Training sessions are recorded as MLflow tracking runs. Model files are also tracked so you can easily log them to the MLflow Model Registry and deploy them for real-time scoring with MLflow Model Serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### With auto-logging

# COMMAND ----------

# Turn on auto-logging
mlflow.sklearn.autolog()

# Fit model
model = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=32)
model.fit(train_x, train_y)

# COMMAND ----------

# MAGIC %md ### MLflow Model Registry
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png" height = 1200 width = 800>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote to Registry

# COMMAND ----------

import mlflow.pyfunc

# Grab the run ID from a prior run to promote artifact in tracking server to registry
run_id = "b46aac475ac44f0088dd5bb4bacc97ad"

model_uri = f"runs:/{run_id}/model"
model_details = mlflow.register_model(model_uri, "{churn_model_name_manual}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Load from Registry

# COMMAND ----------

# Load model version 1 and predict!
model = mlflow.pyfunc.load_model(f"models:/{churn_model_name_manual}/1")
model.predict(test_x)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Feature Store (Optional)

# COMMAND ----------

# DBTITLE 1,Define featurization function
from databricks.feature_store import feature_table

@feature_table
def compute_churn_features(data):
  
  # Convert to koalas (CONSIDER UPDATING THIS PIECE TO USE NATIVE PANDAS API STARTING DBR10.0+)
  data = data.to_pandas_on_spark()
  
  # One-Hot-Encoding while dropping 1 category to avoid colinearities
  data = ps.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phoneService', 'multipleLines', 'internetService',
                                 'onlineSecurity', 'onlineBackup', 'deviceProtection',
                                 'techSupport', 'streamingTV', 'streamingMovies',
                                 'contract', 'paperlessBilling', 'paymentMethod'],
                        drop_first=True,
                        dtype='int64')
  
  # Convert label to int and rename column
  data['churnString'] = data['churnString'].map({'Yes': 1, 'No': 0})
  data = data.astype({'churnString': 'int32'})
  data = data.rename(columns = {'churnString': 'churn'})
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '')
  data.columns = data.columns.str.replace('(', '-')
  data.columns = data.columns.str.replace(')', '')
  
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# DBTITLE 1,Compute Features
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

churn_features_df = compute_churn_features(raw_df)
display(churn_features_df)

# COMMAND ----------

# DBTITLE 1,Create Feature Table in Offline Store (DELTA) and Feature Store - DO ONCE
churn_features_table_name = f"{database_name}.churn_features"
print(churn_features_table_name)
churn_feature_table = fs.create_table(
  name=churn_features_table_name,
  primary_keys='customerID',
  schema=churn_features_df.spark.schema(),
  description=f'These features are derived from the {database_name}.bronze_customers table in the lakehouse. Categorical columns were One-Hot-Encoded with drop, column names were cleaned up, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)
# Push to Feature Store 
fs.write_table(churn_features_table_name, churn_features_df.to_spark(), mode="overwrite")

# COMMAND ----------

# Read table 
display(fs.read_table("rohit_bhagwat_churn_demo.churn_features"))
# Get Metadata of feature table print(fs.get_table("rohit_bhagwat_churn_demo.churn_features"))

# COMMAND ----------

# Drop customer ID for AutoML
automlDF = churn_features_df.drop(key_id)

# Write out silver-level data to autoML Delta lake
automlDF.to_delta(mode='overwrite', path=automl_silver_tbl_path)

# Create autoML table
_ = spark.sql(f''' DROP TABLE IF EXISTS `{database_name}`.{automl_silver_tbl_name} ''')
_ = spark.sql(f''' CREATE TABLE `{database_name}`.{automl_silver_tbl_name}
  USING DELTA 
  LOCATION '{automl_silver_tbl_path}' ''')
print(database_name, automl_silver_tbl_name)
