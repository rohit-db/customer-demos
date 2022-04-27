# Databricks notebook source
# MAGIC %md ---
# MAGIC title: Feature Store Telco Churn
# MAGIC authors:
# MAGIC - Sean Owen
# MAGIC tags:
# MAGIC - python
# MAGIC - sql
# MAGIC - delta
# MAGIC - machine-learning
# MAGIC - classification
# MAGIC - feature-engineering
# MAGIC - pandas
# MAGIC - scikit-learn
# MAGIC - telco
# MAGIC - churn
# MAGIC - feature-store
# MAGIC - mysql
# MAGIC - rds
# MAGIC created_at: 2021-07-12
# MAGIC updated_at: 2021-07-12
# MAGIC tldr: Demonstration of the feature store as applied to a churn prediction problem using telco data. Shows creating feature tables, updating features, registering updated models with different features, and publishing to an online store
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Links
# MAGIC - AWS demo.cloud: [https://demo.cloud.databricks.com/#notebook/10566930](https://demo.cloud.databricks.com/#notebook/10566930)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Store: Telco Churn example
# MAGIC 
# MAGIC This simple example will build a feature store on top of data from a telco customer churn data set: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv and then use it to train a model and deploy both the model and features to production.
# MAGIC 
# MAGIC The goal is to produce a service that can predict in real-time whether a customer churns.
# MAGIC 
# MAGIC We begin by assuming the data engineers have already performed basic cleaning of the data, to result in data like this:

# COMMAND ----------

# This name is used for a database and for a temp directory in DBFS.
DATABASE_NAME = "oetrta"

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")

# COMMAND ----------

# MAGIC %sh wget --directory-prefix /tmp https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

dbutils.fs.mkdirs(f"dbfs:/tmp/{DATABASE_NAME}")

# COMMAND ----------

dbutils.fs.mv("file:/tmp/Telco-Customer-Churn.csv", f"dbfs:/tmp/{DATABASE_NAME}/")
dbutils.fs.ls(f"dbfs:/tmp/{DATABASE_NAME}/")

# COMMAND ----------

import pyspark.sql.functions as F

telco_df = spark.read.option("header", True).option("inferSchema", True).csv(f"dbfs:/tmp/{DATABASE_NAME}/Telco-Customer-Churn.csv")

# 0/1 -> boolean
telco_df = telco_df.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)
# Yes/No -> boolean
for yes_no_col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]:
  telco_df = telco_df.withColumn(yes_no_col, F.col(yes_no_col) == "Yes")
# Contract categorical -> duration in months
telco_df = telco_df.withColumn("Contract",\
    F.when(F.col("Contract") == "Month-to-month", 1).\
    when(F.col("Contract") == "One year", 12).\
    when(F.col("Contract") == "Two year", 24))
# Empty TotalCharges -> NaN
telco_df = telco_df.withColumn("TotalCharges",\
    F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, None).\
    otherwise(F.col("TotalCharges").cast('double')))

display(telco_df.cache().summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining Feature Store Tables
# MAGIC 
# MAGIC This raw data set can be usefully improved for data science and machine learning by:
# MAGIC 
# MAGIC - Splitting it into logical, reusable subsets of columns
# MAGIC - Engineering more useful features
# MAGIC - Publishing the result as feature store tables
# MAGIC 
# MAGIC First we define a set of features by simply selecting some demographic information from the data. This will become one feature store table when written later. A feature table is defined by a function that computes a DataFrame defining the features in the table, from a source 'raw' DataFrame. It can be called directly for testing; this by itself does not persist or publish features.

# COMMAND ----------

demographic_cols = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents"]

def compute_demographic_features(data):
  return data.select(demographic_cols)

demographics_df = compute_demographic_features(telco_df)
display(demographics_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, feature related to the customer's telco service are likewise selected. Note that each of these tables includes `customerID` as a key for joining. A little bit of feature engineering happens here - filling in 0 for `TotalCharges`.

# COMMAND ----------

def compute_service_features(data):
  service_cols = ["customerID"] + [c for c in data.columns if c not in ["Churn"] + demographic_cols]
  return data.select(service_cols).fillna({"TotalCharges": 0.0})

service_df = compute_service_features(telco_df)
display(service_df)

# COMMAND ----------

# MAGIC %md
# MAGIC What's left? only the label, `Churn`. We could consider this a feature and write it in a feature store table. However in this use case, it's of course the very thing to be predicted, so that doesn't quite feel right.
# MAGIC 
# MAGIC Further, there may be other information _can_ be supplied at inference time, but does not make sense to consider a feature to _look up_. 
# MAGIC 
# MAGIC For example, in this case, it may be useful to know whether the customer's last call to customer service was escalated to a manager. Eventually, this information might become available and could be written into a feature store table for lookup. However, this model is probably going to be used to decide whether a customer on the phone who is asking for escalation is likely to churn _right now_. It's not possible to lookup whether the call _was escalated_ because it hasn't happened or not yet.
# MAGIC 
# MAGIC In this (fictional) example, the feature `LastCallEscalated` is "looked up" (really, it's just made up, but pretend!) because it could be important to making a predictive model. It and `Churn` however are not written as a feature store table. They are simply stored as a (normal) table. It's perfectly usable in training. At inference time, this kind of information (without of course the label `Churn`) will be supplied in a request to the model.

# COMMAND ----------

# Made up rule for LastCallEscalated: because ~26% of customers churn, if churned, then 35% chance call was escalated.
# Otherwise 15% chance.

inference_data_df = telco_df.select("customerID", "Churn").withColumn("LastCallEscalated",
    F.when(F.col("Churn"), F.hash(F.col("customerID")) % 100 < 35).otherwise(F.hash(F.col("customerID")) % 100 < 15))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing Feature Store Tables
# MAGIC 
# MAGIC With the tables defined by functions above, next, the feature store tables need to be written out, first as Delta tables. These are the fundamental 'offline' data stores underpinning the feature store tables. Use the client to create the feature tables, defining metadata like which database and table the feature store table will write to, and importantly, its key(s).

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# COMMAND ----------

demographic_features_table = fs.create_table(
  name=f'{DATABASE_NAME}.demographic_features',
  primary_keys='customerID',
  schema=demographics_df.schema,
  description='Telco customer demographics')

service_features_table = fs.create_table(
  name=f'{DATABASE_NAME}.service_features',
  primary_keys='customerID',
  schema=service_df.schema,
  description='Telco customer services')

# COMMAND ----------

# MAGIC %md
# MAGIC The feature table functions themselves are called to add or update data in the feature store table via `write_table`:

# COMMAND ----------

fs.write_table(f"{DATABASE_NAME}.demographic_features", compute_demographic_features(telco_df))
fs.write_table(f"{DATABASE_NAME}.service_features", compute_service_features(telco_df))

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, save the "inference data" including label and escalation status:

# COMMAND ----------

inference_data_df.write.format("delta").mode("overwrite").save(f"/tmp/{DATABASE_NAME}/inference_data")

# COMMAND ----------

spark.sql(f"CREATE TABLE {DATABASE_NAME}.inference_data USING DELTA LOCATION '/tmp/{DATABASE_NAME}/inference_data'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Browsing the Feature Store
# MAGIC 
# MAGIC The tables are now visible and searchable in the [Feature Store](/#feature-store) -- try it!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building Models from a Feature Store
# MAGIC 
# MAGIC Next, it's time to build a simple model predicting whether the customer churns. There are two feature store tables containing information we want to join to the other 'inference data' in order to build a model:
# MAGIC 
# MAGIC - `demographic_features`
# MAGIC - `service_features`
# MAGIC 
# MAGIC We could of course simply read the data from the tables as a DataFrame and join them. However, in a feature store, we want to define the joins declaratively, because the same join will need to happen in a quite different context in online serving. More on that later. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### An initial model
# MAGIC 
# MAGIC Perhaps modeling begins by building a model on the most immediately relevant features available - actual service-related features. Below, a `FeatureLookup` is created for each feature in the `service_features` table. This defines how other features are looked up (table, feature names) and based on what key -- `customerID`.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup

fs = FeatureStoreClient()

service_features_table = fs.get_table(f"{DATABASE_NAME}.service_features")

# COMMAND ----------

# MAGIC %md
# MAGIC Now, modeling can proceed. `create_training_set` pulls together all the features. Its "input" is the `inference_data` table, as discussed above, and the `FeatureLookup`s define how to join in features. Modeling ignores the `customerID` of course; it's a key rather than useful data.
# MAGIC 
# MAGIC The modeling here is simplistic, and just trains a plain `sklearn` gradient boosting classifier.
# MAGIC 
# MAGIC Normally, this model would be logged (or auto-logged) by MLflow. It's necessary however to log to MLflow through the client's `log_model` method instead. The Feature Store needs to package additional metadata about feature lookups and feature tables in order for the model to perform lookups and joins correctly at inference time, when deployed as a service.
# MAGIC 
# MAGIC Note that code from `read_table` through to `log_model` must be inside an MLflow run. When complete, a new run has been logged and registered in the MLflow Model Registry.

# COMMAND ----------

import mlflow
import mlflow.shap
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.autolog(log_input_examples=True) # Optional

# Define a method for reuse later
def fit_model(model_feature_lookups):

  with mlflow.start_run():
    inference_data_df = spark.read.table(f"{DATABASE_NAME}.inference_data")
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="Churn", exclude_columns="customerID")

    training_pd = training_set.load_df().toPandas()
    X = training_pd.drop("Churn", axis=1)
    y = training_pd["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Churn is relatively rare; let's weight equally to start as it's probably at least as important as not-churning
    churn_weight = 1.0 / y_train.sum()
    not_churn_weight = 1.0 / (len(y) - y_train.sum())
    sample_weight = y_train.map(lambda churn: churn_weight if churn else not_churn_weight)

    # Not attempting to tune the model at all for purposes here
    gb_classifier = GradientBoostingClassifier(n_iter_no_change=10)
    # Need to encode categorical cols
    encoders = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), X.columns[X.dtypes == 'object'])])
    pipeline = Pipeline([("encoder", encoders), ("gb_classifier", gb_classifier)])
    pipeline_model = pipeline.fit(X_train, y_train, gb_classifier__sample_weight=sample_weight)

    mlflow.log_metric('test_accuracy', pipeline_model.score(X_test, y_test))
    # mlflow.shap.log_explanation(gb_classifier.predict, encoders.transform(X))

    fs.log_model(
      pipeline_model,
      "model",
      flavor=mlflow.sklearn,
      training_set=training_set,
      registered_model_name=f"{DATABASE_NAME}_telco_churn",
      input_example=X[:100],
      signature=infer_signature(X, y))
      
fit_model([FeatureLookup(table_name = service_features_table.name, lookup_key = 'customerID')])

# COMMAND ----------

# MAGIC %md
# MAGIC It's trivial to apply a registered MLflow model to features with `score_batch`. Again its only input are `customerID`s and `LastCallEscalated` (without the label `Churn` of course!). Everything else is looked up. However, eventually the goal is to produce a real-time service from this model and these features.

# COMMAND ----------

from pyspark.sql.functions import col

batch_input_df = inference_data_df.select("customerID", "LastCallEscalated")
with_predictions = fs.score_batch(f"models:/{DATABASE_NAME}_telco_churn/1", batch_input_df, result_type='string')
with_predictions = with_predictions.withColumn("prediction", col("prediction") == "True")
display(with_predictions.join(inference_data_df.select("customerID", "Churn"), on="customerID"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### An improved model?
# MAGIC 
# MAGIC Of course, there were more features available above, demographic features. In the next iteration, maybe try adding those features?

# COMMAND ----------

demographic_features_table = fs.get_table(f"{DATABASE_NAME}.demographic_features")
service_features_table = fs.get_table(f"{DATABASE_NAME}.service_features")

fit_model([FeatureLookup(table_name = service_features_table.name, lookup_key = 'customerID'),
           FeatureLookup(table_name = demographic_features_table.name, lookup_key = 'customerID')])

# COMMAND ----------

# MAGIC %md
# MAGIC The model doesn't improve much in accuracy, but perhaps slightly. A win? We can generate batch predictions again:

# COMMAND ----------

with_predictions = fs.score_batch(f"models:/{DATABASE_NAME}_telco_churn/2", batch_input_df, result_type='string')
with_predictions = with_predictions.withColumn("prediction", col("prediction") == "True")
display(with_predictions.join(inference_data_df.select("customerID", "Churn"), on="customerID"))

# COMMAND ----------

# MAGIC %md
# MAGIC Notice now in the feature store that the demographic features table shows up as used only by the second version of this model, not the first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Updating Feature Store Tables
# MAGIC 
# MAGIC Imagine sometime later, it's necessary to engineer more features. It's discovered that two more derived features would be useful to the model:
# MAGIC 
# MAGIC - `NumOptionalServices`: the count of optional services the customer has, like `StreamingTV` or `StreamingMovies`
# MAGIC - `AvgPriceIncrease`: the customer's current monthly charges, relative to their historical average monthly charge
# MAGIC 
# MAGIC `compute_service_features` is redefined to compute these new features:

# COMMAND ----------

def compute_service_features(data):
  # Count number of optional services enabled, like streaming TV
  @F.pandas_udf('int')
  def num_optional_services(*cols):
    return sum(map(lambda s: (s == "Yes").astype('int'), cols))
  
  # Below also add AvgPriceIncrease: current monthly charges compared to historical average
  service_cols = ["customerID"] + [c for c in data.columns if c not in ["Churn"] + demographic_cols]
  return data.select(service_cols).fillna({"TotalCharges": 0.0}).\
    withColumn("NumOptionalServices",
        num_optional_services("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")).\
    withColumn("AvgPriceIncrease",
        F.when(F.col("tenure") > 0, (F.col("MonthlyCharges") - (F.col("TotalCharges") / F.col("tenure")))).otherwise(0.0))

display(compute_service_features(telco_df))

# COMMAND ----------

# MAGIC %md
# MAGIC Now, with the new definitions of features in hand, `write_table` can add the new features by `merge`-ing them into the feature store table.

# COMMAND ----------

fs.write_table(f"{DATABASE_NAME}.service_features", compute_service_features(telco_df), mode="merge")

# COMMAND ----------

# MAGIC %md
# MAGIC Try modeling again:

# COMMAND ----------

demographic_features_table = fs.get_table(f"{DATABASE_NAME}.demographic_features")
service_features_table = fs.get_table(f"{DATABASE_NAME}.service_features")

fit_model([FeatureLookup(table_name = service_features_table.name, lookup_key = 'customerID'),
           FeatureLookup(table_name = demographic_features_table.name, lookup_key = 'customerID')])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Publishing Feature Store Tables
# MAGIC 
# MAGIC So far, the features exist 'offline' as Delta tables. It's been very convenient to define features and compute them with Spark and persist them for batch use. However, these features need to be available 'online' too. It would be prohibitively slow recompute features in real-time. The features are already computed, but exist in tables which may be slow to read individual rows from; the Delta tables may not be accessible, even, from the real-time serving context!
# MAGIC 
# MAGIC The Feature Store can 'publish' the data in these tables to an external store that is more suitable for fast online lookups. `publish_table` does this.
# MAGIC 
# MAGIC **Note**: This will require the `oetrta-IAM-access` IAM role for RDS access.

# COMMAND ----------

from databricks.feature_store.online_store_spec import AmazonRdsMySqlSpec
from databricks.feature_store import FeatureStoreClient

# Only run this if you intend to use the Databricks-internal oetrta tooling.
if DATABASE_NAME == "oetrta":
  fs = FeatureStoreClient()
  for table in ["demographic_features", "service_features"]:
    online_store = AmazonRdsMySqlSpec(
      "oetrta-mysql.cmeifwaki1jl.us-west-2.rds.amazonaws.com", 
      3306, 
      read_secret_prefix=f"{DATABASE_NAME}/fs-demo", # really, should be a separate read-only role
      write_secret_prefix=f"{DATABASE_NAME}/fs-demo", 
      database_name=DATABASE_NAME, 
      table_name=table
    )
    fs.publish_table(name=f'{DATABASE_NAME}.{table}', online_store=online_store)

# COMMAND ----------

# MAGIC %md
# MAGIC This becomes visible in the feature table's UI as an "Online Store" also containing the same data.
