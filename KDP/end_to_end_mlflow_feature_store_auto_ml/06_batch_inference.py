# Databricks notebook source
dbutils.widgets.dropdown("stage_mode", "Staging", ["Archived", "None", "Staging", "Production"], "Model Stage to test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Churn Prediction Batch Inference
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/6_run_inference.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Model
# MAGIC 
# MAGIC Loading as a Spark UDF to set us up for future scale.

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

stage = dbutils.widgets.get("stage_mode")
model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{churn_model_name}/{stage}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Features

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
featuresDF = fs.read_table(f'{database_name}.churn_features')

# COMMAND ----------

display(featuresDF.drop(key_id))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference

# COMMAND ----------

predictionsDF = featuresDF.withColumn('predictions', model(*featuresDF.columns))
display(predictionsDF.select(key_id, "predictions"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write to Delta Lake

# COMMAND ----------

predictionsDF.write.format("delta").mode("append").saveAsTable(f"{database_name}.churn_preds")

# COMMAND ----------


