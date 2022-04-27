# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png" height = 1200 width = 800>

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read and prep data

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read churn features dataset (silver)
data = spark.table(f"{database_name}.{automl_silver_tbl_name}").toPandas()

train, test = train_test_split(data, test_size=0.30, random_state=206)
colLabel = 'churn'

# The predicted column is colLabel which is a scalar from [3, 9]
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit model and log with MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Wrappers around your training code

# COMMAND ----------

# Set experiment
mlflow.set_experiment(f"/Users/{current_user_name}/first_churn_experiment")

# Begin training run
with mlflow.start_run(run_name="sklearn") as run:
    run_id = run.info.run_uuid
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",run.info.experiment_id)
    
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
# MAGIC ##### With auto-logging

# COMMAND ----------

# Turn on auto-logging
mlflow.sklearn.autolog()

# Fit model
model = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=32)
model.fit(train_x, train_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### With AUTO-auto-logging :)

# COMMAND ----------

model = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=32)
model.fit(train_x, train_y)

# COMMAND ----------

# MAGIC %md 
# MAGIC **Databricks Autologging is a no-code solution that extends MLflow automatic logging to deliver automatic experiment tracking for machine learning training sessions on Databricks.** 
# MAGIC 
# MAGIC With Databricks Autologging, model parameters, metrics, files, and lineage information are automatically captured when you train models from a variety of popular machine learning libraries. Training sessions are recorded as MLflow tracking runs. Model files are also tracked so you can easily log them to the MLflow Model Registry and deploy them for real-time scoring with MLflow Model Serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Autologging options and configuration

# COMMAND ----------

mlflow.autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=True,
    disable_for_unsupported_versions=True,
    silent=True
)

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


