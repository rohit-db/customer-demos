# Databricks notebook source
# MAGIC %md
# MAGIC ## Monthly AutoML Retrain
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/7_schedule.png?raw=true">

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# DBTITLE 1,Load Features
from databricks.feature_store import FeatureStoreClient

# Set config for database name, file paths, and table names
feature_table = f'{database_name}.churn_features'

fs = FeatureStoreClient()

featuresDF = fs.read_table(feature_table)

# COMMAND ----------

# DBTITLE 1,Create Temporary directory to store autoML experiments data
exp_tmp_path = f"{get_default_path()}/automl_exp"
try:
    dbutils.fs.mkdirs(exp_tmp_path)
except:
    print("Dir already exist")

# COMMAND ----------

# DBTITLE 1,Run AutoML
import databricks.automl

model = databricks.automl.classify(featuresDF.drop(key_id), 
                                   target_col="churn",
                                   data_dir=f"/dbfs/{exp_tmp_path}",
                                   primary_metric="f1",
                                   timeout_minutes=6)

# COMMAND ----------

# DBTITLE 1,Register the Best Run
import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

run_id = model.best_trial.mlflow_run_id
model_uri = f"runs:/{run_id}/model"

client.set_tag(run_id, key='feature_table', value=f'{database_name}.churn_features')
client.set_tag(run_id, key='demographic_vars', value='seniorCitizen,gender_Male')

model_details = mlflow.register_model(model_uri, churn_model_name)

# COMMAND ----------

# DBTITLE 1,Add Descriptions
model_version_details = client.get_model_version(name=churn_model_name, version=model_details.version)

client.update_registered_model(
  name=model_details.name,
  description=f"This model predicts whether a customer will churn using features from the {database_name} database.  It is used to update the Telco Churn Dashboard in DB SQL."
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description=f"This model version was built using {model.best_trial.model_description} for accuracy/F1 validation metric of {round(model.best_trial.metrics['val_f1_score'],2)*100}%"
)

# COMMAND ----------

# DBTITLE 1,Request Transition to Staging
staging_request = {'name': churn_model_name, 'version': model_details.version, 'stage': 'Staging', 'archive_existing_versions': 'true'}
mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests
comment = "Recurring AutoML training."
comment_body = {'name': churn_model_name, 'version': model_details.version, 'comment': comment}
mlflow_call_endpoint('comments/create', 'POST', json.dumps(comment_body))

# COMMAND ----------


