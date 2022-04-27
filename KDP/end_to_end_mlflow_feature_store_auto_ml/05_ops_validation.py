# Databricks notebook source
dbutils.widgets.dropdown("stage_mode", "None", ["Archived", "None", "Staging", "Production"], "Model Stage to test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Tests
# MAGIC In this notebook you can program a series of validation tests to run every time a model transition request is underway:
# MAGIC * Predict on test set
# MAGIC * Signature check
# MAGIC * Other fairness checks (e.g. accuracy by demographics)
# MAGIC * Model documentation (e.g. description, tags, artifcat location)
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/5_automated_ml.png?raw=true">

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch Model in Transition

# COMMAND ----------

from mlflow.tracking import MlflowClient
from databricks.feature_store import FeatureStoreClient

client = MlflowClient()
fs = FeatureStoreClient()

stage_mode = dbutils.widgets.get("stage_mode")

# COMMAND ----------

model_details = client.get_latest_versions(churn_model_name, [stage_mode])[0]
run_info = client.get_run(run_id=model_details.run_id)
model_version = model_details.version

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Validate prediction

# COMMAND ----------

# Read from feature store prod table?
data_source = run_info.data.tags['feature_table']
featuresDF = fs.read_table(data_source)

# COMMAND ----------

# Load model as a Spark UDF
model_uri = f'models:/{churn_model_name}/{model_version}'
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# Predict on a Spark DataFrame
try:
#   display(featuresDF.withColumn('predictions', loaded_model(*featuresDF.drop(key_id).columns)))
  display(featuresDF.withColumn('predictions', loaded_model(*featuresDF.columns)))
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="predicts", value=1)
except Exception: 
  print("Unable to predict on features.")
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="predicts", value=0)
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC #### Signature check
# MAGIC 
# MAGIC When working with ML models you often need to know some basic functional properties of the model at hand, such as “What inputs does it expect?” and “What output does it produce?”.  The model **signature** defines the schema of a model’s inputs and outputs. Model inputs and outputs can be either column-based or tensor-based. 
# MAGIC 
# MAGIC See [here](https://mlflow.org/docs/latest/models.html#signature-enforcement) for more details.

# COMMAND ----------

if not loaded_model.metadata.signature:
  print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="has_signature", value=0)
else:
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="has_signature", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Demographic accuracy
# MAGIC 
# MAGIC How does the model perform across various slices of the customer base?

# COMMAND ----------

import numpy as np
# df_features = featuresDF.withColumn('predictions', loaded_model(*featuresDF.drop(key_id).columns)).toPandas()
df_features = featuresDF.withColumn('predictions', loaded_model(*featuresDF.columns)).toPandas()
df_features['accurate'] = np.where(df_features.churn == df_features.predictions, 1, 0)

# Check run tags for demographic columns and accuracy in each segment
try:
  demographics = run_info.data.tags['demographic_vars'].replace(" ", "").split(",")
  slices = df_features.groupby(demographics).accurate.agg(acc = 'sum', obs = lambda x:len(x), pct_acc = lambda x:sum(x)/len(x))
  
  # Threshold for passing on demographics is 55%
  demo_test = "pass" if slices['pct_acc'].any() > 0.55 else "fail"
  
  # Set tags in registry
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="demo_test", value=demo_test)

  print(slices)
except KeyError:
  print("KeyError: No demographics_vars tagged with this model version.")
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="demo_test", value="none")
  pass

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Documentation 
# MAGIC Is the model documented visually and in plain english? 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Description check
# MAGIC 
# MAGIC Has the data scientist provided a description of the model being submitted?

# COMMAND ----------

# If there's no description or an insufficient number of charaters, tag accordingly
if not model_details.description:
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="has_description", value=0)
  print("Please add a description!")
elif not len(model_details.description) > 20:
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="has_description", value=0)
  print("Your description is too basic, sorry.  Please resubmit with more detail (40 char min).")
else:
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="has_description", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Artifact check
# MAGIC Has the data scientist logged supplemental artifacts along with the original model?

# COMMAND ----------

import os

# Create local directory 
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = client.download_artifacts(run_info.info.run_id, "", local_dir)

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
  client.set_model_version_tag(name=churn_model_name, version=model_version, key="has_artifacts", value=0)
  print("There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more.")
else:
  client.set_model_version_tag(name=churn_model_name, version=model_version, key = "has_artifacts", value = 1)
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC 
# MAGIC Here's a summary of the testing results:

# COMMAND ----------

results = client.get_model_version(churn_model_name, model_version)
results.tags

# COMMAND ----------

# MAGIC %md
# MAGIC Notify the Slack channel with the same webhook used to alert on transition change in MLflow.

# COMMAND ----------

# DBTITLE 1,Get slack webhook URL
# MAGIC %run ./Includes/Slack-Webhook

# COMMAND ----------


import requests, json

slack_message = "Registered model '{}' version {} baseline test results: {}".format(churn_model_name, model_version, results.tags)
#slack_webhook = dbutils.secrets.get("rk_webhooks", "slack")

body = {'text': slack_message}
response = requests.post(
    slack_webhook, data=json.dumps(body),
    headers={'Content-Type': 'application/json'}
)
if response.status_code != 200:
    raise ValueError(
        'Request to slack returned an error %s, the response is:\n%s'
        % (response.status_code, response.text)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Move to Staging or Archived
# MAGIC 
# MAGIC The next phase of this models' lifecycle will be to `Staging` or `Archived`, depending on how it fared in testing.

# COMMAND ----------

# MAGIC %md
# MAGIC **If any checks failed, reject and move to Archived**

# COMMAND ----------

# DBTITLE 1,Create body of json payload
request_body = {'name': model_details.name,
                'version': model_details.version,
                'stage': 'Staging',
                'archive_existing_versions': 'true',
                'comment': 'All tests passed!  Moving to staging.'}
method = 'approve'

if '0' in results or 'fail' in results:
  method = 'reject'
  request_body.pop('archive_existing_versions')
  request_body['comment'] = 'Tests failed - check the tags or the job run to see what happened.'

# COMMAND ----------

mlflow_call_endpoint(f'transition-requests/{method}', 'POST', json.dumps(request_body))

# COMMAND ----------


