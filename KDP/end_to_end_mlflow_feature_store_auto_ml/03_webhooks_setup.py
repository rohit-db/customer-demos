# Databricks notebook source
dbutils.widgets.text("job_id", "1006168833879334", "Job ID")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry Webhooks
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/3_webhook.png?raw=true">
# MAGIC 
# MAGIC ### Supported Events
# MAGIC * Registered model created
# MAGIC * Model version created
# MAGIC * Transition request created
# MAGIC * Accept/Reject transition request
# MAGIC * Comment on a model version
# MAGIC 
# MAGIC ### Types of webhooks
# MAGIC * HTTP webhook -- send triggers to endpoints of your choosing such as slack, AWS Lambda, Azure Functions, or GCP Cloud Functions
# MAGIC * Job webhook -- trigger a job within the Databricks workspace
# MAGIC 
# MAGIC ### Use Cases
# MAGIC * Automation - automated introducing a new model to accept shadow traffic, handle deployments and lifecycle when a model is registered, etc..
# MAGIC * Model Artifact Backups - sync artifacts to a destination such as S3 or ADLS
# MAGIC * Automated Pre-checks - perform model tests when a model is registered to reduce long term technical debt
# MAGIC * SLA Tracking - Automatically measure the time from development to production including all the changes inbetween

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Webhooks
# MAGIC In reality, this notebook/commands only need to be run once every time a new model gets created in the registry, so webhooks can attached at specific events of the model life-cycle. 
# MAGIC 
# MAGIC ___
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/webhooks2.png?raw=true" width = 600>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transition Request Created
# MAGIC 
# MAGIC These fire whenever a transition request is created for a model.
# MAGIC 
# MAGIC **Pre-Requesite(s)**:
# MAGIC * From autoML: take best run and push to the registry as the same `churn_model_name` variable
# MAGIC 
# MAGIC OR
# MAGIC * If no model was registered yet (first time running), create an empty model in the Model Registry and name it as `churn_model_name`

# COMMAND ----------

# DBTITLE 1,IF Model was not registered (manually) after AutoML experiment: Create empty model
result_create_empty_model = client().create_registered_model(churn_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Trigger Job at Model Transition
# MAGIC **Pre-Requesite**:
# MAGIC * Have a notebook (i.e. `05_ops_validation`) already scheduled to run as a job on a job cluster and get the job ID or use default [job](https://e2-demo-west.cloud.databricks.com/?o=2556758628403379#job/87331)

# COMMAND ----------

trigger_job = json.dumps({
  "model_name": churn_model_name,
  "events": ["TRANSITION_REQUEST_CREATED"],
  "description": "Trigger the ops_validation job when a model is requested to transition.",
  "status": "ACTIVE",
  "job_spec": {
    "job_id": dbutils.widgets.get("job_id"),
    "workspace_url": host,
    "access_token": token
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_job)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notifications
# MAGIC 
# MAGIC Webhooks can be used to send emails, Slack messages, and more.  In this case we use Slack.  We also use `dbutils.secrets` to not expose any tokens, but the URL looks more or less like this:
# MAGIC 
# MAGIC `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX`
# MAGIC 
# MAGIC **Pre-Requesites**:
# MAGIC 1. Create a new Slack Workspace (call it MLOps-Env for example)
# MAGIC 2. Create a Slack App in this new workspace **(NOT Databricks workspace)**, activate webhooks and copy the URL - more info [here](https://api.slack.com/messaging/webhooks#create_a_webhook).
# MAGIC 3. Create a secret (scope:"my-databricks-scope", key:"slack-webhook", string_value:"https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX") using the [Secrets API](https://docs.databricks.com/dev-tools/api/latest/secrets.html#secretsecretservicecreatescope)

# COMMAND ----------

# DBTITLE 0,Alternatively update the .Includes/Slack-webhook notebook and run this for demo purposes (REMOVE THIS IN THE FUTURE)
# MAGIC %run ./Includes/Slack-Webhook

# COMMAND ----------

try:
#   slack_webhook = dbutils.secrets.get(scope="my-databricks-scope", key="slack-webhook")
  slack_webhook = slack_webhook
except:
  slack_webhook = None

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Creation Notification

# COMMAND ----------

if slack_webhook:
  trigger_slack = json.dumps({
    "model_name": churn_model_name,
    "events": ["REGISTERED_MODEL_CREATED"],
    "description": "Notify the MLOps team that a new model was created/registered.",
    "status": "ACTIVE",
    "http_url_spec": {
      "url": slack_webhook
    }
  })

  mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Transition Request Notification

# COMMAND ----------

if slack_webhook:
  trigger_slack = json.dumps({
    "model_name": churn_model_name,
    "events": ["TRANSITION_REQUEST_CREATED"],
    "description": "Notify the MLOps team that a model transition is requested.",
    "status": "ACTIVE",
    "http_url_spec": {
      "url": slack_webhook
    }
  })

  mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Version Transition Notification

# COMMAND ----------

if slack_webhook:
  trigger_slack = json.dumps({
    "model_name": churn_model_name,
    "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
    "description": "Notify the MLOps team that a model has succesfully transitioned.",
    "http_url_spec": {
      "url": slack_webhook
    }
  })

  mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Manage Webhooks

# COMMAND ----------

# DBTITLE 1,List active webhooks
list_model_webhooks = json.dumps({"model_name": churn_model_name})

mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Delete
# MAGIC From the list above copy/paste the `id` for the event that you'd like to delete

# COMMAND ----------

# Remove a webhook
mlflow_call_endpoint("registry-webhooks/delete",
                     method="DELETE",
                     body = json.dumps({'id': ''}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow Model Registry?  
# MAGIC **A:** Check out <a href="https://mlflow.org/docs/latest/registry.html#concepts" target="_blank"> for the latest API docs available for Model Registry</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
