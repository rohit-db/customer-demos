# Databricks notebook source
dbutils.widgets.text("experiment_id", "1366206427923884", "Experiment ID (Optional)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Managing the model lifecycle with Model Registry
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/4_promote_best_run.png?raw=true">
# MAGIC 
# MAGIC One of the primary challenges among data scientists and ML engineers is the absence of a central repository for models, their versions, and the means to manage them throughout their lifecycle.  
# MAGIC 
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) addresses this challenge and enables members of the data team to:
# MAGIC <br><br>
# MAGIC * **Discover** registered models, current stage in model development, experiment runs, and associated code with a registered model
# MAGIC * **Transition** models to different stages of their lifecycle
# MAGIC * **Deploy** different versions of a registered model in different stages, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * **Test** models in an automated fashion
# MAGIC * **Document** models throughout their lifecycle
# MAGIC * **Secure** access and permission for model registrations, transitions or modifications
# MAGIC 
# MAGIC <!--<img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg"> -->

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automatically pull best run for given input experiment
# MAGIC _otherwise if you know the `run_id`skip this section and jump to other by manually setting the `run_id` and updating the `best_score` and `run_name` variables_

# COMMAND ----------

from mlflow.entities import ViewType
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

best_model = client.search_runs(
  experiment_ids=dbutils.widgets.get("experiment_id"),
  filter_string="",
  run_view_type=ViewType.ACTIVE_ONLY,
  max_results=1,
  order_by=["metrics.val_f1_score DESC"]
)[0]

best_score = best_model.data.metrics["val_f1_score"]
run_name = best_model.data.tags["mlflow.runName"]

print(f'F1 of Best Run: {best_score}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use the Model Registry
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters. During the course of this development cycle, they will select the best run within an experiment and register its model with the registry.  Think of this as **committing** the model to the registry, much as you would commit code to a version control system.  
# MAGIC 
# MAGIC The registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`. Each stage has a unique meaning. For example, `Staging` is meant for model testing, while `Production` is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote to Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```

# COMMAND ----------

run_id = best_model.info.run_id
model_uri = f"runs:/{run_id}/model"

client.set_tag(run_id, key='feature_table', value=f'{database_name}.churn_features')
client.set_tag(run_id, key='demographic_vars', value='seniorCitizen, gender_Male')

model_details = mlflow.register_model(model_uri, churn_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC At this point the model will be in `None` stage.  Let's update the description before moving it to `Staging`.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Update Descriptions

# COMMAND ----------

# DBTITLE 1,General Model Description (Do ONCE)
client.update_registered_model(
  name=model_details.name,
  description=f"This model predicts whether a customer will churn using features from the {database_name} database.  It is used to update the Telco Churn Dashboard in DB SQL."
)

# COMMAND ----------

# DBTITLE 1,Specific Version Description
model_version_details = client.get_model_version(name=churn_model_name, version=model_details.version)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description=f"This model version was built using {run_name} for accuracy/F1 validation metric of {round(best_score,2)*100}%"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Request Transition to Staging
# MAGIC 
# MAGIC <!--<img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/webhooks2.png?raw=true" width = 800> -->

# COMMAND ----------

# DBTITLE 1,Transition request triggers testing job 
# Transition request to staging
staging_request = {'name': churn_model_name,
                   'version': model_details.version,
                   'stage': 'Staging',
                   'archive_existing_versions': 'true'}

mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))

# COMMAND ----------

# MAGIC %md
# MAGIC _UPDATE THIS LINK TO POINT TO YOUR JOB_
# MAGIC 
# MAGIC Verify [job](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#job/1006168833879334)

# COMMAND ----------

# DBTITLE 1,Leave comments in model registry (OPTIONAL / Human-In-The-Loop) for ML engineers to review
comment = "Best AutoML model, can be used as challenger/baseline model."
comment_body = {'name': churn_model_name, 'version': model_details.version, 'comment': comment}
mlflow_call_endpoint('comments/create', 'POST', json.dumps(comment_body))

# COMMAND ----------


