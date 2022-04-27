# Databricks notebook source
# MAGIC %md
# MAGIC ## Churn Prediction Feature Engineering
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/1_data_prep.png?raw=true">

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization Logic
# MAGIC 
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

# DBTITLE 1,Read in Bronze Delta table using Spark
# Read into Spark
telcoDF = spark.table(f"{database_name}.bronze_customers")
display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Using `koalas` allows us to scale `pandas` code.
# MAGIC 
# MAGIC _A function that is decorated with the @feature_table decorator will gain these function attributes:
# MAGIC databricks.feature_store.decorators.compute_and_write(input: Dict[str, Any], feature_table_name: str, mode: str = 'merge') → pyspark.sql.dataframe.DataFrame_

# COMMAND ----------

# DBTITLE 1,Define featurization function
from databricks.feature_store import feature_table
import databricks.koalas as ks
import pyspark.pandas as ps

@feature_table
def compute_churn_features(data):
  
  # Convert to koalas (CONSIDER UPDATING THIS PIECE TO USE NATIVE PANDAS API STARTING DBR10.0+)
  data = data.to_koalas()
  
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

churn_features_df = compute_churn_features(telcoDF)
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

# MAGIC %sql 
# MAGIC select * from rohit_bhagwat_churn_demo.churn_features

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Delta Lake for interactive autoML purposes (OPTIONAL)
# MAGIC Drop ID/keys and non-necessary columns and do UI/demo - otherwise you can skip and run any MLflow experiment on this table

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

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from rohit_bhagwat_churn_demo.silver_customers_automl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Churn Prediction Feature Engineering
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/2_baseline_automl.png?raw=true">

# COMMAND ----------


