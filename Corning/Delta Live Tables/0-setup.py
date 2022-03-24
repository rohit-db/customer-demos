# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Environment
# MAGIC _Note the files are uploaded in a S3 bucket which has public access._ 

# COMMAND ----------

aws_bucket_name = "quentin-demo-resources"
mount_name = "rb-demo-resources"


try:
  dbutils.fs.ls("/mnt/%s" % mount_name)
  
except:
  print("bucket isn't mounted, mounting the demo bucket under %s" % mount_name)
  dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)

# COMMAND ----------

import re 
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = re.sub(r'\W+', '_', current_user)
path = f"/Users/{dbName}/demo"
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("dbName", dbName, "dbName")
print(f"path (default path): {path}")
spark.sql(f"create database if not exists {dbName} LOCATION '{path}/global_demo/tables'")
spark.sql(f"USE {dbName}")
print("dbName (using database): {dbName}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's look at incoming data

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Turbine Sensor Data

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/mnt/rb-demo-resources/turbine/incoming-data-json'))

# COMMAND ----------

# MAGIC %fs head /mnt/rb-demo-resources/turbine/incoming-data-json/part-00000-tid-5699454748636473773-d65702b7-724d-4b96-825b-84a0568d85bf-2417-1-c000.json

# COMMAND ----------

data = json.loads('{"ID":162.0,"AN3":-0.25877,"AN4":2.3713,"AN5":-1.1298,"AN6":1.5778,"AN7":-3.7185,"AN8":5.5253,"AN9":2.8008,"AN10":-1.5991,"SPEED":6.3287E-4,"TORQUE":"","TIMESTAMP":"2020-05-27T08:27:20.000Z"}')

import json 
print(json.dumps(data, indent = 4))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Turbine Status

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/mnt/rb-demo-resources/turbine/status'))

# COMMAND ----------

display(spark.read.parquet('/mnt/rb-demo-resources/turbine/status/part-00000-tid-2294590807750575412-617a7d3b-1131-4a3e-b6eb-8456b6473a52-1106687-1-c000.snappy.parquet'))

# COMMAND ----------


