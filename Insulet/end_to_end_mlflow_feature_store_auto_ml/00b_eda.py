# Databricks notebook source
# MAGIC %md
# MAGIC ## EDA
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/rohit-db/customer-demos/master/images/1_data_prep.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC # Use spark dataframes profiling
# MAGIC From [github repo](https://github.com/julioasotodv/spark-df-profiling)

# COMMAND ----------

# MAGIC %pip install git+https://github.com/julioasotodv/spark-df-profiling

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read bronze delta table as spark dataframe

# COMMAND ----------

bronzeDF = spark.table(f"{database_name}.bronze_customers").drop(key_id).cache()
display(bronzeDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Profile

# COMMAND ----------

import spark_df_profiling
spark_df_profiling.ProfileReport(bronzeDF)

# COMMAND ----------

# MAGIC %md
# MAGIC # (OR) Use Pandas Profiling if data fits in memory of driver

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(bronzeDF.toPandas(), title="IBM Telco-Churn Raw Data Exploration", progress_bar=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------


