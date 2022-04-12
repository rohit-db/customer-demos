# Databricks notebook source

# Setup
import pyspark.sql.functions as F
import re

username = spark.sql("SELECT current_user()").collect()[0][0]
course = "streaming_poc"
userhome = f"dbfs:/user/{username}/{course}"
database = f"""{course}_{re.sub("[^a-zA-Z0-9]", "_", username)}_db"""

print(f"""
username: {username}
userhome: {userhome}
database: {database}""")

schema = "Arrival_Time BIGINT, Creation_Time BIGINT, Device STRING, Index BIGINT, Model STRING, User STRING, geolocation STRUCT<city: STRING, country: STRING>, gt STRING, id BIGINT, x DOUBLE, y DOUBLE, z DOUBLE"

dataPath = "/mnt/training/definitive-guide/data/activity-json/streaming"

streamingDF = (spark
  .readStream # readStream instead of read
  .format("json")
  .schema(schema)
  .option("maxFilesPerTrigger", 1)     # Optional; force processing of only 1 file per trigger, other options available 
  .load(dataPath)
)

# COMMAND ----------

outputPath = userhome + "/streaming-concepts-job"
checkpointPath = outputPath + "/checkpoint"

dbutils.fs.rm(outputPath, True)    # clear this directory in case this has been run previously

streamingQuery = (streamingDF                                
  .writeStream                                                
  .format("delta")                                          
  .option("checkpointLocation", checkpointPath)               
  .outputMode("append")
#   .queryName("my_stream")        # optional argument to register stream to Spark catalog
  .start(outputPath)                                       
)

# COMMAND ----------


