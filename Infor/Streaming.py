# Databricks notebook source
# MAGIC %md
# MAGIC # Topics to Cover 
# MAGIC 
# MAGIC - **Basic Streaming Concepts**
# MAGIC   - Read Stream
# MAGIC   - Write Stream (Use CDC file)
# MAGIC   - Merge Example for CDC
# MAGIC   - Checkpointing
# MAGIC   
# MAGIC - **Autoloader**
# MAGIC   - Cloudfiles, Schema inference and evolution
# MAGIC   
# MAGIC - **Loading from CDC data**
# MAGIC   - Streaming example
# MAGIC   
# MAGIC - **DLT!**
# MAGIC   - How DLT makes everything evern more simple!
# MAGIC   
# MAGIC - **Other Advanced concepts (to be covered later)**
# MAGIC   - Streaming Aggregations
# MAGIC   - Windowing
# MAGIC   - Watermarking

# COMMAND ----------

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


spark.sql(f"DROP DATABASE IF EXISTS {database} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
spark.sql(f"USE {database}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Micro-Batches as a Table
# MAGIC 
# MAGIC For more information, see the analogous section in the [Structured Streaming Programming Guide](http://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#basic-concepts) (from which several images have been borrowed).
# MAGIC 
# MAGIC Spark Structured Streaming approaches streaming data by modeling it as a series of continuous appends to an unbounded table. While similar to defining **micro-batch** logic, this model allows incremental queries to be defined against streaming sources as if they were static input.
# MAGIC 
# MAGIC <img src="http://spark.apache.org/docs/latest/img/structured-streaming-stream-as-a-table.png" style="height: 300px"/>
# MAGIC 
# MAGIC ### Basic Concepts
# MAGIC 
# MAGIC - The developer defines an **input table** by configuring a streaming read against a **source**. The syntax provides entry that is nearly analogous to working with static data.
# MAGIC - A **query** is defined against the input table. Both the DataFrames API and Spark SQL can be used to easily define transformations and actions against the input table.
# MAGIC - This logical query on the input table generates the **results table**. The results table contains the incremental state information of the stream.
# MAGIC - The **output** of a streaming pipeline will persist updates to the results table by writing to an external **sink**. Generally, a sink will be a durable system such as files or a pub/sub messaging bus.
# MAGIC - New rows are appended to the input table for each **trigger interval**. These new rows are essentially analogous to micro-batch transactions and will be automatically propagated through the results table to the sink.
# MAGIC 
# MAGIC <img src="http://spark.apache.org/docs/latest/img/structured-streaming-model.png" style="height: 300px"/>
# MAGIC 
# MAGIC This lesson will demonstrate the ease of adapting batch logic to streaming data to run data workloads in near real-time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Used
# MAGIC The source contains smartphone accelerometer samples from devices and users with the following columns:
# MAGIC 
# MAGIC | Field          | Description |
# MAGIC | ------------- | ----------- |
# MAGIC | Arrival_Time | time data was received |
# MAGIC | Creation_Time | event time |
# MAGIC | Device | type of Model |
# MAGIC | Index | unique identifier of event |
# MAGIC | Model | i.e Nexus4  |
# MAGIC | User | unique user identifier |
# MAGIC | geolocation | city & country |
# MAGIC | gt | transportation mode |
# MAGIC | id | unused null field |
# MAGIC | x | acceleration in x-dir |
# MAGIC | y | acceleration in y-dir |
# MAGIC | z | acceleration in z-dir |

# COMMAND ----------

dataPath = "/mnt/training/definitive-guide/data/activity-json/streaming"

display(dbutils.fs.ls(dataPath))

# COMMAND ----------

dbutils.fs.head(f"{dataPath}/01.json")

# COMMAND ----------

# MAGIC %md 
# MAGIC   {"User":"g"  
# MAGIC   ,"Arrival_Time":1424687290197  
# MAGIC   ,"Creation_Time":1424687288205373378  
# MAGIC   ,"Device":"nexus4_1"  
# MAGIC   ,"Index":88614  
# MAGIC   ,"Model":"nexus4"  
# MAGIC   ,"gt":"sit"  
# MAGIC   ,"x":-6.256104E-4  
# MAGIC   ,"y":0.0066833496  
# MAGIC   ,"z":-0.0018920898  
# MAGIC   ,"geolocation":{"city":"Nanjing","country":"China"}  
# MAGIC   }

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Traditional Reading and writing with Spark 

# COMMAND ----------

schema = "Arrival_Time BIGINT, Creation_Time BIGINT, Device STRING, Index BIGINT, Model STRING, User STRING, geolocation STRUCT<city: STRING, country: STRING>, gt STRING, id BIGINT, x DOUBLE, y DOUBLE, z DOUBLE"

dataPath = "/mnt/training/definitive-guide/data/activity-json/streaming"

staticDF = (spark
  .read
  .format("json")
  .schema(schema)
  .load(dataPath)
)

display(staticDF)

# COMMAND ----------

outputPath = userhome + "/static-write"

dbutils.fs.rm(outputPath, True)    # clear this directory in case lesson has been run previously

(staticDF                                
  .write                                               
  .format("delta")                                          
  .mode("append")                                       
  .save(outputPath))

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select count(1) from delta.`dbfs:/user/rohit.bhagwat@databricks.com/streaming_poc/static-write`

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Spark Structured Streaming

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading a Stream

# COMMAND ----------

streamingDF = (spark
  .readStream # readStream instead of read
  .format("json")
  .schema(schema)
  .option("maxFilesPerTrigger", 1)     # Optional; force processing of only 1 file per trigger, other options available 
  .load(dataPath)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Writing a Stream
# MAGIC The method DataFrame.writeStream returns a DataStreamWriter used to configure the output of the stream.
# MAGIC 
# MAGIC There are a number of required parameters to configure a streaming write:
# MAGIC 
# MAGIC The format of the output sink (see documentation)
# MAGIC The location of the checkpoint directory
# MAGIC The output mode
# MAGIC Configurations specific to the output sink, such as:
# MAGIC Kafka
# MAGIC Event Hubs
# MAGIC A <a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=foreach#pyspark.sql.streaming.DataStreamWriter.foreach"target="_blank">custom sink via writeStream.foreach(...)
# MAGIC Once the configuration is completed, trigger the job with a call to .start(). When writing to files, use .start(filePath).

# COMMAND ----------

outputPath = userhome + "/streaming-concepts"
checkpointPath = outputPath + "/checkpoint"

dbutils.fs.rm(outputPath, True)    # clear this directory in case this has been run previously

streamingQuery = (streamingDF                                
  .writeStream                                                
  .format("delta")                                          
  .option("checkpointLocation", checkpointPath)
#   .trigger(once=True)
  .outputMode("append")
#   .queryName("my_stream")        # optional argument to register stream to Spark catalog
  .start(outputPath)                                       
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select count(1) from delta.`dbfs:/user/rohit.bhagwat@databricks.com/streaming_poc/streaming-concepts`

# COMMAND ----------

display(dbutils.fs.ls(checkpointPath))

# COMMAND ----------

detaStreamingSource = (spark
  .readStream # readStream instead of read
  .format("delta")
  .option("maxFilesPerTrigger", 1)     # Optional; force processing of only 1 file per trigger, other options available 
  .load(outputPath)
)

# COMMAND ----------

display(detaStreamingSource)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checkpointing
# MAGIC Databricks creates checkpoints by storing the current state of your streaming job to Azure Blob Storage or ADLS.
# MAGIC Checkpointing combines with write ahead logs to allow a terminated stream to be restarted and continue from where it left off.
# MAGIC Checkpoints cannot be shared between separate streams.

# COMMAND ----------

# Display dbfs contents
display(dbutils.fs.ls(checkpointPath))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC #### Output Modes
# MAGIC 
# MAGIC Streaming jobs have output modes similar to static/batch workloads. [More details here](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#output-modes).
# MAGIC 
# MAGIC | Mode   | Example | Notes |
# MAGIC | ------------- | ----------- |
# MAGIC | **Append** | `.outputMode("append")`     | _DEFAULT_ - Only the new rows appended to the Result Table since the last trigger are written to the sink. |
# MAGIC | **Complete** | `.outputMode("complete")` | The entire updated Result Table is written to the sink. The individual sink implementation decides how to handle writing the entire table. |
# MAGIC | **Update** | `.outputMode("update")`     | Only the rows in the Result Table that were updated since the last trigger will be outputted to the sink. Since Spark 2.1.1 |
# MAGIC 
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Not all sinks will support `update` mode.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### The Display function
# MAGIC 
# MAGIC Within the Databricks notebooks, we can use the `display()` function to render a live plot. This stream is written to memory; **generally speaking this is most useful for debugging purposes**.
# MAGIC 
# MAGIC When you pass a "streaming" `DataFrame` to `display()`:
# MAGIC * A "memory" sink is being used
# MAGIC * The output mode is complete
# MAGIC * *OPTIONAL* - The query name is specified with the `streamName` parameter
# MAGIC * *OPTIONAL* - The trigger is specified with the `trigger` parameter
# MAGIC * *OPTIONAL* - The checkpointing location is specified with the `checkpointLocation`
# MAGIC 
# MAGIC `display(myDF, streamName = "myQuery")`

# COMMAND ----------

display(streamingDF, streamName = "streaming_display")
# display(streamingDF)

# COMMAND ----------

for stream in spark.streams.active:
  if stream.name == "streaming_display":
    print(f"Found {stream.name} {stream.id}")

# COMMAND ----------

spark.catalog.listTables()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM streaming_display WHERE gt = "stand"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autoloader

# COMMAND ----------

schemaLocation = outputPath + "/schema"
autoloaderStreamingDF = (spark
                         .readStream
                         .format("cloudFiles")
                         .option("cloudFiles.format", "json")
                         .option("cloudFiles.schemaLocation", schemaLocation)
                         .load(dataPath))

# load("/mnt/field-demos/retail/users_json", "json")

# COMMAND ----------

display(autoloaderStreamingDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Autoloader Benefits
# MAGIC In Apache Spark, you can read files incrementally using spark.readStream.format(fileFormat).load(directory). Auto Loader provides the following benefits over the file source:
# MAGIC 
# MAGIC **Scalability**: Auto Loader can discover billions of files efficiently. Backfills can be performed asynchronously to avoid wasting any compute resources.
# MAGIC 
# MAGIC **Performance**: The cost of discovering files with Auto Loader scales with the number of files that are being ingested instead of the number of directories that the files may land in. See Optimized directory listing.
# MAGIC 
# MAGIC **Schema inference and evolution support**: Auto Loader can detect schema drifts, notify you when schema changes happen, and rescue data that would have been otherwise ignored or lost. See Schema inference and evolution.
# MAGIC 
# MAGIC **Cost**: Auto Loader uses native cloud APIs to get lists of files that exist in storage. In addition, Auto Loader’s file notification mode can help reduce your cloud costs further by avoiding directory listing altogether. Auto Loader can automatically set up file notification services on storage to make file discovery much cheaper.

# COMMAND ----------

# MAGIC %md 
# MAGIC # CDC input from tools like Debezium
# MAGIC For each change, we receive a JSON message containing all the fields of the row being updated (customer name, email, address...). In addition, we have extra metadata informations including:
# MAGIC 
# MAGIC operation: an operation code, typically (DELETE, APPEND, UPDATE)
# MAGIC operation_date: the date and timestamp for the record came for each operation action
# MAGIC Tools like Debezium can produce more advanced output such as the row value before the change, but we'll exclude them for the clarity of the demo

# COMMAND ----------

# MAGIC %python
# MAGIC display(spark.read.json("/tmp/demo/cdc_raw/customers"))
# MAGIC 
# MAGIC cdcSchema = spark.read.json("/tmp/demo/cdc_raw/customers").schema

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Steps to Ingest 
# MAGIC - Load incoming JSON files into Bronze CDC table 
# MAGIC - Merge incoming data into Clean Silver layer table

# COMMAND ----------

cdcStream = (spark
             .readStream
             .format("json")
             .schema(cdcSchema)
             .option("maxFilesPerTrigger", 1)     # Optional; force processing of only 1 file per trigger, other options available 
             .load("/tmp/demo/cdc_raw/customers")
            )

# COMMAND ----------

outputPath = userhome + "/cdc-stream"
checkpointPath = outputPath + "/checkpoint"

dbutils.fs.rm(outputPath, True)    # clear this directory in case this has been run previously
dbutils.fs.rm('dbfs:/user/rohit.bhagwat@databricks.com/streaming_poc/customers', True)

# COMMAND ----------

streamingQuery = (cdcStream
                  .writeStream
                  .format("delta")
                  .option("checkpointLocation", checkpointPath)
                  .outputMode("append")
                  #   .queryName("my_stream")        # optional argument to register stream to Spark catalog
                  .start(outputPath)
                 )

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from delta.`dbfs:/user/rohit.bhagwat@databricks.com/streaming_poc/cdc-stream`

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(1) from delta.`dbfs:/user/rohit.bhagwat@databricks.com/streaming_poc/cdc-stream`

# COMMAND ----------

# MAGIC %md
# MAGIC **Merge into Silver layer table**

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC drop  
# MAGIC create table streaming_poc_rohit_bhagwat_databricks_com_db.customers(
# MAGIC   `address` STRING,
# MAGIC   `email` STRING,
# MAGIC   `firstname` STRING,
# MAGIC   `id` STRING,
# MAGIC   `lastname` STRING,
# MAGIC   `operation` STRING,
# MAGIC   `operation_date` STRING)
# MAGIC USING delta
# MAGIC LOCATION 'dbfs:/user/rohit.bhagwat@databricks.com/streaming_poc/customers'

# COMMAND ----------

# MAGIC %sql 
# MAGIC merge into delta.`dbfs:/user/rohit.bhagwat@databricks.com/streaming_poc/customers` c
# MAGIC using
# MAGIC  (select address, email, firstname, id, lastname, operation, operation_date
# MAGIC  from delta.`dbfs:/user/rohit.bhagwat@databricks.com/streaming_poc/cdc-stream`
# MAGIC  order by operation_date desc
# MAGIC  ) i
# MAGIC on c.id = i.id
# MAGIC when matched and i.operation = 'DELETE' and i.operation_date > c.operation_date then delete
# MAGIC when matched and i.operation = 'UPDATE' then update set *
# MAGIC when not matched and i.operation = 'APPEND' then insert *

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Other key topics 
# MAGIC 
# MAGIC ## Deploying Streaming Applications
# MAGIC 
# MAGIC - Always On
# MAGIC - Trigger Once
# MAGIC 
# MAGIC 
# MAGIC # Jobs
# MAGIC - Notebook Job
# MAGIC - Spark Submit Job 

# COMMAND ----------



# COMMAND ----------


