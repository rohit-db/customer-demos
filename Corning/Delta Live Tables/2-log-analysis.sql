-- Databricks notebook source
-- MAGIC %md
-- MAGIC # DLT pipeline log analysis
-- MAGIC Please Make sure you specify your own Database and Storage location. You'll find this information in the configuration menu of your Delta Live Table Pipeline.
-- MAGIC 
-- MAGIC **NOTE:** Please use Databricks Runtime 9.1 or above when running this notebook

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.widgets.text('storage_location', '/Users/rohit_bhagwat_databricks_com/demo', 'Storage Location')
-- MAGIC dbutils.widgets.text('db', 'rohit_bhagwat_dlt_demo', 'DLT Database')

-- COMMAND ----------

-- MAGIC %python display(dbutils.fs.ls(dbutils.widgets.get('storage_location')))

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS ${db};
USE ${db};

-- COMMAND ----------

CREATE OR REPLACE VIEW pipeline_logs AS SELECT * FROM delta.`${storage_location}/system/events`

-- COMMAND ----------

SELECT * FROM pipeline_logs ORDER BY timestamp

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The `details` column contains metadata about each Event sent to the Event Log. There are different fields depending on what type of Event it is. Some examples include:
-- MAGIC * `user_action` Events occur when taking actions like creating the pipeline
-- MAGIC * `flow_definition` Events occur when a pipeline is deployed or updated and have lineage, schema, and execution plan information
-- MAGIC   * `output_dataset` and `input_datasets` - output table/view and its upstream table(s)/view(s)
-- MAGIC   * `flow_type` - whether this is a complete or append flow
-- MAGIC   * `explain_text` - the Spark explain plan
-- MAGIC * `flow_progress` Events occur when a data flow starts running or finishes processing a batch of data
-- MAGIC   * `metrics` - currently contains `num_output_rows`
-- MAGIC   * `data_quality` - contains an array of the results of the data quality rules for this particular dataset
-- MAGIC     * `dropped_records`
-- MAGIC     * `expectations`
-- MAGIC       * `name`, `dataset`, `passed_records`, `failed_records`
-- MAGIC   

-- COMMAND ----------

SELECT
  details:flow_definition.output_dataset,
  details:flow_definition.input_datasets,
  details:flow_definition.flow_type,
  details:flow_definition.schema,
  details:flow_definition
FROM pipeline_logs
WHERE details:flow_definition IS NOT NULL
ORDER BY timestamp

-- COMMAND ----------

SELECT
  id,
  date(timestamp),
  expectations.dataset,
  expectations.name,
  expectations.failed_records,
  expectations.passed_records
FROM(
  SELECT 
    id,
    timestamp,
    details:flow_progress.metrics,
    details:flow_progress.data_quality.dropped_records,
    explode(from_json(details:flow_progress:data_quality:expectations
             ,schema_of_json("[{'name':'str', 'dataset':'str', 'passed_records':42, 'failed_records':42}]"))) expectations
  FROM pipeline_logs
  WHERE details:flow_progress.metrics IS NOT NULL) data_quality

-- COMMAND ----------


