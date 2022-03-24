-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Introducing Delta Live Tables
-- MAGIC A simple way to build and manage data pipelines for fresh, high quality data! 
-- MAGIC 
-- MAGIC 
-- MAGIC DLT makes Data Engineering accessible for all. Just declare your transformations in SQL or Pythin, and DLT will handle the Data Engineering complexity for you.
-- MAGIC 
-- MAGIC **Accelerate ETL development** <br/>
-- MAGIC Enable analysts and data engineers to innovate rapidly with simple pipeline development and maintenance 
-- MAGIC 
-- MAGIC **Remove operational complexity** <br/>
-- MAGIC By automating complex administrative tasks and gaining broader visibility into pipeline operations
-- MAGIC 
-- MAGIC **Trust your data** <br/>
-- MAGIC With built-in quality controls and quality monitoring to ensure accurate and useful BI, Data Science, and ML 
-- MAGIC 
-- MAGIC **Simplify batch and streaming** <br/>
-- MAGIC With self-optimization and auto-scaling data pipelines for batch or streaming processing 
-- MAGIC 
-- MAGIC ## Our Delta Live Table pipeline
-- MAGIC Our dataset consists of vibration readings coming off sensors located in the gearboxes of wind turbines. 
-- MAGIC We will use Delta Live Tables to build a ETL pipeline which ingests this data in near real time and builds tables ready for consumption by our Data Analyst teams.
-- MAGIC 
-- MAGIC <img width="900" src="https://i.ibb.co/DWdnZYM/image.png">

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Ingest Raw Data into Bronze Layer

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ### Incremental Live Table

-- COMMAND ----------

create incremental live table turbine_bronze_dlt

tblproperties ("quality" = "bronze")

comment "Raw user data coming from json files ingested in incremental with Auto Loader to support schema inference and evolution"

as select * from cloud_files("/mnt/rb-demo-resources/turbine/incoming-data-json", "json")

-- COMMAND ----------

create incremental live table turbine_status_dlt (id int, status string)

tblproperties ("quality" = "bronze")

as select id, status 
from cloud_files("/mnt/rb-demo-resources/turbine/status", "parquet", map("schema", "id int, status string"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Clean and Load Data into Silver Layer
-- MAGIC Join tables, ensure data quality
-- MAGIC Once the bronze layer is defined, we'll create the sliver layers by Joining data. Note that bronze tables are referenced using the `LIVE` spacename. 
-- MAGIC To consume only increment from the Bronze layer like `turbine_bronze_dlt`, we'll be using the `stream` keyworkd: `stream(LIVE.turbine_bronze_dlt)`
-- MAGIC Note that we don't have to worry about compactions, DLT handles that for us.
-- MAGIC 
-- MAGIC #### Expectations
-- MAGIC By defining expectations (`CONSTRANT <name> EXPECT <condition>`), you can enforce and track your data quality. See the [documentation](https://docs.databricks.com/data-engineering/delta-live-tables/delta-live-tables-expectations.html) for more details

-- COMMAND ----------

create incremental live table turbine_silver_dlt(
  constraint valid_id expect (id is not null) on violation drop row,
  constraint valid_status expect (s.status is not null) on violation drop row,
  constraint valid_an10 expect (an10 < 10),
  constraint valid_speed expect (speed > 2)
  -- Roadmap: Quarantine 
  
)
tblproperties ("quality" = "silver")

comment "Cleanup incoming turbine data, join to status table and drop unnecessary columns"

as 
select cast(t.id as int) as id
  , timestamp 
  , s.status  
  , cast(an10 as double)
  , cast(an3 as double)
  , cast(an4 as double)
  , cast(an5 as double)
  , cast(an6 as double)
  , cast(an7 as double)
  , cast(an8 as double)
  , cast(an9 as double)
  , cast(speed as double)
from stream(live.turbine_bronze_dlt) t
join (select id, max(status) as status from live.turbine_status_dlt group by 1) s using (id)

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Gold Layer - Aggregates

-- COMMAND ----------

create live table turbine_gold_daily_dlt(
  constraint valid_id expect (id is not null) on violation drop row
)
  tblproperties (
    "quality" = "gold",
    "pipelines.autoOptimize.zOrderCols" = "id"
  )
comment "Daily Aggregate for all sensors"
as 
select id
  , date(timestamp) as record_date
  , extract(year from timestamp) as record_year
  , avg(an10) as an10
  , avg(an3) as an3
  , avg(an4) as an4 
  , avg(an5) as an5
  , avg(an6) as an6
  , avg(an7) as an7
  , avg(an8) as an8
  , avg(an9) as an9
  , avg(speed) as speed
from live.turbine_silver_dlt sd
group by id, record_date, record_year

-- COMMAND ----------

create live table turbine_gold_sensor_dlt(
  constraint valid_id expect (id is not null) on violation drop row 
)
tblproperties (
    "quality" = "gold",
    "pipelines.autoOptimize.zOrderCols" = "id"
  )
comment "Monthly Aggregate for all sensors"
as 
select id
  , date_format(timestamp, 'y-MM') as record_month
  , extract(year from timestamp) as record_year
  , avg(an10) as an10
  , avg(an3) as an3
  , avg(an4) as an4 
  , avg(an5) as an5
  , avg(an6) as an6
  , avg(an7) as an7
  , avg(an8) as an8
  , avg(an9) as an9
  , avg(speed) as speed
from live.turbine_silver_dlt sd
group by id, record_month, record_year

-- COMMAND ----------

-- MAGIC %md ## Next steps
-- MAGIC 
-- MAGIC Your DLT pipeline is ready to be started.
-- MAGIC Open the DLT menu, create a pipeline and select this notebook to run it. 
-- MAGIC 
-- MAGIC Datas Analyst can start using DBSQL to analyze data and track our Loan metrics.  Data Scientist can also access the data to start building models to predict payment default or other more advanced use-cases.

-- COMMAND ----------

-- MAGIC %md ## Tracking data quality
-- MAGIC 
-- MAGIC Expectations stats are automatically available as system table.
-- MAGIC 
-- MAGIC This information let you monitor your data ingestion quality. 
-- MAGIC 
-- MAGIC You can leverage DBSQL to request these table and build custom alerts based on the metrics your business is tracking.
-- MAGIC 
-- MAGIC 
-- MAGIC See [how to access your DLT metrics]($./02-Log-Analysis)
-- MAGIC 
-- MAGIC <img width="500" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/retail-dlt-data-quality-dashboard.png">
-- MAGIC 
-- MAGIC <a href="https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/6f73dd1b-17b1-49d0-9a11-b3772a2c3357-dlt---retail-data-quality-stats?o=1444828305810485" target="_blank">Data Quality Dashboard example</a>
