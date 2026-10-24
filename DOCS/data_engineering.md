Data Engineering Concepts

## ETL vs ELT

ETL (Extract, Transform, Load) transforms data before loading into the target. ELT (Extract, Load, Transform) loads raw data first, then transforms in place. ELT is preferred for modern cloud data warehouses because they have powerful compute for transformations.

## Data Warehouses

A data warehouse is a centralized repository for structured, filtered data. It serves analytical queries and business intelligence. Modern examples include Snowflake, BigQuery, and Redshift.

## Medallion Architecture

The medallion architecture organizes data into Bronze (raw), Silver (cleaned), and Gold (business-ready) layers. Bronze holds raw ingested data. Silver applies cleaning and conforming. Gold contains aggregated, business-level tables ready for analytics.

## Orchestration

Data orchestration tools manage the execution order and dependencies of data pipelines. Apache Airflow is the most popular open-source orchestrator. It uses DAGs (Directed Acyclic Graphs) to define workflow dependencies.