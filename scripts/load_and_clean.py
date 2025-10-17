#!/usr/bin/env python3
"""
load_and_clean.py

Robust loader for the flights dataset.

Behavior:
 - Create SparkSession with Hive support.
 - Try to load one of several Hive tables (in order).
 - If no Hive table, try a list of Parquet HDFS paths.
 - If no Parquet, try a CSV fallback on HDFS.
 - Cast ArrDelay -> Double and drop null/NaN; produce `clean_df`.
 - Print row counts and a small sample.

Usage:
  spark-submit scripts/load_and_clean.py
or (if pyspark is on PYTHONPATH)
  python scripts/load_and_clean.py

You can override defaults via CLI args:
  --master, --warehouse-dir, --metastore-uri
"""
import argparse
import sys
import traceback

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType


def build_spark(master, warehouse_dir, metastore_uri, app_name="load_and_clean"):
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    # enable Hive support so spark.table(...) can use the metastore if available
    builder = builder.enableHiveSupport()
    if warehouse_dir:
        builder = builder.config("spark.sql.warehouse.dir", warehouse_dir)
    if metastore_uri:
        builder = builder.config("spark.hadoop.hive.metastore.uris", metastore_uri)
    spark = builder.getOrCreate()
    return spark


def hdfs_path_exists(spark, hdfs_uri):
    """
    Check HDFS path existence using the Spark JVM Hadoop FileSystem.
    Returns True if exists, False on any error or not found.
    """
    try:
        conf = spark._jsc.hadoopConfiguration()
        uri = spark._jvm.java.net.URI.create(hdfs_uri)
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(uri, conf)
        Path = spark._jvm.org.apache.hadoop.fs.Path
        return fs.exists(Path(hdfs_uri))
    except Exception:
        # silent false on any error to keep fallback logic simple
        return False


def try_load(spark, candidate_tables, candidate_paths, csv_fallback):
    df = None
    # 1) Try Hive tables
    for t in candidate_tables:
        try:
            tbls = spark.sql(f"SHOW TABLES IN default LIKE '{t}'")
            if tbls.count() > 0:
                print(f"Loading from Hive table: default.{t}")
                df = spark.table(f"default.{t}")
                return df
        except Exception:
            # ignore and continue to next candidate
            pass

    # 2) Try Parquet paths
    for p in candidate_paths:
        try:
            exists = hdfs_path_exists(spark, p)
            if exists:
                print(f"Loading from Parquet path: {p}")
                df = spark.read.option("mergeSchema", "true").parquet(p)
                return df
            else:
                print(f"Parquet path not found: {p}")
        except Exception as e:
            print(f"Error reading parquet at {p}: {e}")
            traceback.print_exc()

    # 3) Try CSV fallback
    try:
        if csv_fallback and hdfs_path_exists(spark, csv_fallback):
            print(f"Loading raw CSV fallback: {csv_fallback}")
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_fallback)
            return df
        else:
            print(f"CSV fallback path not found: {csv_fallback}")
    except Exception as e:
        print("CSV fallback failed:", e)
        traceback.print_exc()

    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load flights data and create clean_df")
    parser.add_argument("--master", default="spark://spark-master:7077", help="Spark master URL")
    parser.add_argument("--warehouse-dir", default="hdfs://namenode:8020/user/hive/warehouse",
                        help="Spark SQL warehouse dir (HDFS)")
    parser.add_argument("--metastore-uri", default="thrift://hive-metastore:9083",
                        help="Hive metastore thrift URI")
    parser.add_argument("--show-sample", type=int, default=5, help="Number of sample rows to show")
    args = parser.parse_args(argv)

    spark = build_spark(args.master, args.warehouse_dir, args.metastore_uri, app_name="load_and_clean")

    candidates_tables = [
        "flights_2006_cleaned",
        "flights_2006_staged",
        "flights_2006",
    ]
    candidates_paths = [
        "hdfs://namenode:8020/data/parquet/flights_2006_cleaned",
        "hdfs://namenode:8020/data/parquet/flights_2006",
        "hdfs://namenode:8020/data/parquet/flights_2006_features",
    ]
    csv_fallback = "hdfs://namenode:8020/data/flights/2006.csv"

    try:
        df = try_load(spark, candidates_tables, candidates_paths, csv_fallback)
        if df is None:
            # Diagnostic listing of /data/parquet to help debugging
            try:
                conf = spark._jsc.hadoopConfiguration()
                fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
                    spark._jvm.java.net.URI.create("hdfs://namenode:8020"), conf
                )
                Path = spark._jvm.org.apache.hadoop.fs.Path
                base = "/data/parquet"
                print(f"\nDiagnostic: listing {base}")
                status = fs.listStatus(Path(base))
                for s in status:
                    print("-", s.getPath().toString())
            except Exception as e:
                print("Failed to list HDFS /data/parquet:", e)
            raise RuntimeError("No suitable data source found (Hive table, Parquet path, or CSV).")

        # Create clean_df: cast ArrDelay to double and drop null/NaN
        df_cast = df.withColumn("ArrDelay", F.col("ArrDelay").cast(DoubleType()))
        count_before = df_cast.count()
        clean_df = df_cast.filter(F.col("ArrDelay").isNotNull() & (~F.isnan(F.col("ArrDelay"))))
        count_after = clean_df.count()

        print(f"rows before: {count_before:,} | rows after (clean): {count_after:,} | removed: {count_before - count_after:,}")
        print("\nSchema of clean_df:")
        clean_df.printSchema()
        print(f"\nShowing up to {args.show_sample} sample rows:")
        # convert to pandas for nicer printing if small
        try:
            pdf = clean_df.limit(args.show_sample).toPandas()
            print(pdf)
        except Exception:
            clean_df.show(args.show_sample, truncate=False)

    except Exception:
        print("Error during loading/cleaning:")
        traceback.print_exc()
        sys.exit(2)
    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()