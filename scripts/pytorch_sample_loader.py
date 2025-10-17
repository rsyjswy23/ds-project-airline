"""
Robust sample loader (metastore → parquet fallback)

This script tries the metastore table first. If the sampled result is empty it will:
1. Try reading the parquet fallback and take a larger sample.
2. If that still yields no rows, try a direct limit() read from parquet.
3. If no rows are returned, it prints diagnostics and stops with actionable instructions.

This prevents the StandardScaler from failing when the sample is empty.
"""

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# Initialize or reuse existing Spark session
try:
    spark
except NameError:
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

# Configuration
tbl_name = 'default.flights_2006_transformed'
parquet_path = 'hdfs://namenode:8020/data/parquet/flights_2006_transformed'

def load_small_sample(spark, tbl_name, parquet_path, sample_fraction=0.0005, max_rows=3000, seed=42):
    """Load a small sample from Hive metastore with parquet fallback."""
    df_spark = None

    # Try loading from metastore
    try:
        df_spark = spark.table(tbl_name)
        print('Loaded table from metastore:', tbl_name)
    except Exception as e:
        warnings.warn(f'Metastore load failed: {e}; will try parquet fallback')

    # Attempt sampling from metastore
    if df_spark is not None:
        df_sample = df_spark.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
        cnt = df_sample.count()
        print('Sample rows (from metastore sampling):', cnt)
        if cnt > 0:
            return df_sample
        else:
            warnings.warn('Metastore sampling returned 0 rows; will try parquet fallback with larger fraction/limit')

    # Parquet fallback
    try:
        df_par = spark.read.parquet(parquet_path)
        print('Loaded parquet fallback:', parquet_path)
    except Exception as e:
        raise RuntimeError(f'Failed to read parquet fallback {parquet_path}: {e}')

    # Larger sample from parquet
    larger_fraction = max(sample_fraction * 20, 0.001)
    try:
        df_sample = df_par.sample(withReplacement=False, fraction=larger_fraction, seed=seed)
        cnt = df_sample.count()
        print('Sample rows (from parquet sampling, fraction=', larger_fraction, '):', cnt)
        if cnt > 0:
            return df_sample.limit(max_rows)
    except Exception as e:
        warnings.warn(f'Parquet sampling failed: {e}; will try direct limit()')

    # Final fallback: direct limit()
    try:
        df_direct = df_par.limit(max_rows)
        cnt = df_direct.count()
        print('Sample rows (direct limit from parquet):', cnt)
        if cnt > 0:
            return df_direct
    except Exception as e:
        warnings.warn(f'Direct parquet limit() failed: {e}')

    # Nothing found
    return None


# === Main script ===
if __name__ == "__main__":
    df_sample = load_small_sample(spark, tbl_name, parquet_path,
                                  sample_fraction=0.0005, max_rows=3000, seed=42)

    if df_sample is None:
        print("\nERROR: No rows returned from metastore or parquet fallback.\n")
        print("Actions to investigate:")
        print(" - Verify the Hive metastore table exists: spark.sql('SHOW TABLES IN default').show()")
        print(" - Inspect the parquet path: hadoop fs -ls hdfs://namenode:8020/data/parquet/flights_2006_transformed")
        print(" - Ensure fs.defaultFS is correctly configured for your SparkSession")
        raise RuntimeError("Could not obtain any sample rows from metastore or parquet fallback")

    # Convert to pandas safely (sample should be small)
    df_pd = df_sample.toPandas()
    print('Sample rows (pandas):', len(df_pd))

    if len(df_pd) == 0:
        raise RuntimeError('Pandas sample is empty after all fallbacks — aborting to avoid scaler errors')

    # Ensure target exists and numeric extraction remains robust
    if 'ArrDelay' not in df_pd.columns:
        try:
            df_pd['ArrDelay'] = pd.to_numeric(df_pd.get('ArrDelay', pd.Series(dtype=float)), errors='coerce')
        except Exception:
            df_pd['ArrDelay'] = pd.Series([None] * len(df_pd))

    numeric = df_pd.select_dtypes(include=[np.number]).copy()
    if 'ArrDelay' not in numeric.columns:
        numeric['ArrDelay'] = pd.to_numeric(df_pd['ArrDelay'], errors='coerce')

    # Prepare feature and target arrays
    top_k = min(8, max(1, numeric.shape[1] - 1))
    feat_cols = [c for c in numeric.columns if c != 'ArrDelay'][:top_k]
    X = numeric[feat_cols].copy()
    y = numeric['ArrDelay'].values.astype(np.float32) if 'ArrDelay' in numeric.columns else np.zeros(len(X), dtype=np.float32)

    if len(X) == 0 or X.shape[1] == 0:
        raise RuntimeError('No numeric features available after fallbacks. Check the transformed dataset contents.')

    if len(X) > 3000:
        sampled_idx = np.random.RandomState(42).choice(len(X), size=3000, replace=False)
        X = X.iloc[sampled_idx].reset_index(drop=True)
        y = y[sampled_idx]

    print('Features shape:', X.shape)

    # Split, scale, and prepare numpy arrays
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print('Prepared numpy arrays:')
    print('  X_train:', X_train_scaled.shape)
    print('  y_train:', y_train.shape)