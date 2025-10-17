from pyspark.sql import functions as F, types as T

def select_features(clean_df, persist_hdfs=False, out_path=None):
    """
    Select relevant features from clean_df, cast to sensible types,
    report missing values, and optionally persist to HDFS.
    
    Args:
        clean_df (DataFrame): Input cleaned Spark DataFrame
        persist_hdfs (bool): Whether to write the resulting features_df to HDFS
        out_path (str): HDFS path to write features_df if persist_hdfs=True
    
    Returns:
        features_df (DataFrame): Spark DataFrame with selected features
        miss_df (pd.DataFrame): pandas DataFrame summarizing missing values
    """
    
    # Desired features to keep
    desired = [
        'Year', 'Month', 'DayofMonth', 'DayOfWeek',
        'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime',
        'DepDelay', 'Distance', 'Canceled', 'TaxiIn', 'TaxiOut',
        'ArrDelay'
    ]
    
    # Tolerate common column-name variants
    cols = set(clean_df.columns)
    
    # Map 'Canceled' vs 'Cancelled'
    if 'Canceled' not in cols and 'Cancelled' in cols:
        desired = [('Cancelled' if c == 'Canceled' else c) for c in desired]
    
    # DayOfMonth vs DayofMonth
    if 'DayOfMonth' in cols and 'DayofMonth' not in cols:
        desired = [('DayOfMonth' if c == 'DayofMonth' else c) for c in desired]
    elif 'DayofMonth' in cols and 'DayOfMonth' not in cols:
        desired = [('DayofMonth' if c == 'DayOfMonth' else c) for c in desired]
    
    # Keep only existing columns
    existing = [c for c in desired if c in cols]
    missing = [c for c in desired if c not in cols]
    if missing:
        print("⚠️ Warning: these expected columns are missing and will be skipped:", missing)
    
    # Build select expressions with sensible casts
    exprs = []
    for c in existing:
        if c in ('Year', 'Month', 'DayofMonth', 'DayOfWeek', 'Canceled', 'Cancelled'):
            exprs.append(F.col(c).cast(T.IntegerType()).alias(c))
        elif c in ('DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime'):
            exprs.append(F.col(c).cast(T.IntegerType()).alias(c))
        elif c in ('DepDelay', 'ArrDelay', 'TaxiIn', 'TaxiOut', 'Distance'):
            exprs.append(F.col(c).cast(T.DoubleType()).alias(c))
        else:
            exprs.append(F.col(c))
    
    # Create features_df
    features_df = clean_df.select(*exprs).persist()
    
    # Materialize and show schema/sample
    count_rows = features_df.count()
    print("✅ features_df row count:", count_rows)
    print("\nfeatures_df schema:")
    features_df.printSchema()
    print("\nSample rows:")
    features_df.show(5, truncate=False)
    
    # Quick missingness report
    import pandas as pd
    total = features_df.count()
    miss = []
    for c in features_df.columns:
        col_expr = F.col(c)
        try:
            nnull = features_df.filter(col_expr.isNull() | F.isnan(col_expr) | (col_expr == '')).count()
        except Exception:
            nnull = features_df.filter(col_expr.isNull() | (col_expr == '')).count()
        miss.append((c, int(nnull), round(nnull / total * 100, 3)))
    
    miss_df = pd.DataFrame(miss, columns=['column', 'n_missing', 'pct_missing']).sort_values('pct_missing', ascending=False)
    print("\nMissing values report:")
    print(miss_df)
    
    if persist_hdfs and out_path:
        features_df.write.mode('overwrite').partitionBy('Year', 'Month').option('compression', 'snappy').parquet(out_path)
        print("✅ Wrote features_df to", out_path)
    else:
        print("ℹ️ Features not persisted. Set persist_hdfs=True and provide out_path to save to HDFS.")
    
    return features_df, miss_df
