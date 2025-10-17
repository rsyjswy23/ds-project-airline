import os
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyspark.sql import SparkSession
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
local_ckpt = 'models/pytorch_small_regressor.pth'
tbl_name = 'default.flights_2006_transformed'
parquet_path = 'hdfs://namenode:8020/data/parquet/flights_2006_transformed'


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def load_checkpoint(path):
    """Load a PyTorch checkpoint file."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print('Loading checkpoint from', path)
    return torch.load(path, map_location='cpu')


def get_small_sample(spark, tbl_name, parquet_path, sample_fraction=0.0005, max_rows=3000, seed=42):
    """Try to get a small sample from Hive metastore; fallback to Parquet."""
    # Try metastore
    try:
        df_spark = spark.table(tbl_name)
        df_sample = df_spark.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
        if df_sample.count() > 0:
            print('Loaded sample from metastore (sampled)')
            return df_sample
        else:
            print('Metastore sampling returned 0 rows; trying parquet fallback')
    except Exception as e:
        print('Metastore read failed, will try parquet fallback:', e)

    # Parquet fallback
    df_par = spark.read.parquet(parquet_path)
    try:
        larger_fraction = max(sample_fraction * 20, 0.001)
        df_sample = df_par.sample(withReplacement=False, fraction=larger_fraction, seed=seed)
        if df_sample.count() > 0:
            print('Loaded sample from parquet (sampled)')
            return df_sample.limit(max_rows)
    except Exception as e:
        print('Parquet sampling failed or empty, will try direct limit:', e)

    # Last resort: direct limit()
    df_direct = df_par.limit(max_rows)
    if df_direct.count() > 0:
        print('Loaded sample via direct parquet.limit()')
        return df_direct

    return None


# -------------------------------------------------------------------
# Initialize Spark
# -------------------------------------------------------------------
try:
    spark
except NameError:
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()


# -------------------------------------------------------------------
# 1. Load checkpoint
# -------------------------------------------------------------------
try:
    ckpt = load_checkpoint(local_ckpt)
except FileNotFoundError:
    raise RuntimeError(f'Local checkpoint not found at {local_ckpt}. '
                       'Copy it from HDFS first or run the training step.')

# Extract state dict
state = ckpt.get('model_state_dict') if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
if not isinstance(state, dict):
    raise RuntimeError('Checkpoint did not contain a state dict')

# Infer input dimension
weight_tensor = None
for k, v in state.items():
    if 'weight' in k and hasattr(v, 'shape'):
        weight_tensor = v
        break
if weight_tensor is None:
    raise RuntimeError('Could not find weight tensor in checkpoint state dict')
in_features = weight_tensor.shape[1]
print('Inferred model input features =', in_features)

# Build model
model = nn.Linear(in_features, 1)
try:
    model.load_state_dict(state)
    print('Loaded full state dict into model')
except Exception:
    ms = model.state_dict()
    matched = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
    ms.update(matched)
    model.load_state_dict(ms)
    print('Loaded partial/matched state into model')
model.eval()


# -------------------------------------------------------------------
# 2. Load sample data
# -------------------------------------------------------------------
df_sample = get_small_sample(spark, tbl_name, parquet_path, sample_fraction=0.0005, max_rows=3000, seed=42)
if df_sample is None:
    raise RuntimeError('No rows found in metastore or parquet fallback — cannot compute metrics')

df_pd = df_sample.toPandas()
print('Pandas sample rows:', len(df_pd))


# -------------------------------------------------------------------
# 3. Prepare numeric features and target
# -------------------------------------------------------------------
if 'ArrDelay' not in df_pd.columns:
    try:
        df_pd['ArrDelay'] = pd.to_numeric(df_pd.get('ArrDelay', pd.Series(dtype=float)), errors='coerce')
    except Exception:
        df_pd['ArrDelay'] = pd.Series([np.nan] * len(df_pd))

numeric = df_pd.select_dtypes(include=[np.number]).copy()
if 'ArrDelay' not in numeric.columns:
    numeric['ArrDelay'] = pd.to_numeric(df_pd['ArrDelay'], errors='coerce')

feat_cols = [c for c in numeric.columns if c != 'ArrDelay'][:in_features]
if len(feat_cols) < in_features:
    raise RuntimeError(f'Not enough numeric features ({len(feat_cols)}) to match model input ({in_features})')

X = numeric[feat_cols].astype(np.float32).reset_index(drop=True)
y = numeric['ArrDelay'].values.astype(np.float32)

if len(X) == 0:
    raise RuntimeError('Pandas sample is empty after robust loading — cannot compute metrics')


# -------------------------------------------------------------------
# 4. Split and scale data
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if isinstance(ckpt, dict) and 'scaler' in ckpt:
    scaler = ckpt['scaler']
    print('Using scaler loaded from checkpoint (type:', type(scaler), ')')
else:
    print('No scaler in checkpoint; fitting StandardScaler on training split')
    scaler = StandardScaler()
    scaler.fit(X_train.values)

X_train_scaled = scaler.transform(X_train.values).astype(np.float32)
X_test_scaled = scaler.transform(X_test.values).astype(np.float32)


# -------------------------------------------------------------------
# 5. Run inference and compute metrics
# -------------------------------------------------------------------
with torch.no_grad():
    Xt = torch.from_numpy(X_test_scaled)
    preds = model(Xt).cpu().numpy().reshape(-1)

rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
r2 = float(r2_score(y_test, preds))

print('\nPyTorch baseline metrics:')
print(f' RMSE = {rmse:.4f}')
print(f' R2   = {r2:.4f}')


# -------------------------------------------------------------------
# 6. Compare with previous PySpark baselines
# -------------------------------------------------------------------
prev_lr_rmse = 11.2483
prev_lr_r2 = 0.9056
prev_rf_rmse = 17.0213
prev_rf_r2 = 0.7837

print('\nPrevious PySpark baselines:')
print(f' LinearRegression RMSE={prev_lr_rmse:.4f}, R2={prev_lr_r2:.4f}')
print(f' RandomForest     RMSE={prev_rf_rmse:.4f}, R2={prev_rf_r2:.4f}')

print('\nComparison: lower RMSE and higher R² indicate better performance.')


# -------------------------------------------------------------------
# 7. Show a few prediction samples
# -------------------------------------------------------------------
res_df = pd.DataFrame({
    'ArrDelay': y_test.flatten(),
    'prediction': preds
})
print('\nSample predictions:')
print(res_df.head(10).to_string(index=False))

print('\nDone — use these metrics to compare with your earlier models.')