"""plot_heatmap.py

Reusable helper to compute a correlation matrix and plot it as a seaborn heatmap
with a matplotlib fallback. Designed to be safe in Jupyter notebooks and to accept
both pandas.DataFrame and pyspark.sql.DataFrame objects.

Usage (in a Jupyter cell):

from scripts.plot_heatmap import plot_corr_heatmap
# If you have a Spark DataFrame `df`: it will sample and convert to pandas safely
plot_corr_heatmap(df, sample_fraction=0.02, max_rows=50000, figsize=(8,8))

# If you already have a pandas DataFrame `pdf` containing numeric columns:
plot_corr_heatmap(pdf, from_pandas=True, figsize=(10,10), annot=True)

API:
- plot_corr_heatmap(df_or_pdf, from_pandas=False, numeric_cols=None,
                    sample_fraction=0.02, max_rows=50000, seed=42,
                    annot=True, fmt='.2f', cmap='vlag', center=0,
                    figsize=(8,8), save_path=None, show=True)

The function returns the pandas correlation DataFrame for further inspection.
"""

from typing import Optional, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False

try:
    import pandas as pd
except Exception:
    raise RuntimeError("pandas is required by scripts/plot_heatmap.py")


def _is_spark_df(obj) -> bool:
    # duck-type check to avoid importing pyspark if not present
    return hasattr(obj, "rdd") and hasattr(obj, "schema") and hasattr(obj, "limit")


def _sample_spark_df_to_pandas(spark_df, sample_fraction: float, max_rows: int, seed: int) -> pd.DataFrame:
    # try sampling; if it fails or yields empty, fallback to limit
    try:
        sampled = spark_df.select(*spark_df.columns).na.drop().sample(False, sample_fraction, seed)
        pdf = sampled.toPandas()
        if pdf.shape[0] == 0:
            raise ValueError("sample returned 0 rows")
        # if it's too large still, downsample
        if pdf.shape[0] > max_rows:
            pdf = pdf.sample(n=max_rows, random_state=seed)
        return pdf
    except Exception:
        # fallback: collect up to max_rows using limit
        try:
            pdf = spark_df.select(*spark_df.columns).na.drop().limit(max_rows).toPandas()
            return pdf
        except Exception as ex:
            raise RuntimeError("Failed to convert Spark DataFrame to pandas for plotting: " + str(ex))


def _select_numeric_columns(pdf: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if numeric_cols:
        missing = [c for c in numeric_cols if c not in pdf.columns]
        if missing:
            warnings.warn(f"Requested numeric_cols not present in dataframe and will be ignored: {missing}")
        use_cols = [c for c in numeric_cols if c in pdf.columns]
        return pdf[use_cols]
    # infer numeric columns
    num_pdf = pdf.select_dtypes(include=["number"]).copy()
    return num_pdf


def plot_corr_heatmap(
    df_or_pdf: Union["pyspark.sql.DataFrame", pd.DataFrame],
    from_pandas: bool = False,
    numeric_cols: Optional[Sequence[str]] = None,
    sample_fraction: float = 0.02,
    max_rows: int = 50000,
    seed: int = 42,
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "vlag",
    center: Optional[float] = 0,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> pd.DataFrame:
    """Compute correlation matrix and plot heatmap.

    Args:
        df_or_pdf: Spark DataFrame or pandas DataFrame.
        from_pandas: set True if df_or_pdf is already a pandas DataFrame (skips spark checks).
        numeric_cols: optional list of columns to include (must be numeric); if None, infer numeric columns.
        sample_fraction: fraction to sample when converting Spark DataFrame to pandas.
        max_rows: maximum rows to pull into pandas for correlation (limits memory use).
        seed: RNG seed for sampling.
        annot, fmt, cmap, center: seaborn heatmap plotting options.
        figsize: figure size in inches.
        save_path: optional path to save the figure (PNG). Works in notebooks and scripts.
        show: if True call plt.show(). If False, returns the figure object for external handling.

    Returns:
        pandas.DataFrame of the correlation matrix (corr).
    """

    # Convert Spark DataFrame to pandas if needed
    if not from_pandas and _is_spark_df(df_or_pdf):
        pdf = _sample_spark_df_to_pandas(df_or_pdf, sample_fraction=sample_fraction, max_rows=max_rows, seed=seed)
    elif from_pandas or isinstance(df_or_pdf, pd.DataFrame):
        pdf = df_or_pdf.copy()
    else:
        # last resort: try to treat as pandas-like
        try:
            pdf = pd.DataFrame(df_or_pdf)
        except Exception as e:
            raise ValueError("Unsupported dataframe input. Provide a pyspark DataFrame or a pandas DataFrame")

    # select numeric columns
    num_pdf = _select_numeric_columns(pdf, numeric_cols)
    if num_pdf.shape[1] == 0:
        raise ValueError("No numeric columns available for correlation heatmap")

    # drop rows with NA in numeric columns
    num_pdf = num_pdf.dropna()
    if num_pdf.shape[0] == 0:
        raise ValueError("No rows left after dropping NA in numeric columns")

    # compute correlation
    corr = num_pdf.corr()

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    try:
        if _HAS_SEABORN and sns is not None:
            sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, center=center, ax=ax)
        else:
            # matplotlib fallback: imshow with colorbar and ticks
            im = ax.imshow(corr.values, cmap="bwr", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
            if annot:
                # add text annotations
                for i in range(len(corr.index)):
                    for j in range(len(corr.columns)):
                        t = f"{corr.values[i,j]:.2f}"
                        ax.text(j, i, t, ha="center", va="center", color="black", fontsize=8)
        ax.set_title("Correlation matrix")
    except Exception as e:
        # If seaborn plotting fails, fallback to matplotlib imshow and still return corr
        warnings.warn(f"Heatmap plotting failed, falling back to matplotlib imshow. Error: {e}")
        ax.clear()
        im = ax.imshow(corr.values, cmap="bwr", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)

    plt.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save heatmap to {save_path}: {e}")

    if show:
        plt.show()
    else:
        return corr

    return corr
