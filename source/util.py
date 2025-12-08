import pandas as pd


def impute_missing_values(x: pd.Series, method: str = "mean") -> pd.Series:
    """
    Imputes the mean/median for numeric columns or the mode for other types.
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method == "mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x
