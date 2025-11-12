import pandas as pd
import numpy as np
from ..utils import desribe_numeric_col
from ..utils import impute_missing_values

def basic_cleaning(df: pd.Dataframe) -> pd.Dataframe:
    df = df.copy()
    #dropping columns
    to_drop = ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen", "domain", "country", "visited_learn_before_booking", "visited_faq"]
    df = df.drop(columns = [c for c in to_drop])

    #do string cleaning
    for c in ["lead_indicator", "lead_id", "customer_code"]:
        df[c].replace("", np.nan, inplace=True)
    
    df = df.dropna(subset=["lead_indicator", "lead_id"], axis = 0)
    df = df[df.source == "signup"]
    return df

def split_cat_cont(df: pd.DataFrame):
    vars = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in vars:
        df[col] = df[col].astype("object")
    
    cont_vars = df.loc[:, ((df.dtypes=="float64")|(df.dtypes=="int64"))]
    cat_vars = df.loc[:, (df.dtypes=="object")]

    return cat_vars, cont_vars

def handle_outliers(cont_vars: pd.Dataframe) -> pd.DataFrame:
    cont_vars_cleaned = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()), upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars_cleaned.apply(describe_numeric_col).T
    outlier_summary.to_csv('./artifacts/outlier_summary.csv')
    return cont_vars_cleaned

def impute(cat_vars: pd.DataFrame, cont_vars: pd.Dataframe):
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")

    # Continuous variables missing values
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T

    # Cat vars missing values
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T

    return cat_vars, cont_vars



