import pandas as pd
import numpy as np

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