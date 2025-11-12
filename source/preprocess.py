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
