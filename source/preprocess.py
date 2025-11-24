import json
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from util import impute_missing_values
from config import OUTLIER_SUMMARY_FILE, CAT_MISSING_IMPUTE_FILE, SCALER_FILE, COLUMNS_DRIFT_FILE, TRAINING_DATA_FILE, TRAIN_GOLD_FILE

def describe_numeric_col(x: pd.Series) -> pd.Series:
    '''
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    '''
    return pd.Series(
        [x.count(), x.isnull().sum(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Takes a pandas Dataframe and drop columns.
    Then performs string cleaning
    '''
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
    '''
    Splits a pandas dataframe into categorical and continuous
    '''
    vars = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in vars:
        df[col] = df[col].astype("object")
    
    cont_vars = df.loc[:, ((df.dtypes=="float64")|(df.dtypes=="int64"))]
    cat_vars = df.loc[:, (df.dtypes=="object")]

    return cat_vars, cont_vars

def handle_outliers(cont_vars: pd.DataFrame) -> pd.DataFrame:
    '''
    Handles outliers for cont vars and prints summary to csv
    '''
    cont_vars_cleaned = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()), upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars_cleaned.apply(describe_numeric_col).T
    outlier_summary.to_csv(OUTLIER_SUMMARY_FILE)
    return cont_vars_cleaned

def impute(cat_vars: pd.DataFrame, cont_vars: pd.DataFrame):
    '''
    impute missing values using utils function
    '''
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(CAT_MISSING_IMPUTE_FILE)

    # Continuous variables missing values
    cont_vars = cont_vars.apply(impute_missing_values)

    # Cat vars missing values
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T

    return cat_vars, cont_vars

def scale(cont_vars: pd.DataFrame):
    '''
    scale the cont vars
    '''
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    joblib.dump(value=scaler, filename=SCALER_FILE)
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

    return cont_vars

def combine_and_document_drift(cont_vars: pd.DataFrame, cat_vars: pd.DataFrame) -> pd.DataFrame:
    '''
    combine the cont vars and cat vars then dump into json for data drift comp
    '''
    #combining the data
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)

    #exporting data drift artifact
    data_columns = list(data.columns)
    with open(COLUMNS_DRIFT_FILE,'w+') as f:           
        json.dump(data_columns,f)
    
    data.to_csv(TRAINING_DATA_FILE, index=False)

    return data

def binning(data: pd.DataFrame) -> pd.DataFrame:
    '''
    bin the data
    '''
    data = data.copy()
    data['bin_source'] = data['source']
    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {'li' : 'socials', 
            'fb' : 'socials', 
            'organic': 'group1', 
            'signup': 'group1'
            }
    data['bin_source'] = data['source'].map(mapping)
    return data

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    '''
    takes raw data and runs all above functions
    returns a csv with the gold data
    '''
    data = df.copy()
    #apply cleaning
    data = basic_cleaning(data)
    #apply splitting
    cat_vars, cont_vars = split_cat_cont(data)
    #handle outliers
    cont_vars_cleaned = handle_outliers(cont_vars)
    #impute missing
    cat_vars_imputed, cont_vars_imputed = impute(cat_vars, cont_vars_cleaned)
    #scale cont
    cont_vars_scaled = scale(cont_vars_imputed)
    #combine it together
    combined = combine_and_document_drift(cont_vars_scaled, cat_vars_imputed)
    #bin it
    final = binning(combined)
    #save gold data
    final.to_csv(TRAIN_GOLD_FILE, index=False)
    return final

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA_FILE)
    preprocess_pipeline(df)
