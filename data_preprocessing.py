import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def encode_labels(df, column):
    label_enc = LabelEncoder()
    df[column] = label_enc.fit_transform(df[column])
    return df

def handle_outliers(df, method="winsorize", z_thresh=3, cap_percentile=0.01):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = zscore(df[col].dropna())
        outliers = abs(z_scores) > z_thresh
        if method == "winsorize":
            df[col] = winsorize(df[col], limits=[cap_percentile, cap_percentile])
        elif method == "cap":
            df[col] = df[col].clip(lower=df[col].quantile(cap_percentile), upper=df[col].quantile(1 - cap_percentile))
        elif method == "impute":
            df.loc[outliers, col] = df.loc[~outliers, col].median()
    return df
