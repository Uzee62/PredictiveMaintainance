def feature_engineering(df, window_size=5):
    df['Pressure_mean'] = df['Differential_pressure'].rolling(window=window_size, min_periods=1).mean()
    df['Pressure_std'] = df['Differential_pressure'].rolling(window=window_size, min_periods=1).std()
    df['Flow_mean'] = df['Flow_rate'].rolling(window=window_size, min_periods=1).mean()
    df['Flow_std'] = df['Flow_rate'].rolling(window=window_size, min_periods=1).std()
    df['Differential_pressure_lag1'] = df.groupby('Data_No')['Differential_pressure'].shift(1)
    df['Flow_rate_lag1'] = df.groupby('Data_No')['Flow_rate'].shift(1)
    df.fillna(method="bfill", inplace=True)
    return df
