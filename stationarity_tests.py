from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series, column_name):
    result = adfuller(series)
    return {
        "Column": column_name,
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary": result[1] < 0.05
    }

def kpss_test(series, column_name):
    statistic, p_value, Critical_Value, _ = kpss(series, regression='c', nlags="auto")
    return {
        "Column": column_name,
        "KPSS Statistic": statistic,
        "p-value": p_value,
        "Stationary": p_value > 0.05,
        "Critical_Value": Critical_Value
    }
