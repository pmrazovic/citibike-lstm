import pandas as pd

def convert_to_sequence(df, lags=1, aheads=0, dropnan=True):
    columns = list(df.columns.values)
    for lag in range(1, lags+1, 1):
        col_idx = 0
        for column in columns:
            df.insert(loc=col_idx, column=column+"_lag_"+str(lag), value=df[column].shift(lag))
            col_idx += 1
    for ahead in range(1, aheads+1, 1):
        for column in columns:
            df[column+"_ahead_"+str(ahead)] = df[column].shift(-ahead)