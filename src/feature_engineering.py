from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# 1. Aggregate Features
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_col='AccountId'):
        self.groupby_col = groupby_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby(self.groupby_col)['Amount'].agg([
            ('TotalTransactionAmount', 'sum'),
            ('AverageTransactionAmount', 'mean'),
            ('TransactionCount', 'count'),
            ('TransactionAmountStdDev', 'std')
        ]).reset_index()
        return pd.merge(X, agg_df, on=self.groupby_col, how='left')

# 2. Extract Date/Time Features
class ExtractDateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        X['TransactionHour'] = X[self.time_col].dt.hour
        X['TransactionDay'] = X[self.time_col].dt.day
        X['TransactionMonth'] = X[self.time_col].dt.month
        X['TransactionYear'] = X[self.time_col].dt.year
        return X

# 3. Encode Categorical Variables
from sklearn.preprocessing import OneHotEncoder

class EncodeCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols=None):
        self.categorical_cols = categorical_cols
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X[self.categorical_cols])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=self.encoder.get_feature_names_out(self.categorical_cols),
            index=X.index
        )
        X = X.drop(self.categorical_cols, axis=1)
        return pd.concat([X, encoded_df], axis=1)

# 4. Handle Missing Values
class HandleMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.fill_values = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if X[col].isnull().any():
                if self.strategy == 'mean':
                    self.fill_values[col] = X[col].mean()
                elif self.strategy == 'median':
                    self.fill_values[col] = X[col].median()
                elif self.strategy == 'mode':
                    self.fill_values[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for col, val in self.fill_values.items():
            X[col].fillna(val, inplace=True)
        return X

# 5. Normalize/Standardize
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class NormalizeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = None

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(X[numeric_cols])
        self.numeric_cols = numeric_cols
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X
