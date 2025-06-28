from sklearn.pipeline import Pipeline
from feature_engineering import (
    AggregateFeatures,
    ExtractDateTimeFeatures,
    EncodeCategorical,
    HandleMissingValues,
    NormalizeFeatures
)

def create_pipeline():
    return Pipeline(steps=[
        ('aggregate', AggregateFeatures(groupby_col='AccountId')),
        ('datetime', ExtractDateTimeFeatures(time_col='TransactionStartTime')),
        ('categorical', EncodeCategorical(categorical_cols=[
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
            'ProductCategory', 'ChannelId', 'PricingStrategy'
        ])),
        ('missing', HandleMissingValues(strategy='mean')),
        ('normalize', NormalizeFeatures(method='standard')),
    ])
