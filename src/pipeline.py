from sklearn.pipeline import Pipeline
from feature_engineering import (
    AggregateFeatures,
    CyclicalDateTimeFeatures,
    EncodeCategorical,
    HandleMissingValues,
    NormalizeFeatures
)

def create_pipeline():
    return Pipeline(steps=[
        ('aggregate', AggregateFeatures(groupby_col='AccountId')),
        ('datetime', CyclicalDateTimeFeatures(time_col='TransactionStartTime')),
        ('categorical', EncodeCategorical(categorical_cols=[
            'ProviderId', 'ProductId',
            'ProductCategory', 'ChannelId', 'PricingStrategy'
        ])),
        ('missing', HandleMissingValues(strategy='mean')),
        ('normalize', NormalizeFeatures(
            method='standard',
            exclude_columns=['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        )),
    ])
