from pydantic import BaseModel
from typing import Optional

class TransactionData(BaseModel):
    Amount: float
    Value: float
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: int
    TransactionAmountStdDev: float
    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float
    month_sin: float
    month_cos: float
    ProductCategory_utility_bill: Optional[int] = 0
    ChannelId_ChannelId_1: Optional[int] = 0
    ChannelId_ChannelId_2: Optional[int] = 0
    ChannelId_ChannelId_3: Optional[int] = 0
    ChannelId_ChannelId_5: Optional[int] = 0
    PricingStrategy_0: Optional[int] = 0
    PricingStrategy_1: Optional[int] = 0
    PricingStrategy_2: Optional[int] = 0
    PricingStrategy_4: Optional[int] = 0

class PredictionResponse(BaseModel):
    is_high_risk: bool
    probability: float
