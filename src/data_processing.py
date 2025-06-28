from pipeline import create_pipeline
import pandas as pd

df_raw = pd.read_csv("../data/raw/data.csv")

pipeline = create_pipeline()
df_transformed = pipeline.fit_transform(df_raw)

df_transformed.to_csv("../data/processed/data_processed.csv", index=False)