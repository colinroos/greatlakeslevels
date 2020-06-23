import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('data/merged_master.csv', index_col=0)

lakes = ['Superior', 'Michigan-Huron', 'St. Clair', 'Erie', 'Ontario']

for lake in lakes:
    # Extract targets to predict
    lake_levels = pd.DataFrame()
    lake_levels['y'] = df[lake]
    lake_levels['ds'] = lake_levels.index
    lake_levels.reset_index(inplace=True, drop=True)

    # Initialize the Prophet
    m = Prophet(changepoint_prior_scale=0.05, changepoint_range=0.9)
    m.fit(lake_levels)

    # Extend dataframe in to the future
    future_lake_levels = m.make_future_dataframe(periods=120, freq='M')

    # Predict futures
    future = m.predict(future_lake_levels)
    future['y'] = lake_levels['y']
    future.to_csv(f'data/predictions_{lake}.csv')

    fig = m.plot(future, xlabel='', ylabel='Lake Level (m)', figsize=(12, 3))
    plt.show()



