import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import weather data
df_weather = pd.read_csv('data/Daily_weather_data.csv')

print(df_weather.info())
print(df_weather['STATION'].unique())

dfs = []
for station in df_weather['STATION'].unique():
    df_pivot = pd.DataFrame()
    df_filtered = df_weather[df_weather['STATION'] == station]
    df_pivot[f'{station}_date'] = pd.to_datetime(df_weather['DATE'], format='%Y-%m-%d')
    df_pivot[f'{station}_prcp'] = df_filtered['PRCP']
    df_pivot[f'{station}_snow'] = df_filtered['SNOW']
    df_pivot[f'{station}_snwd'] = df_filtered['SNWD']
    df_pivot[f'{station}_tmax'] = df_filtered['TMAX']
    df_pivot[f'{station}_tmin'] = df_filtered['TMIN']
    df_pivot.set_index(f'{station}_date', drop=True, inplace=True)
    dfs.append(df_pivot)

df_merged = pd.merge(dfs[0], dfs[-1], left_index=True, right_index=True)
df_resampled = df_merged.resample('M').mean()
df_isna = df_resampled.isna()
x_dates = df_isna.index.strftime('%Y-%m-%d')
plt.figure(figsize=(10, 5))
ax = sns.heatmap(df_isna.T, cbar=False)
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
df_resampled = df_resampled.ffill()
df_resampled = df_resampled.dropna()
plt.show()

# Import hydro data
df_hydro = pd.read_csv('data/GLHYD_data_metric.csv')
df_hydro['date'] = df_hydro['year'].astype(str) + df_hydro['month']
df_hydro['date'] = pd.to_datetime(df_hydro['date'], format='%Y%b')
df_hydro.set_index('date', drop=True, inplace=True)
df_hydro = df_hydro.resample('M').mean()
df_hydro.drop(columns=['year'], inplace=True)

df_merged_master = pd.merge(df_resampled, df_hydro, left_index=True, right_index=True)
df_merged_master_annual = df_merged_master.resample('A').mean()
df_merged_master.to_csv('data/merged_master.csv')
df_merged_master_annual.to_csv('data/merged_master_annual.csv')


