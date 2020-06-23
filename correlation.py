import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import merged master
df = pd.read_csv('data/merged_master.csv', index_col=0)

corr = df.corr()

sns.heatmap(corr)
plt.show()

corr.to_csv('data/correlations.csv')
