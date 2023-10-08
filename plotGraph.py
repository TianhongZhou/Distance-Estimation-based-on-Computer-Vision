import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
df = df.drop(columns=["name", "dist"])
plt.figure(figsize=(15, 15))
plt.title('Correlation Heatmap of Features', y=1.05, size=15)
sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, linecolor='white', annot=False)
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()