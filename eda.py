import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df.describe())
df.hist(figsize=(10,8))
plt.show()
plt.figure(figsize=(10,5))
sns.boxplot(data=df[["Age", "Fare"]])
plt.title("Boxplot of Age and Fare")
plt.show()

correlation = df.select_dtypes(include="number").corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df.select_dtypes(include="number"))
plt.show()