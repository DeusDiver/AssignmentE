import pandas as pd


# df.isnull().sum() output 0 on all columns, meaning no values are missing in the dataset



gymData = "gym_data.csv"
df = pd.read_csv(gymData)
#print(df.head(), df.columns, df.info(), df.describe())
print(df.info())