# Read the file into dataframes
import pandas as pd
df = pd.read_excel("jektistravel.xlsx",index_col=False,keep_default_na=True,sheet_name='Sheet1', header=0)

df.columns

df.describe()

df.info()

df.dtypes

df.head()

df.tail()