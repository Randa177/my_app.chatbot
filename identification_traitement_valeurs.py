import pandas as pd
import numpy as np


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer,SimpleImputer
from sklearn.impute import IterativeImputer

import seaborn as sns

# Read the file into dataframes
import pandas as pd
df = pd.read_excel("jektistravel.xlsx",index_col=False,keep_default_na=True,sheet_name='Sheet1', header=0) 
print(df.head())

print(df.info())

print(df.info())

# Visualiser les valeurs manquantes avec missingno
import matplotlib.pyplot as plt
import missingno as msno
msno.matrix(df)
plt.show()

msno.bar(df)
plt.show()

# Tableau des valeurs manquantes par colonne
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_table = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print(missing_table)

