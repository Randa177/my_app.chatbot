# Import des librairies nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lire le fichier Excel (à partir de la ligne 3 avec les bons noms de colonnes)
data = pd.read_excel("jektistravel.xlsx", sheet_name='Sheet1', header=0)

# Renommer les colonnes pour simplifier
data.columns = ['ID', 'Lien', 'Destination', 'Prix', 'Date']

# Nettoyer la colonne 'Prix' (enlever " TND", espaces) et convertir en float
data['Prix'] = data['Prix'].replace(r'\s*TND', '', regex=True).replace(r'\s', '', regex=True).astype(float)

# Afficher les premières lignes pour vérifier
print(data.head())
print("Colonnes :", data.columns)

# Description statistique
print(data['Prix'].describe())

# Détection des outliers via IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)]

# Appliquer sur la colonne 'Prix'
outliers_iqr = detect_outliers_iqr(data, 'Prix')
print("Outliers détectés avec IQR :")
print(outliers_iqr[['Prix']])

# Marquer les outliers dans le DataFrame
data['Outlier'] = False
data.loc[outliers_iqr.index, 'Outlier'] = True

# Visualisation avec un scatter plot (Date vs Prix)
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='Prix', y='Date', hue='Outlier', palette='viridis')
plt.xlabel('Prix (TND)')
plt.ylabel('Date du voyage')
plt.title("Nuage de points entre le Prix et la Date (valeurs extrêmes en couleur)")
plt.show()

# Visualisation supplémentaire : boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x='Prix')
plt.xlabel('Prix (TND)')
plt.title("Boxplot des Prix avec outliers")
plt.show()

# Affichage de l'histogramme initial
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(data['Prix'], bins=30, color='skyblue', edgecolor='black')
plt.title('Données originales')

# Transformation logarithmique des données (en excluant les valeurs négatives)
data_log = np.log(data['Prix'][data['Prix'] > 0])

# Affichage de l'histogramme après transformation logarithmique
plt.subplot(1, 2, 2)
plt.hist(data_log, bins=30, color='salmon', edgecolor='black')
plt.title('Transformation logarithmique')

plt.tight_layout()
plt.show()

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Générer des données de régression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.5, random_state=42)

# Ajouter des outliers à la cible
outlier_indices = np.random.choice(len(y), size=100, replace=False)
outlier_values = np.random.normal(loc=100, scale=10, size=len(outlier_indices))
y[outlier_indices] = outlier_values

# Séparer les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Liste des quantiles à tester
quantile_alphas = [0.1, 0.5, 0.9]

# Dictionnaire pour stocker les prédictions
predictions = {}

# Entraîner et évaluer le modèle pour chaque quantile
for alpha in quantile_alphas:
    model = GradientBoostingRegressor(loss='quantile', alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[alpha] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Quantile {alpha:.1f} - MAE: {mae:.2f}")

# Visualisation des résultats pour les 3 quantiles
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label='Valeurs réelles', alpha=0.5)

for alpha in quantile_alphas:
    plt.plot(predictions[alpha], label=f"Quantile {alpha:.1f}")

plt.title("Comparaison des prédictions par quantile (Gradient Boosting)")
plt.xlabel("Index")
plt.ylabel("Valeur prédite / réelle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Détection des outliers avec IQR pour la colonne 'Prix'
outliers_iqr = detect_outliers_iqr(data, 'Prix')

# Remplacer les valeurs aberrantes par np.nan dans la colonne 'Prix'
data.loc[outliers_iqr.index, 'Prix'] = np.nan

# Afficher les données après remplacement
print(data)

# Afficher uniquement les valeurs aberrantes de la colonne 'Prix' après remplacement
print("Valeurs aberrantes remplacées par NaN :")
print(data.loc[outliers_iqr.index, 'Prix'])

# Read the file into dataframes
import pandas as pd
data = pd.read_excel("jektistravel.xlsx",index_col=False,keep_default_na=True,sheet_name='Sheet1', header=0)
print(data.head())

