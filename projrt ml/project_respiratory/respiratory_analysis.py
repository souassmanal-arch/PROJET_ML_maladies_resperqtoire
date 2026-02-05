# =========================
# 1. Import des librairies
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE


# =========================
# 2. Chargement du dataset
# =========================
data = pd.read_csv("dataset.csv")

print("Aperçu des données :")
print(data.head())

# Sauvegarde de l'aperçu du dataset en image
# Sauvegarde de l'aperçu du dataset en image (Version Améliorée)
plt.figure(figsize=(14, 5))
ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)

# Création du tableau
tbl = table(ax, data.head(), loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 2)

# Coloriage du tableau pour qu'il soit plus clair
for (row, col), cell in tbl.get_celld().items():
    if row == 0: # Header
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e') # Bleu foncé
        cell.set_edgecolor('white')
    else:
        cell.set_edgecolor('#dddddd')
        if row % 2 == 1:
            cell.set_facecolor('#e6e6e6') # Gris clair
        else:
            cell.set_facecolor('white')

plt.title("Aperçu des Données (5 premières lignes)", fontsize=16, weight='bold', color='#333333')
plt.savefig("dataset_preview.png", bbox_inches='tight', dpi=300)
plt.close()

print("\nInfos générales :")
print(data.info())


# =========================
# 3. Analyse exploratoire
# =========================
print("\nDistribution des classes :")
print(data['disease_class'].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(x='disease_class', data=data)
plt.title("Distribution des maladies respiratoires")
plt.savefig("distribution_classes.png")
plt.close()

# Histogrammes des caractéristiques
# Histogrammes des caractéristiques
features = ['age', 'oxygen_saturation', 'heart_rate', 'respiratory_rate']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1) # 2x2 grid for 4 features
    if feature in data.columns:
        sns.histplot(data=data, x=feature, hue='disease_class', kde=True, element="step")
        plt.title(f"Distribution: {feature}")
plt.tight_layout()
plt.savefig("feature_histograms.png")
plt.close()

# 3b. Heatmap de Corrélation
plt.figure(figsize=(10, 8))
numeric_data = data.select_dtypes(include=[np.number])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap de Corrélation des Caractéristiques")
plt.savefig("correlation_heatmap.png")
plt.close()

# 3c. Heatmap des Caractéristiques Moyennes par Classe
plt.figure(figsize=(10, 6))
# Grouper par classe et calculer la moyenne des features numériques
mean_features = data.groupby('disease_class')[numeric_data.columns.drop('disease_class', errors='ignore')].mean()
# Normaliser pour une meilleure visualisation (MinMax scaling local pour le plot)
mean_features_norm = (mean_features - mean_features.min()) / (mean_features.max() - mean_features.min())
sns.heatmap(mean_features_norm, annot=True, cmap='YlGnBu')
plt.title("Heatmap des Caractéristiques Moyennes par Classe (Normalisé)")
plt.savefig("features_by_class_heatmap.png")
plt.close()


# =========================
# 4. Prétraitement et Séparation X / y
# =========================

# Encodage des variables catégorielles (Text -> Nombres)
cat_cols = ['gender', 'smoker', 'fever', 'cough', 'breath_difficulty']
for col in cat_cols:
    if col in data.columns:
        data[col] = data[col].astype('category').cat.codes

# Séparation des features et de la cible
X = data.drop('disease_class', axis=1)
y = data['disease_class']
feature_names = X.columns # Sauvegarde des noms de colonnes pour l'importance des features


# =========================
# 5. Traitement des valeurs manquantes
# =========================
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


# =========================
# 6. Normalisation / Standardisation
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)


# =========================
# 7. Gestion du déséquilibre des classes
# =========================
smote = SMOTE(random_state=42)
# Check if we have enough samples for SMOTE (needs > 1 sample per class)
try:
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("\nClasses après SMOTE :")
    print(pd.Series(y_resampled).value_counts())
except ValueError:
    # Fallback if dataset is too small for SMOTE
    print("\nAttention: Pas assez de données pour SMOTE. Utilisation des données originales.")
    X_resampled, y_resampled = X, y


# =========================
# 8. Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)


# =========================
# 9. Modèle Decision Tree
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)


# =========================
# 10. Modèle Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


# =========================
# 11. Évaluation des modèles
# =========================
def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision :", precision_score(y_true, y_pred, average='weighted', zero_division=0))
    print("Recall :", recall_score(y_true, y_pred, average='weighted', zero_division=0))
    print("F1-score :", f1_score(y_true, y_pred, average='weighted', zero_division=0))
    print("\nRapport de classification :")
    print(classification_report(y_true, y_pred, zero_division=0))


evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)




# =========================
# 13. Matrices de confusion (Heatmaps)
# =========================
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,4))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion - Decision Tree")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("confusion_matrix_dt.png")
plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.title("Matrice de confusion - Random Forest")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("confusion_matrix_rf.png")
plt.close()
