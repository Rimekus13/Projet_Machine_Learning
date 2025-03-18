import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer

import sys
sys.path.append('../scripts')
from preprocessing import clear_missing_data, delete_columnns_treshold, non_useful_columns, clear_missing_line, get_numerical, get_categorical, removal_of_duplicates
from pretraitement import imputation_of_categorical_val, imputation_of_numerical_val, onehotencoder
from training import determine_clusters, train_kmeans,predict_data
from visualize import word_cloud
from evaluation import *
from visualize import *

# Chargement du dataset

path = "../data/dataset_10000.csv" # Chemin du dataset
df = pd.read_csv(path, nrows=10000, sep=',',encoding="utf-8")

# Suppressions des colonnes avec 100% données manquantes
df = clear_missing_data(df)

# Suppressions des colonnes avec threshold données manquantes, 70% par defaut
df = delete_columnns_treshold(df)

# Suppressions des colonnes non pertinentes, soit une lsite par defaut, sous passer une liste en argument 2
df = non_useful_columns(df)

# Suppressions des lignes vides
df = clear_missing_line(df)

# Netoyage des doublons
df = removal_of_duplicates(df)

df_num = df.select_dtypes(include=['number'])  # Garder uniquement les colonnes numériques
df_cat = df.select_dtypes(exclude=['number'])  # Garder uniquement les colonnes catégorielles

# Imputation des valeurs manquantes pour les colonnes numériques
df_num = imputation_of_numerical_val(df_num)

# Imputation des valeurs manquantes pour les colonnes catégorielles
df_cat = imputation_of_categorical_val(df_cat)

# Assurer que df_num et df_cat ont les mêmes index que df
df[df_num.columns] = df_num
df[df_cat.columns] = df_cat

#encodage des variables
df = onehotencoder(df)

#Scaling des donnees

scaler = StandardScaler()
df_num_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)

# Réassembler les données (optionnel : si besoin de garder les catégories)
df_scaled = pd.concat([df_num_scaled, df_cat], axis=1)

# Trouver le bon nombre de clusters, (graphique visuel)
#determine_clusters(df_num_scaled)

#Entraîner le modèle avec un nombre de clusters optimal (ex: 4)
model = train_kmeans(df_num, nb_clusters=6)

predictions = predict_data(model, df_num)

# Ajouter les prédictions au DataFrame
df_num['Cluster'] = predictions


