import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from wordcloud import WordCloud
from preprocessing import get_categorical
import os

def mean_column_per_cluster(labels,dataset,cluster_id=1):

    cluster_points = dataset[labels == cluster_id]

    cluster_mean = cluster_points.mean(axis=0)

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(cluster_mean)), cluster_mean, tick_label=[f"Colonne {i + 1}" for i in range(len(cluster_mean))])
    plt.xlabel('Caractéristiques')
    plt.ylabel('Moyenne')
    plt.title(f'Moyennes des caractéristiques pour le cluster {cluster_id}')
    plt.show()


def word_cloud(df):

    output_folder = "../../image/kmeans_1000"
    # Récupérer les colonnes catégorielles du DataFrame
    df_cat = df.select_dtypes(exclude=['number'])

    # Concaténer toutes les valeurs textuelles en une seule chaîne par colonne
    columns_values_list = {col: ' '.join(df_cat[col].astype(str)) for col in df_cat.columns}

    # Créer un nuage de mots pour chaque colonne catégorielle
    for col, text in columns_values_list.items():
        # Générer le nuage de mots pour la colonne
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        # Afficher le nuage de mots
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Nuage de mots pour la colonne: {col}', fontsize=16)
        plt.axis('off')  # Masquer les axes
        filename = f"{output_folder}/wordcloud_kmeans_1000_{col}.png"
        plt.savefig(filename, bbox_inches='tight')

import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def word_cloud_for_cluster(df, cluster_number, predictions, output_folder="../../image/clusters"):
    """
    Crée un nuage de mots pour les colonnes catégorielles d'un cluster spécifique.

    Paramètres :
    - df : DataFrame contenant les données.
    - cluster_number : Numéro du cluster pour lequel créer le nuage de mots.
    - predictions : Série ou tableau contenant les labels de cluster.
    - output_folder : Dossier où sauvegarder les images des nuages de mots.
    """
    # Vérifier si le DataFrame contient des colonnes catégorielles
    df_cat = df.select_dtypes(exclude=['number'])
    if df_cat.empty:
        print("Aucune colonne catégorielle trouvée dans le DataFrame.")
        return

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Dossier créé : {output_folder}")
        except Exception as e:
            print(f"Erreur lors de la création du dossier : {e}")
            return

    # Filtrer les données pour obtenir uniquement celles du cluster spécifié
    cluster_data = df[predictions == cluster_number]

    # Concaténer toutes les valeurs textuelles en une seule chaîne par colonne
    columns_values_list = {col: ' '.join(cluster_data[col].astype(str)) for col in df_cat.columns}

    # Créer un nuage de mots pour chaque colonne catégorielle
    for col, text in columns_values_list.items():
        # Générer le nuage de mots pour la colonne
        wordcloud = WordCloud(width=800, height=400, background_color='white',include_numbers=True).generate(text)

        # Afficher le nuage de mots
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Nuage de mots pour la colonne: {col} (Cluster {cluster_number})', fontsize=16)
        plt.axis('off')  # Masquer les axes

        # Sauvegarder le nuage de mots
        filename = f"{output_folder}/wordcloud_cluster_{cluster_number}_{col}.png"
        try:
            plt.savefig(filename, bbox_inches='tight')
            print(f"Image sauvegardée : {filename}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'image : {e}")

        plt.close()

# Exemple d'utilisation :
# word_cloud_for_cluster(df, cluster_number=2, predictions=predictions)


