import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from wordcloud import WordCloud
from preprocessing import get_categorical


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

