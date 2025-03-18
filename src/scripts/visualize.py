import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def mean_column_per_cluster(labels,dataset,cluster_id=1):

    cluster_points = dataset[labels == cluster_id]

    cluster_mean = cluster_points.mean(axis=0)

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(cluster_mean)), cluster_mean, tick_label=[f"Colonne {i + 1}" for i in range(len(cluster_mean))])
    plt.xlabel('Caractéristiques')
    plt.ylabel('Moyenne')
    plt.title(f'Moyennes des caractéristiques pour le cluster {cluster_id}')
    plt.show()