import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def save_model(model, filename):
    """
    Sauvegarde un modèle entraîné dans un fichier.

    Paramètres :
    model : Le modèle entraîné à sauvegarder.
    filename (str) : Nom du fichier de sauvegarde.
    """
    joblib.dump(model, filename)
    # print(f"Modèle sauvegardé sous {filename}")


def split_dataset(dataset, test_size=0.2):
    """
    Sépare un dataset en ensembles d'entraînement et de test.

    Paramètres :
    dataset (pd.DataFrame) : Le dataset contenant les features et la cible.
    test_size (float) : Le pourcentage du dataset à allouer à l'ensemble de test.

    Retourne :
    X_train, X_test, y_train, y_test
    """
    X = dataset.iloc[:, :-1]  # Toutes les colonnes sauf la dernière (features)
    y = dataset.iloc[:, -1]  # Dernière colonne (cible)

    return train_test_split(X, y, test_size=test_size, random_state=42)



def determine_clusters(df):
    """
       Calcul l'inertie en fonction du nombre de clusters
        Paramètres :
        - df: DataFrame

        Sortie :
        Aucune sortie, seulement de l'affichage pour analyse visuelle
        """
    # Calculer la somme des erreurs quadratiques pour chaque nombre de clusters
    inertias = []
    for i in range(1, 11):  # Correction ici : aller jusqu'à 10 inclus
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)

    # Afficher la courbe pour l'inertie/cluster
    plt.plot(range(1, 11), inertias)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Inertie par cluster')
    plt.show()

    return plt


def visualize_with_tsne(df_num, predictions, perplexity=30, random_state=42):
    """
    Réduit les dimensions d'un DataFrame avec PCA et t-SNE, puis visualise les résultats.

    Paramètres :
    - df_num : DataFrame contenant les données numériques.
    - predictions : Série ou tableau contenant les prédictions ou labels pour la coloration des points.
    - perplexity : Paramètre de perplexité pour t-SNE.
    - random_state : Graine aléatoire pour la reproductibilité.
    """
    # Réduction à 2 dimensions avec PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_num.drop(columns=['Cluster'], errors='ignore'))

    # Application de t-SNE pour réduire à 2 dimensions
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(df_pca)

    # Visualisation des résultats
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=predictions, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="Classes")
    plt.title(f"Visualisation des données avec t-SNE (Perplexité : {perplexity})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


def train_kmeans(df, nb_clusters=4):
    """
        Entrainement un optimisation d'un kmeans
        Paramètres :
        - df : DataFrame
        - nb_clusters : Nombre de cluster a créer par le modèle

        Sortie :
        - grid_kmeans : le modele kmeans entrainé et optimisé
        """
    # Initialize the KMeans model
    kmeans = KMeans()

    # Define parameters for GridSearch
    param_grid = {
        'n_clusters': [1,2,3,4,5,6,7,8,9],  # Around the specified nb_clusters
        'init': ['k-means++', 'random'],  # Initialization methods
        'max_iter': [300],  # Classic number of iterations
        'n_init': [10]  # Classic number of restarts
    }

    # GridSearch
    grid_search = GridSearchCV(estimator=kmeans, param_grid=param_grid, cv=3)
    grid_search.fit(df)  # Training with GridSearch

    # Best parameters found
    best_params = grid_search.best_params_

    # Create KMeans model with the best parameters found
    grid_kmeans = KMeans(
        n_clusters=best_params['n_clusters'],
        init=best_params['init'],
        max_iter=best_params['max_iter'],
        n_init=best_params['n_init']
    )


    # Train the model with the best parameters found 
    grid_kmeans.fit(df)

    return grid_kmeans


def predict_data(model, dataset):

    '''
    :param model: a pretrained/preoptimised  model
    :param dataset: any dataset that's relevent for the model
    :return: the predictions of the model
    '''
    predictions = model.predict(dataset)
    return predictions

def marque_predominante_par_cluster(df, cluster_col, marque_col):
    """
    Identifie la marque prédominante dans chaque cluster.

    :param df: DataFrame contenant les aliments avec leurs clusters et leurs marques.
    :param cluster_col: Nom de la colonne contenant les numéros des clusters.
    :param marque_col: Nom de la colonne contenant les marques des aliments.
    :return: Un dictionnaire avec les clusters comme clés et la marque dominante comme valeur.
    """
    # Vérifier que les colonnes existent
    if cluster_col not in df.columns or marque_col not in df.columns:
        raise ValueError("Les colonnes spécifiées ne sont pas présentes dans le DataFrame.")

    # Groupement des données et identification de la marque dominante
    marque_predominante = (
        df.groupby(cluster_col)[marque_col]
        .agg(lambda x: x.value_counts().idxmax())  # Trouver la marque la plus fréquente
    )

    return marque_predominante.to_dict()




