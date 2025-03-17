from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_test, y_test, metric='accuracy'):
    """Évalue le modèle sur l'ensemble de test."""
    y_pred = model.predict(X_test)
    if metric == 'accuracy':
        return accuracy_score(y_test, y_pred)
    elif metric == 'f1':
        return f1_score(y_test, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y_test, y_pred)

def cross_validate_model(model, X, y, cv=5):
    """Effectue une validation croisée."""
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean()

def eval_silouhette(df_test, predictions):
    return silhouette_score(df_test, predictions)


def eval_davies_bouldin(X, labels):
    """
    Fonction qui prend un set de données (X) et des predictions (labels),
    calcule l'indice de Davies-Bouldin et l'affiche.

    :param model: Un modèle de clustering (ex : KMeans)
    :param X: Un tableau de données (matrice de caractéristiques)
    """
    # Calculer l'indice de Davies-Bouldin
    db_index = davies_bouldin_score(X, labels)

    # Afficher l'indice
    print(f"L'indice de Davies-Bouldin est: {db_index}")


def calculer_et_afficher_calinski_harabasz(X, labels):
    """
    Fonction qui prend un set de données (X) et des predictions (labels),
    calcule l'indice de Calinski-Harabasz et l'affiche.

    :param model: Un modèle de clustering (ex : KMeans)
    :param X: Un tableau de données (matrice de caractéristiques)
    """

    # Calculer l'indice de Calinski-Harabasz
    ch_index = calinski_harabasz_score(X, labels)

    # Afficher l'indice
    print(f"L'indice de Calinski-Harabasz est: {ch_index}")