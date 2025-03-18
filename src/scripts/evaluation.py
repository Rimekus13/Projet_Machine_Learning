from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score

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
    """
        :param predictions: any labels
        :param df_test: any test dataset relevent for the model
        """
    return silhouette_score(df_test, predictions)


def eval_davies_bouldin(df_test, predictions):
    """
    :param model: any trained model
    :param df_test: Uany test dataset relevent for the model
    """
    db_index = davies_bouldin_score(df_test, predictions)
    print(f"L'indice de Davies-Bouldin est: {db_index}")


def calculer_et_afficher_calinski_harabasz(X, predictions):
    """
    :param model: any trained model
    :param X: Uany test dataset relevent for the model
    """

    ch_index = calinski_harabasz_score(X, predictions)
    print(f"L'indice de Calinski-Harabasz est: {ch_index}")