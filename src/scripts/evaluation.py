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
