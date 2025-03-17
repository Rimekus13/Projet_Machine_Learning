from sklearn.model_selection import GridSearchCV

def optimize_hyperparameters(model, param_grid, X_train, y_train, cv=5):
    """Optimise les hyperparamètres du modèle."""
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_
