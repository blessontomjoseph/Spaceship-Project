from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()


def results(trainx, trainy, valx, valy, test, params, model=model):
    """function that trains a model and validates it"""
    model = model.set_params(**params)
    # once set this seems reproduceable than settting cv in cross val score
    folds = KFold(n_splits=5)
    tr_score = cross_val_score(
        model, trainx, trainy, cv=folds, scoring='accuracy').mean()
    model.fit(trainx, trainy)
    vl_score = accuracy_score(model.predict(valx), valy)
    test_preds = model.predict(test)
    return tr_score, vl_score, test_preds
