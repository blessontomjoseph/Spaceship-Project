from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

model_rf=Pipeline([('model',RandomForestClassifier())])

def model_rf(trainx,trainy,valx,valy,model):
    """function that trains a model and validates it"""
    folds=KFold(n_splits=5) #once set this seems reproduceable than settting cv in cross val score
    tr_score=cross_val_score(model,trainx,trainy,cv=folds,scoring='accuracy').mean()
    model.fit(trainx,trainy)
    vl_score=accuracy_score(model.predict(valx),valy)
    return tr_score,vl_score
