from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




def model(trainx,trainy,valx,valy):
    """function that outputs the model given input the train and test data"""
    # create the column transformer
    model_rf=Pipeline([('model',RandomForestClassifier())])
    folds=KFold(n_splits=5) #once set this seems reproduceable than settting cv in cross val score
    tr_score=cross_val_score(model_rf,trainx,trainy,cv=folds,scoring='accuracy').mean()
    model_rf.fit(trainx,trainy)
    vl_score=accuracy_score(model_rf.predict(valx),valy)
    return tr_score,vl_score


