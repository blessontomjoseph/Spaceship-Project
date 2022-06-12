import model
from hyperopt import hp,fmin,tpe
from hyperopt.pyll.stochastic import sample
from hyperopt import Trials
import numpy as np
from sklearn.metrics import accuracy_score

# bayesian search logistic regression
seed=5
model=model.model_rf

def objective(parameters,trainx,trainy,valx,valy,model):
    """objective function for """
    ne=int(parameters['n_estimators'])
    md=int(parameters['max_depth'])
    mss=int(parameters['min_samples_split'])
    msl=int(parameters['min_samples_leaf'])
    # extra features
#     mln=parameters['max_leaf_nodes']
#     mf=parameters['max_features']
    model.fit(trainx,trainy)
    preds=model.predict(valx)
    acc=accuracy_score(preds,valy)
    return (1-acc)


def optimizer(trial):
    parameters={'n_estimators':hp.uniform('n_estimators',1,200),
               'max_depth':hp.uniform('max_depth',1,200),
               'min_samples_split':hp.uniform('min_samples_split',2,10),
               'min_samples_leaf':hp.uniform('min_samples_leaf',1,10)}
#               'max_leaf_nodes':hp.uniform('max_leaf_nodes',),
#               'max_features':hp.uniform('ma_features',)}
    best=fmin(fn=objective,space=parameters,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


def bayesian_search():
    trial=Trials()
    best=optimizer(trial)
    return best


def combining_models(preds1,preds2,preds3):
    """this function combines models predictions to make a final prediction"""
    preds=preds1+preds2+preds3
    preds[preds>1]=1
    preds[preds==1]==0
    return preds



