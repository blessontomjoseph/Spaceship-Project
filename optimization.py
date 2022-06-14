from sklearn.ensemble import RandomForestClassifier
import model
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
import numpy as np
from sklearn.metrics import accuracy_score
from functools import partial


param_space={
        'n_estimators':hp.quniform('n_estimators',10,100,1),
        'max_depth':hp.quniform('max_depth',1,10,1),
        'min_samples_split':hp.quniform('min_samples_split',2,10,1),
        'min_samples_leaf':hp.quniform('min_samples_leaf',1,10,1),
        'max_features':hp.quniform('max_features',0.1,1,0.1),
        'bootstrap':hp.choice('bootstrap',[True,False]),
        'criterion':hp.choice('criterion',['gini','entropy']),
        'class_weight':hp.choice('class_weight',[None,'balanced']),
        'random_state':hp.choice('random_state',[0,1,2,3,4,5,6,7,8,9]),
        'verbose':hp.choice('verbose',[0,1,2]),
        'warm_start':hp.choice('warm_start',[True,False]),
        'over_fitting':hp.choice('over_fitting',[True,False]),
        'max_samples':hp.choice('max_samples',[0.5,0.75,1,1.25,1.5,1.75,2]),
        'max_leaf_nodes':hp.choice('max_leaf_nodes',[None,2,4,8,16,32,64,128,256,512,1024]),
        'min_impurity_decrease':hp.choice('min_impurity_decrease',[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'min_impurity_split':hp.choice('min_impurity_split',[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'ccp_alpha':hp.choice('ccp_alpha',[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'max_ccp_alpha':hp.choice('max_ccp_alpha',[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'max_iter':hp.choice('max_iter',[0,100,200,300,400,500,600,700,800,900,1000]),
        'tol':hp.choice('tol',[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'presort':hp.choice('presort',[True,False]),
    }

def optimizer(param_space,trainx,trainy,valx,valy):
    model=RandomForestClassifier(**param_space)
    model.fit(trainx,trainy)
    acc=accuracy_score(model.predict(valx,valy))
    return -1*acc

def bayes(param_space,trainx,trainy,valx,valy):
    trials=Trials()
    op_fn=partial(func=optimizer,trainx=trainx,trainy=trainy,valx=valx,valy=valy)
    result=fmin(fn=op_fn,
            space=param_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=100
            )
    return result


def combining_models(preds1,preds2,preds3):
    """this function combines models predictions to make a final prediction"""
    preds=preds1+preds2+preds3
    preds[preds>1]=1
    preds[preds==1]==0
    return preds







