from sklearn.ensemble import RandomForestClassifier
import model
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
import numpy as np
from sklearn.metrics import accuracy_score
from functools import partial
import model


p_space={
        'n_estimators':scope.int(hp.quniform('n_estimators',10,100,1)),
        'max_depth':scope.int(hp.quniform('max_depth',1,10,1)),
        'min_samples_split':scope.int(hp.quniform('min_samples_split',2,10,1)),
        'min_samples_leaf':scope.int(hp.quniform('min_samples_leaf',1,10,1)),
        'max_features':hp.quniform('max_features',0.1,1,0.1),
        'bootstrap':hp.choice('bootstrap',[True,False]),
        'criterion':hp.choice('criterion',['gini','entropy']),
        # 'class_weight':hp.choice('class_weight',[None,'balanced']),
        # 'max_samples':hp.choice('max_samples',[0.5,0.75,1,1.25,1.5,1.75,2]),
        # 'max_leaf_nodes':hp.choice('max_leaf_nodes',[None,2,4,8,16,32,64,128,256,512,1024]),
        # 'min_impurity_decrease':hp.choice('min_impurity_decrease',[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        # 'min_impurity_split':hp.choice('min_impurity_split',[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
    }

rf=model.model
def optimizer(param_space,trainx,trainy,valx,valy,model=rf):
    model=rf.set_params(**param_space)
    model.fit(trainx,trainy)
    acc=accuracy_score(model.predict(valx),valy)
    return -1*acc

def bayesian_search(trainx,trainy,valx,valy,param_space=p_space):
    trials=Trials()
    op_fn=partial(optimizer,trainx=trainx,trainy=trainy,valx=valx,valy=valy)
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







