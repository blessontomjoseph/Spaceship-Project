import model
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
import numpy as np
from sklearn.metrics import accuracy_score
from functools import partial











def combining_models(preds1,preds2,preds3):
    """this function combines models predictions to make a final prediction"""
    preds=preds1+preds2+preds3
    preds[preds>1]=1
    preds[preds==1]==0
    return preds







