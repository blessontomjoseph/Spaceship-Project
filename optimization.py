import model
from hyperopt import hp,fmin,tpe
from hyperopt.pyll.stochastic import sample
from hyperopt import Trials
import numpy as np
from sklearn.metrics import accuracy_score


# bayesian search logistic regression
# class Bayes:
#     def __init__(self,trainx,trainy,valx,valy,model):
#         self.trainx=trainx
#         self.trainy=trainy
#         self.valx=valx
#         self.valy=valy
#         self.model=model
#         self.parameters={'n_estimators':hp.uniform('n_estimators',1,200),
#                         'max_depth':hp.uniform('max_depth',1,200),
#                         'min_samples_split':hp.uniform('min_samples_split',2,10),
#                         'min_samples_leaf':hp.uniform('min_samples_leaf',1,10)}
# #                       'max_leaf_nodes':hp.uniform('max_leaf_nodes',),
# #                       'max_features':hp.uniform('ma_features',)}


#     def objective(self):
#         """objective function for """
#         ne=int(self.parameters['n_estimators'])
#         md=int(self.parameters['max_depth'])
#         mss=int(self.parameters['min_samples_split'])
#         msl=int(self.parameters['min_samples_leaf'])
#         # extra features
#     #     mln=parameters['max_leaf_nodes']
#     #     mf=parameters['max_features']
#         # model=model_rf
#         self.model.fit(self.trainx,self.trainy)
#         preds=self.model.predict(self.valx)
#         acc=accuracy_score(preds,self.valy)
#         return (1-acc)


#     def optimizer(self,trial):
#         best=fmin(fn=self.objective,space=self.parameters,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(5))
#         return best

seed=5
def objective(parameters,trainx,trainy,valx,valy):
    """objective function for """
    ne=int(parameters['n_estimators'])
    md=int(parameters['max_depth'])
    mss=int(parameters['min_samples_split'])
    msl=int(parameters['min_samples_leaf'])
#     mln=parameters['max_leaf_nodes']
#     mf=parameters['max_features']
    
    model=RandomForestClassifier(n_estimators=ne,max_depth=md,min_samples_split=mss,min_samples_leaf=msl)
    model.fit(trainx,trainy)
    preds=model.predict(valx)
    acc=accuracy_score(preds,valy)
    return (1-acc)

def optimizer(trial,trainx,trainy,valx,valy):
    parameters={'n_estimators':hp.uniform('n_estimators',1,200),
               'max_depth':hp.uniform('max_depth',1,200),
               'min_samples_split':hp.uniform('min_samples_split',2,10),
               'min_samples_leaf':hp.uniform('min_samples_leaf',1,10)}
#                'max_leaf_nodes':hp.uniform('max_leaf_nodes',),
#                'max_features':hp.uniform('ma_features',)}
    best=fmin(fn=objective(parameters,trainx,trainy,valx,valy),space=parameters,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(5))
    return best


def bayesian_search(trainx,trainy,valx,valy):
    trial=Trials()
    best=optimizer(trial,trainx,trainy,valx,valy)
    return best


def combining_models(preds1,preds2,preds3):
    """this function combines models predictions to make a final prediction"""
    preds=preds1+preds2+preds3
    preds[preds>1]=1
    preds[preds==1]==0
    return preds







