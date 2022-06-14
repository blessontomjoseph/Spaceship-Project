import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd


def plotter(data, columns, fig_size):
    """function plots a bunch of columns of a data"""
    rows = math.ceil(len(columns)/4)
    i = 0
    sns.set_style('darkgrid')
    plt.subplots(rows, 4, figsize=fig_size)
    plt.tight_layout()
    for col in columns:
        i += 1
        plt.subplot(rows, 4, i)
        sns.kdeplot(data[col], shade=True)

# learning curve


class Learning_curve:
    def __init__(self, train_x, train_y, val_x, val_y, model):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.model = model

    def learning_curve(self):
        loss_tr = []
        loss_val = []
        points = np.linspace(10, len(self.train_x), 50)
        for i in points:
            i = math.ceil(i)
            score = cross_val_score(
                self.model, self.train_x[:i], self.train_y[:i], cv=3, scoring='accuracy').mean()
            self.model.fit(self.train_x[:i], self.train_y[:i])
            loss1 = 1-score
            loss2 = 1-accuracy_score(self.val_y,
                                     self.model.predict(self.val_x))
            loss_tr.append(loss1)
            loss_val.append(loss2)
        return loss_tr, loss_val

    def plot(self):
        l1, l2 = self.learning_curve()
        plt.plot([i for i in range(len(l1))], l1)
        plt.plot([i for i in range(len(l2))], l2)
        plt.ylim((0, 0.3))
        plt.legend(['train', 'test'])


def data_info(data):
    details = {'info': data.info(), 'description': data.describe(),
               'null': data.isna().sum()}
    return details


def fetch_submission(predictions):
    """function that outputs the submission file given input the raw predictions
    output from a model"""

    test_original = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
    test_original['Transported'] = predictions.astype(bool)
    submission = test_original[['PassengerId', 'Transported']]
    submission.to_csv('submission.csv', index=False)


def plotmi(mi):
    sns.barplot(mi['mi_score'], mi.index)
