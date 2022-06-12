from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')


def data_load(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    return train, test, val


def column_transformer1(data):
    """expanding cabin and passeengerid"""
    data.columns = [i.lower() for i in data.columns]
    data[['deck', 'num', 'side']] = data['cabin'].str.split('/', expand=True)
    data['num'].astype(float)
    data.drop(['cabin', 'name'], axis=1, inplace=True)
    data[['group', 'passenger']] = data['passengerid'].str.split(
        '_', expand=True).astype(int)
    data.drop('passengerid', axis=1, inplace=True)
    return data


# a manuel encoder for all the data just naively transforms all the cat and bool to num
def transformation(data):
    planet = {'Earth': 1, 'Europa': 2, 'Mars': 3}
    data.homeplanet = data.homeplanet.map(planet)
    destination = {'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3}
    data.destination = data.destination.map(destination)
    data[['cryosleep', 'vip', ]] = data[['cryosleep', 'vip', ]].astype(float)
    deck = {'F': 1, 'C': 2, 'G': 3, 'B': 4, 'E': 5, 'D': 6, 'A': 7, 'T': 8}
    data.deck = data.deck.map(deck)
    data.drop(['side', 'num'], 1, inplace=True)
    try:
        # cuz we gonna use train data in this fn as well which doesnt have transported column
        data['transported'] = data['transported'].astype(float)
        return data
    except:
        return data


def impute_split(data):
    # this is used for training knn
    data_good = data[data.isna().sum(axis=1).eq(0)]
    data_transformable = data[data.isna().sum(
        axis=1).eq(1)]  # transformed wwith knn
    # need fit_transform impute later
    data_non_transformable = data[data.isna().sum(axis=1).gt(1)]
    return data_good, data_transformable, data_non_transformable


class Knn_imputation:
    """defining knn classifier and regressor """
    knn = KNeighborsClassifier(n_neighbors=100)
    knn_r = KNeighborsRegressor(n_neighbors=100)

    def __init__(self, training_data, transforming_data, rest_of_data, categorical, numerical):
        """instance takes input training and transforming data"""
        self.training_data = training_data
        self.transforming_data = transforming_data
        self.rest_of_data = rest_of_data
        self.categorical = categorical
        self.numerical = numerical

    def trans_cols(self):
        cat = [i for i in self.transforming_data.columns[self.transforming_data.isna(
        ).sum().gt(0)] if i in self.categorical]
        num = [i for i in self.transforming_data.columns[self.transforming_data.isna(
        ).sum().gt(0)] if i in self.numerical]
        return cat, num

    def knn_impute_cat(self):
        """columns must be categorical
            columns are the columns which has non zero nan values
           trasformable data are data you wanna impute using knn 
           but only has esxactly one nan value per row ,so that we 
           can train it efectively"""
        cat_columns, _ = self.trans_cols()
        for i in cat_columns:
            train_x = self.training_data.drop(i, axis=1)
            train_y = self.training_data[i]
            test_x = self.transforming_data[self.transforming_data[i].isna()]
            test_x = test_x.drop(i, axis=1)
            self.knn.fit(train_x, train_y)
            preds = self.knn.predict(test_x)
            ind = test_x.index
            self.transforming_data.loc[ind, i] = preds
        return self.transforming_data

    def knn_impute_num(self):
        """columns must be categorical
            columns are the columns which has non zero nan values
           trasformable data are data you wanna impute using knn 
           but only has esxactly one nan value per row ,so that we 
           can train it efectively"""
        _, num_columns = self.trans_cols()
        for i in num_columns:
            train_x = self.training_data.drop(i, axis=1)
            train_y = self.training_data[i]
            test_x = self.transforming_data[self.transforming_data[i].isna()]
            test_x = test_x.drop(i, axis=1)
            self.knn_r.fit(train_x, train_y)
            preds = self.knn_r.predict(test_x)
            ind = test_x.index
            self.transforming_data.loc[ind, i] = preds
        return self.transforming_data

    def knn_implement(self):
        """implementing knn imputation"""
        cat, num = self.trans_cols()
        data_transformable_im = self.knn_impute_cat()
        data_transformable_im = self.knn_impute_num()
        result = pd.concat([self.training_data, data_transformable_im,
                            self.rest_of_data]).sort_index(ascending=True)
        return result


def simple_im(train, test, val, categorical, numerical):
    # we dont want to impute names
    # but we'd impute the rest of the things based i=on median and equalent strategies

    im_c = SimpleImputer(strategy='most_frequent')
    im_n = SimpleImputer(strategy='median')

    train[categorical] = im_c.fit_transform(train[categorical])
    train[numerical] = im_n.fit_transform(train[numerical])

    val[categorical] = im_c.transform(val[categorical])
    val[numerical] = im_n.transform(val[numerical])

    test['transported'] = np.zeros([test.shape[0]])
    test[categorical] = im_c.transform(test[categorical])
    test[numerical] = im_n.transform(test[numerical])
    test.drop('transported', axis=1, inplace=True)
    return train, test, val


def final_col_trans(train, test, val):

    cols_to_transform = ['homeplanet', 'destination', 'deck', 'side']
    encoder = ColumnTransformer(
        [('o_encoding', OrdinalEncoder(), cols_to_transform)])
    trans_pipe = Pipeline([('encoding', encoder)])

    train[cols_to_transform] = trans_pipe.fit_transform(
        train[cols_to_transform])
    val[cols_to_transform] = trans_pipe.transform(val[cols_to_transform])
    test[cols_to_transform] = trans_pipe.transform(test[cols_to_transform])
    return train, test, val
