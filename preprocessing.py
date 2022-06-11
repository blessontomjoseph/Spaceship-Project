from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

def data_load(train_path,test_path):
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    train,val=train_test_split(train,test_size=0.2,random_state=42)
    return train,test,val



def column_transformer1(data):
    """expanding cabin and passeengerid"""
    data.columns=[i.lower() for i in data.columns]
    data[['deck','num','side']]=data['cabin'].str.split('/',expand=True)
    data['num'].astype(float)
    data.drop(['cabin','name'],axis=1,inplace=True)
    data[['group','passenger']]=data['passengerid'].str.split('_',expand=True).astype(int)
    data.drop('passengerid',axis=1,inplace=True)
    return data

    
# a manuel encoder for all the data just naively transforms all the cat and bool to num 
def transformation(data): 
        planet={'Earth':1, 'Europa':2, 'Mars':3}
        data.homeplanet=data.homeplanet.map(planet)
        destination={'TRAPPIST-1e':1, '55 Cancri e':2, 'PSO J318.5-22':3}
        data.destination=data.destination.map(destination)
        data[['cryosleep','vip',]]=data[['cryosleep','vip',]].astype(float)
        deck={'F':1, 'C':2, 'G':3, 'B':4, 'E':5, 'D':6, 'A':7, 'T':8}
        data.deck=data.deck.map(deck)
        data.drop(['side','num'],1,inplace=True)
        try:
            # cuz we gonna use train data in this fn as well which doesnt have transported column
            data['transported']=data['transported'].astype(float)
            return data
        except:
            return data
        

def reverse_transformation(copy_original,current_data):
    # lil' reverse transform  better for eda to make sense
    current_data[['name','side','num']]=copy_original.loc[current_data.index,['name','side','num']]
    current_data['num']=current_data['num'].astype(float)
    planet={1:'Earth', 2:'Europa',3:'Mars'}
    current_data['homeplanet']=current_data['homeplanet'].map(planet)
    destination={ 1:'TRAPPIST-1e', 2:'55 Cancri e', 3:'PSO J318.5-22'}
    current_data.destination=current_data.destination.map(destination)
    deck={1:'F', 2:'C', 3:'G', 4:'B', 5:'E', 6:'D', 7:'A', 8:'T'}
    current_data.deck=current_data.deck.map(deck)
    return current_data

  
def impute_split(data):
    data_good = data[data.isna().sum(axis=1).eq(0)] #this is used for trraining knn
    data_transformable = data[data.isna().sum(axis=1).eq(1)] #transformed wwith knn
    data_non_transformable = data[data.isna().sum(axis=1).gt(1)] #need fit_transform impute later
    return data_good,data_transformable,data_non_transformable




class Knn_imputation:
    """defining knn classifier and regressor """
    knn=KNeighborsClassifier(n_neighbors=100)
    knn_r=KNeighborsRegressor(n_neighbors=100)
    
    def __init__(self,training_data,transforming_data,rest_of_data,categorical,numerical):
        """instance takes input training and transforming data"""
        self.training_data=training_data
        self.transforming_data=transforming_data
        self.rest_of_data=rest_of_data
        self.categorical=categorical
        self.numerical=numerical

        
    def trans_cols(self):
        cat=[i for i in self.transforming_data.columns[self.transforming_data.isna().sum().gt(0)] if i in self.categorical]
        num=[i for i in self.transforming_data.columns[self.transforming_data.isna().sum().gt(0)] if i in self.numerical]
        return cat,num

    def knn_impute_cat(self):
        """columns must be categorical
            columns are the columns which has non zero nan values
           trasformable data are data you wanna impute using knn 
           but only has esxactly one nan value per row ,so that we 
           can train it efectively"""
        cat_columns,_=self.trans_cols()
        for i in cat_columns:
            train_x=self.training_data.drop(i,axis=1)
            train_y=self.training_data[i]
            test_x=self.transforming_data[self.transforming_data[i].isna()]
            test_x=test_x.drop(i,axis=1)
            self.knn.fit(train_x,train_y)
            preds=self.knn.predict(test_x)
            ind=test_x.index
            self.transforming_data.loc[ind,i]=preds
        return self.transforming_data


    def knn_impute_num(self):

        """columns must be categorical
            columns are the columns which has non zero nan values
           trasformable data are data you wanna impute using knn 
           but only has esxactly one nan value per row ,so that we 
           can train it efectively"""
        _,num_columns=self.trans_cols()
        for i in num_columns:
            train_x=self.training_data.drop(i,axis=1)
            train_y=self.training_data[i]
            test_x=self.transforming_data[self.transforming_data[i].isna()]
            test_x=test_x.drop(i,axis=1)
            self.knn_r.fit(train_x,train_y)
            preds=self.knn_r.predict(test_x)
            ind=test_x.index
            self.transforming_data.loc[ind,i]=preds
        return self.transforming_data
    
    def knn_implement(self):
        """implementing knn imputation"""
        cat,num=self.trans_cols()
        data_transformable_im=self.knn_impute_cat()
        data_transformable_im=self.knn_impute_num()
        result=pd.concat([self.training_data,data_transformable_im,self.rest_of_data]).sort_index(ascending=True)
        return result    
       



def simple_im(train,test,val,categorical,numerical):
    # we dont want to impute names
    # but we'd impute the rest of the things based i=on median and equalent strategies

    im_c=SimpleImputer(strategy='most_frequent')
    im_n=SimpleImputer(strategy='median')
    
    train[categorical]=im_c.fit_transform(train[categorical])
    train[numerical]=im_n.fit_transform(train[numerical])
   
    val[categorical]=im_c.transform(val[categorical])
    val[numerical]=im_n.transform(val[numerical])
    
    test['transported']=np.zeros([test.shape[0]])
    test[categorical]=im_c.transform(test[categorical])
    test[numerical]=im_n.transform(test[numerical])
    test.drop('transported',axis=1,inplace=True)
    return train,test,val
    

def name_splitter(data):
    # name splitting to first and last name
    data[['f_name','l_name']]=data.name.str.split(' ',expand=True)
    data.drop('name',axis=1,inplace=True)
    return data



def name_impute(train_data,transform_data):
    # imputing missing names by most frequent names 
    for i in transform_data.homeplanet.unique():
        mf_fname=train_data.loc[train_data.homeplanet==i].f_name.value_counts().index[0]
        mf_lname=train_data[train_data.homeplanet==i].l_name.value_counts().index[0]

        mask=transform_data.loc[(transform_data.homeplanet==i) & (transform_data.f_name.isna())].index
        transform_data.loc[mask,'f_name']=mf_fname
        mask=transform_data.loc[(transform_data.homeplanet==i) & (transform_data.l_name.isna())].index
        transform_data.loc[mask,'l_name']=mf_lname
    return transform_data


def final_col_trans(train,test,val):
    
    cols_to_transform=['homeplanet','destination','deck','side']
    encoder=ColumnTransformer([('o_encoding',OrdinalEncoder(),cols_to_transform)])
    trans_pipe=Pipeline([('encoding',encoder)])

    train[cols_to_transform]=trans_pipe.fit_transform(train[cols_to_transform])
    val[cols_to_transform]=trans_pipe.transform(val[cols_to_transform])
    test[cols_to_transform]=trans_pipe.transform(test[cols_to_transform])
    return train,test,val