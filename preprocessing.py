from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor


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
    data.drop('cabin',axis=1,inplace=True)
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
        data.drop(['name','side','num'],1,inplace=True)
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
    
    def __init__(self,training_data,transforming_data):
        """instance takes input training and transforming data"""
        self.training_data=training_data
        self.transforming_data=transforming_data

    def knn_impute_cat(self,cat_columns):
        """columns must be categorical
            columns are the columns which has non zero nan values
           trasformable data are data you wanna impute using knn 
           but only has esxactly one nan value per row ,so that we 
           can train it efectively"""
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


    def knn_impute_num(self,num_columns):

        """columns must be categorical
            columns are the columns which has non zero nan values
           trasformable data are data you wanna impute using knn 
           but only has esxactly one nan value per row ,so that we 
           can train it efectively"""

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
           

