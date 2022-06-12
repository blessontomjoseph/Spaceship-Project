from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



def mutual_information(x,y,mask=None):
    """function calculates the mi score in descendinhg trend given x and y"""
    if mask is not None:
        mi=mutual_info_classif(x.iloc[:,:mask],y)
        mi=pd.DataFrame(mi,columns=['mi_score'],index=x.columns[:mask])
    elif mask is None:  
        mi=mutual_info_classif(x,y)
        mi=pd.DataFrame(mi,columns=['mi_score'],index=x.columns)
        
    mi=mi.sort_values("mi_score",ascending=False)
    return mi


def pca_ing(x,standardize=None):
    """function standardizes the data is not standardized and performs pca and outputs its componets in a df also loadings"""
    if standardize:
        sc=StandardScaler()
        x_scaled=sc.fit_transform(x)
        x=pd.DataFrame(x_scaled,columns=x.columns)
    pca=PCA()
    x_pca=pca.fit_transform(x)
    components=[f'pca_{i}' for i in x.columns.values]
    x_pca=pd.DataFrame(x_pca,columns=components)
    loadings=pd.DataFrame(pca.components_.T,columns=components,index=x.columns)
    return x_pca,loadings

def auto_best_features(x,other_data,n_features,standardize_on_pca=False):
    """best features(having most mi scores) among x and its pca version """
    x_pca=pca_ing(x,standardize=standardize_on_pca)
    all_features=x.join(x_pca)
    mutual_info=mutual_information(all_features,other_data)
    selected_cols=mutual_info.index[:n_features]
    other_data_selected=[]
    for i in other_data:
        x_pca=pca_ing(i,standardize=standardize_on_pca)
        all_features=x.join(x_pca)
        other_data_selected.append(all_features[selected_cols])
    return x[selected_cols],other_data_selected            
        


#new features
def create_features(data):
    """function creates the following sorts of features for a given data"""
    #technical ones
    data[['roomservice','foodcourt','shoppingmall','spa','vrdeck']]=data[['roomservice','foodcourt','shoppingmall','spa','vrdeck']].apply(np.log1p)
    
    #feature development basis 
    data['mall_avrg']= data.groupby(['deck','homeplanet'])['shoppingmall'].transform('mean')
    data['food_avrg']= data.groupby(['deck','homeplanet'])['foodcourt'].transform('mean')
    data['cnt_deckplanet']= data.groupby(['deck','homeplanet'])['homeplanet'].transform('count')
    
    # cryosleep feature (it has high mutual info)
    data['cnt_cryodeckplnt']= data.groupby(['deck','homeplanet'])['cryosleep'].transform('count')
    data['total_spend']=data[['roomservice','foodcourt','shoppingmall','spa','vrdeck']].sum(axis=1)
    data['spend_sub1']=data[['foodcourt','shoppingmall']].sum(axis=1)
    data['spend_sub2']=data[['roomservice','spa','vrdeck']].sum(axis=1)
    return data
