from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.preprocessing import StandardScaler



def mutual_information(x,y,mask=None):
    """function calculates and plots the mi score given x and y"""
    if mask is not None:
        mi=mutual_info_classif(x.iloc[:,:mask],y)
        mi=pd.DataFrame(mi,columns=['mi_score'],index=x.columns[:mask])
    elif mask is None:  
        mi=mutual_info_classif(x,y)
        mi=pd.DataFrame(mi,columns=['mi_score'],index=x.columns)
        
    mi=mi.sort_values("mi_score",ascending=False)
    return mi

    
def some_ok_mi(x,y,num):
    mi=mutual_info_classif(x,y)
    mi=pd.DataFrame(mi,columns=['mi_score'],index=x.columns)    
    mi=mi.sort_values("mi_score",ascending=False)
    return mi.iloc[:num,:]


def pca_ing(x,y,standardize=None):
    """function standardizes the data is not standardized and preforms pca and outputs its componets in a df"""
    if standardize:
        sc=StandardScaler()
        x_scaled=sc.fit_transform(x)
        x=pd.DataFrame(x_scaled,columns=x.columns)
    pca=PCA()
    x_pca=pca.fit_transform(x,y)
    components=[f'pca_{i}' for i in x.columns.values]
    x_pca=pd.DataFrame(x_pca,columns=components)
    loadings=pd.DataFrame(pca.components_.T,columns=components,index=x.columns)
    return x_pca,loadings