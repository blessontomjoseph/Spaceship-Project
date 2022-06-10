from preprocessing import *
from utils import *
from features import *
from model import *


train_path='/kaggle/input/spaceship-titanic/train.csv'
test_path='/kaggle/input/spaceship-titanic/test.csv'
train,test,val=data_load(train_path,test_path)


train=column_transformer1(train)
val=column_transformer1(val)
test=column_transformer1(test)


train=transformation(train)
val=transformation(val)
test=transformation(test)


categorical=train.select_dtypes('object','category')
bool_columns=train.select_dtypes('bool')
categorical=categorical.join(bool_columns).columns
numerical=train.select_dtypes('number').columns

train_copy=train.copy()
val_copy=val.copy()
test_copy=test.copy()



train_good,train_transformable,train_non_transformable = impute_split(train)
val_good,val_transformable,val_non_transformable = impute_split(val)
test_good,test_transformable,test_non_transformable = impute_split(test)


# make knn

train_imputer=Knn_imputation(train_good,train_transformable,train_non_transformable,categorical,numerical)
train=train_imputer.knn_implement()
val_imputer=Knn_imputation(val_good,val_transformable,val_non_transformable,categorical,numerical)
val=train_imputer.knn_implement()
test_imputer=Knn_imputation(test_good,test_transformable,test_non_transformable,categorical,numerical)
test=train_imputer.knn_implement()

# train,test,val=simple_im(train,test,val,categorical,numerical)

# train=reverse_transformation(train_copy,train)
# val=reverse_transformation(val_copy,val)
# test=reverse_transformation(test_copy,test)

# train=name_splitter(train)
# val=name_splitter(val)
# test=name_splitter(test)

# train=name_impute(train,train)
# val=name_impute(train,val)
# test=name_impute(train,test)

# train,test,val=final_col_trans(train,test,val,categorical,numerical)

# trainx=train.drop(['transported'],axis=1)
# trainy=train['transported'].astype(int)
# valx=val.drop(['transported'],axis=1)
# valy=val['transported'].astype(int)

print(train)
print(train.columns)
print(train.isna().sum().sum())
