from preprocessing import *
from utils import *

train_path='/kaggle/input/spaceship-titanic/train.csv'
test_path='/kaggle/input/spaceship-titanic/test.csv'
train,test,val=data_load(train_path,test_path)

train=transformation(train)
# val=transformation(val)
# test=transformation(test)

train=column_transformer1(train)
# val=column_transformer1(val)
# test=column_transformer1(test)

train_copy=train.copy()
# val_copy=val.copy()
# test_copy=test.copy()


train_good,train_transformable,train_non_transformable = impute_split(train)
# val_good,val_transformable,val_non_transformable = impute_split(val)
# test_good,test_transformable,test_non_transformable = impute_split(test)

