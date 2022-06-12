from preprocessing import *
from utils import *
from features import *
from model import *
from optimization import *


train_path = '/kaggle/input/spaceship-titanic/train.csv'
test_path = '/kaggle/input/spaceship-titanic/test.csv'
train, test, val = data_load(train_path, test_path)

train = column_transformer1(train)
val = column_transformer1(val)
test = column_transformer1(test)

categorical = train.select_dtypes('object', 'category')
bool_columns = train.select_dtypes('bool')
categorical = categorical.join(bool_columns).columns
numerical = train.select_dtypes('number').columns

train_copy = train.copy()
val_copy = val.copy()
test_copy = test.copy()

train = transformation(train)
val = transformation(val)
test = transformation(test)

train_good, train_transformable, train_non_transformable = impute_split(train)
val_good, val_transformable, val_non_transformable = impute_split(val)
test_good, test_transformable, test_non_transformable = impute_split(test)

# make knn
train_imputer = Knn_imputation(
    train_good, train_transformable, train_non_transformable, categorical, numerical)
train = train_imputer.knn_implement()
val_imputer = Knn_imputation(
    val_good, val_transformable, val_non_transformable, categorical, numerical)
val = val_imputer.knn_implement()
test_imputer = Knn_imputation(
    test_good, test_transformable, test_non_transformable, categorical, numerical)
test = test_imputer.knn_implement()

train, test, val = simple_im(train, test, val, categorical, numerical)

trainx = train.drop(['transported'], axis=1)
trainy = train['transported'].astype(int)
valx = val.drop(['transported'], axis=1)
valy = val['transported'].astype(int)

train_score, val_score,test_preds = results(trainx, trainy, valx, valy,test, model_rf)
print('train score:', train_score)
print('val score:', val_score)
# fetch_submission(test_preds)



# train_x, other = auto_best_features(trainx, trainy,[valx, test], n_features=15, standardize_on_pca=True)
# valx, test = other[0], other[1]


best_params = bayesian_search(trainx, trainy, valx, valy, model)
# model_rf['model'].set_params(best_params)
# train_score, val_score = model_rf(trainx, trainy, valx, valy, model)
# print('train score:', train_score)
# print('val score:', val_score)


print(best_params)
