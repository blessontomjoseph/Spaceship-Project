import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plot

# train_path='/kaggle/input/spaceship-titanic/train.csv'
# test_path='/kaggle/input/spaceship-titanic/test.csv'

# train=pd.read_csv(train_path)
# test=pd.read_csv(test_path)

# num=train.select_dtypes('number')
# plot.plotter(train,num,(10,10))

a=np.array([1,2,3,4])
b=np.array([2,3,4,5])
plt.scatter(a,b)
plt.show()
