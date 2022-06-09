import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_path='/kaggle/input/spaceship-titanic/train.csv'
test_path='/kaggle/input/spaceship-titanic/test.csv'

train=pd.read_csv(train_path)
test=pd.read_csv(test_path)

print(train.head(5))