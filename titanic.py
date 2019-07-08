import numpy as np
import pandas as pd

data=pd.read_csv('/home/ravi/Documents/Titanic/gender_submission.csv',dtype=str)
print data

datan=data.dropna(axis='rows')
print data
datat=pd.read_csv('/home/ravi/Documents/Titanic/train.csv',dtype=str)
print datat
datatp=datat.dropna(axis='columns')
print datatp
