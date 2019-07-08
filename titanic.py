import numpy as np
import pandas as pd

data=pd.read_csv('/home/ravi/Documents/Titanic/gender_submission.csv',dtype=str)
print data

data.dropna(axis='rows')
print data
