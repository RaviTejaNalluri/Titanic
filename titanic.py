import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

data=pd.read_csv('/home/ravi/Documents/Titanic/train.csv')
#print data.head()



#print data.head()

#calculating age where data is misssing.Taking average age wrt pclass
def calc_age(columns):
    Age=columns[0]
    Pclass=columns[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37#37,29,24 are from graph
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

#label encoding to give nummerical values to gender snfd embarked
'''
le=preprocessing.LabelEncoder()
le.fit(data['Sex'])
data['Sex']=le.transform(data['Sex'])
data=data.dropna()
le.fit(data['Embarked'])
data['Embarked']=le.transform(data['Embarked'])
print data.head(7)

train_X,test_X, train_Y, test_Y = train_test_split(data.drop('Survived',axis=1),
                                                    data['Survived'], test_size=0.80,
                                                    random_state=150)



model = LogisticRegression()
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print 'The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y)
'''

data=pd.read_csv('/home/ravi/Documents/Titanic/train.csv')


data.drop('Name',axis=1,inplace=True)
data.drop('Ticket',axis=1,inplace=True)
data['Age'] = data[['Age','Pclass']].apply(calc_age,axis=1)
data.drop('Cabin',axis=1,inplace=True)
data = data.dropna()



cumsum = data.groupby('Sex')['Survived'].cumsum() - data['Survived']
cumcnt = data.groupby('Sex').cumcount()
data['Sex'] = cumsum/cumcnt
data['Sex'].fillna(0.3343, inplace = True)

cumsum = data.groupby('Embarked')['Survived'].cumsum() - data['Survived']
cumcnt = data.groupby('Embarked').cumcount()
data['Embarked'] = cumsum/cumcnt
data['Embarked'].fillna(0.3343, inplace = True)


print data.head()

from sklearn.linear_model import LogisticRegressionCV
LogisticRegressionCV(Cs=10)
from sklearn.model_selection import train_test_split
train_X,test_X, train_Y, test_Y = train_test_split(data.drop('Survived',axis=1),
                                                    data['Survived'], test_size=0.30,
                                                    random_state=101)
from sklearn import metrics
model = LogisticRegressionCV()
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))
