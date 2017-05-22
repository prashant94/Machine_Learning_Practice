# Titanic Dataset
# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

style.use('ggplot')

# Dataset Description
'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numeric_data(df):
    cols = df.columns.values

    for col in cols:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            col_content = df[col].values.tolist()
            unique_elements = set(col_content)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[col] = list(map(convert_to_int, df[col]))

    return df

df = handle_non_numeric_data(df)

# Apparently ticket numbers matter
# df.drop(['ticket'], 1, inplace=True)

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    prediction = np.array(X[i].astype(float))
    prediction = prediction.reshape(-1,len(prediction))
    new_predict = clf.predict(prediction)
    if new_predict[0] == y[i]:
        correct += 1

# Accuracy may be x or 100-x depending on label assigned by clusters
print(correct/len(X))
    





