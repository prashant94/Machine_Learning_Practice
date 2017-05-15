# KNN classifier built from scratch tested against sklearn version
# Using UCI dataset

import numpy as np
import warnings
from collections import Counter
from math import sqrt
import pandas as pd
import random
from sklearn import preprocessing, model_selection, neighbors

def euclidean_distance(feature, predict):
    total = 0
    for i in list(zip(feature, predict)):
        total += (i[0]-i[1])**2
    return sqrt(total)

def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to value less than total voting groups')

    distances = []
    for group in  data:
        for features in data[group]:
            euc_dist = euclidean_distance(features,predict)
            distances.append([euc_dist, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
            
    return vote_result, confidence


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

accuracies_scratch = []
accuracies_sklearn = []

for test_no in range(25):
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
            
    scratch_accuracy = correct / total

    X = np.array(df.drop(['class'],1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    sk_learn_accuracy = clf.score(X_test, y_test)
    
    print('Accuracy of test no {}: Scratch -> {}, Sklearn -> {}'.format(test_no, scratch_accuracy,sk_learn_accuracy))
    accuracies_scratch.append(scratch_accuracy)
    accuracies_sklearn.append(sk_learn_accuracy)

print('Final Scratch Accuracy: ',sum(accuracies_scratch) / len(accuracies_scratch))
print('Final Sklearn Accuracy: ',sum(accuracies_sklearn) / len(accuracies_sklearn))       
