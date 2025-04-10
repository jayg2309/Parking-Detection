import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# prepare data
input_dir = 'Enter your path for clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))

# Code for getting the accuracy based on C and gamma parameters

# Modify the parameters grid
parameters = [{'gamma': [0.01, 0.001, 0.0001], 
              'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters, cv=5, verbose=2)
grid_search.fit(x_train, y_train)

# Print detailed accuracy results
print("\nDetailed Parameter Accuracy Results:")
print("=====================================")
means = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

for mean_score, param in zip(means, params):
    print(f"\nC: {param['C']}")
    print(f"gamma: {param['gamma']}")
    print(f"Accuracy: {mean_score*100:.2f}%")

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best Cross-validation Accuracy: {grid_search.best_score_*100:.2f}%")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_prediction)*100:.2f}%")
