import pandas as pd
import numpy as np
from IPython.core.display_functions import display

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, \
    recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

bank_data = pd.read_csv('/Users/idilsevinc/Downloads/bank+marketing/bank'
                        '-additional/bank-additional.csv', sep=';')
bank_data = bank_data.loc[:,
            ['age', 'default', 'cons.price.idx', 'cons.conf.idx', 'y']]
bank_data.head(5)

bank_data['default'] = bank_data['default'].map(
    {'no': 0, 'yes': 1, 'unknown': 0})
bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})

X = bank_data.drop('y', axis=1)
Y = bank_data['y']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print('Accuracy: ', accuracy)

for i in range(1):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree, feature_names=X_train.columns, filled=True,
                               max_depth=2, impurity=False, proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)
