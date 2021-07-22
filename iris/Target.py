from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Target(object):
    def __init__(self, type):
        #self.model = self.load_model()
        self.model = self.train_model(type)

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction

    def load_model(self):
        model = joblib.load("model/target.pkl")
        print("load target model")
        return model

    def train_model(self, type):
        #df = pd.read_csv("data/adult.csv", sep=',').sample(frac=1)
        df = load_iris()

        data = pd.DataFrame({
            'sepal length': df.data[:, 0],
            'sepal width': df.data[:, 1],
            'petal length': df.data[:, 2],
            'petal width': df.data[:, 3],
            'species': df.target
        })

        X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
        y = data['species']  # Labels


        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.train = X_train
        self.train_lb = y_train
        self.test = X_test
        self.test_lb = y_test

        # clf = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=10, min_samples_split=2,
        #                                       random_state=100)
        if type == "XGB":
            clf = xgb.XGBClassifier(learning_rate=0.1, subsample=0.5,
                                    colsample_bytree=0.7, gamma=0.3,
                                    max_depth=10, n_estimators=20)
        elif type == "RF":
            clf = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=100,
                                                  max_depth=10, min_samples_split=2,
                                                  random_state=100)
        elif type == "Stacking":
            base_models = [('random_forest', RandomForestClassifier(n_estimators=100)),
                           ('svm', make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))),
                           ('knn', KNeighborsClassifier(n_neighbors=11)),
                           # ('dt', DecisionTreeClassifier(criterion='entropy',
                           #                               max_depth=None,min_samples_split=2,
                           #                               min_samples_leaf=1))
                           ]
            clf = StackingClassifier(estimators=base_models,
                                     final_estimator=LogisticRegression(),
                                     passthrough=True, cv=5, verbose=2, n_jobs=1)

        else: print("error type!")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))

        return clf