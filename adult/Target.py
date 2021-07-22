from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
#rimport lightgbm as lgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


class Target(object):
    def __init__(self, type):
        '''accu of target model is around 86.7'''
        #self.model = self.load_model()
        self.model = self.train_model(type)

    def predict(self, X):
        prediction = map(int, self.model.predict(X))
        return prediction

    def load_model(self):
        model = joblib.load("model/target.pkl")
        print("load target model")
        return model

    def train_model(self, type):
        df = pd.read_csv("data/adult.csv", sep=',').sample(frac=1)
        df = df[(df.astype(str) != ' ?').all(axis=1)]

        # Create a new income_bi column
        #df['income_bi'] = df.apply(lambda row: 1 if '>50K' in row['income'] else 0, axis=1)
        # Remove redundant columns
        #df = df.drop(['income', 'fnlwgt', 'capital.gain', 'capital.loss', 'native.country'], axis=1)
        #df = df.drop(['income'], axis=1)
        # Use one-hot encoding on categorial columns
        df = pd.get_dummies(df, columns=['workclass', 'education', 'marital.status',
                                         'occupation', 'relationship', 'race', 'sex', 'native.country'])
        d_train = df[:15000]
        d_test = df[15000:]
        d_train_att = d_train.drop(['income'], axis=1)
        self.train_x = d_train_att
        self.train = d_train
        d_train_gt50 = d_train['income']
        self.test = d_test
        d_test_att = d_test.drop(['income'], axis=1)
        d_test_gt50 = d_test['income']
        d_att = df.drop(['income'], axis=1)
        d_gt50 = df['income']

        # clf = ensemble.RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=40, min_samples_split=2,
        #                                       random_state=100)
        # clf.fit(d_train_att, d_train_gt50)

        # clf = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1,
        #                         subsample=.7, colsample_bytree=0.6, gamma=0.05)
        if type == "XGB":
            clf = xgb.XGBClassifier(learning_rate=0.1, subsample=0.5,
                                    colsample_bytree=0.7, gamma=0.05,
                                    max_depth=10, n_estimators=30)
        elif type == "Ada":
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=60, min_samples_split=10, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=100, learning_rate=0.8)
        elif type == "RF":
            clf = ensemble.RandomForestClassifier(criterion='gini', n_estimators=100,
                                                  max_depth=40, min_samples_split=2,
                                               random_state=100)
        elif type == "Stacking1":
            base_models = [('dt', DecisionTreeClassifier(criterion='entropy')),
                           ('svm', make_pipeline(StandardScaler(), SVC(gamma='auto'))),
                           ('knn', KNeighborsClassifier(n_neighbors=11)),
                           # ('dt', DecisionTreeClassifier(criterion='entropy',
                           #                               max_depth=None,min_samples_split=2,
                           #                               min_samples_leaf=1))
                           ]
            clf = StackingClassifier(estimators=base_models,
                                     final_estimator=LogisticRegression(),
                                     passthrough=True, cv=5, verbose=2, n_jobs=1)
        elif type == "Stack2":
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=33)),
                ('ada', AdaBoostClassifier(n_estimators=100, algorithm='SAMME', learning_rate=0.8)),
                ('knn', KNeighborsClassifier(n_neighbors=11))
            ]
            clf = StackingClassifier(estimators= base_models,
                                final_estimator=LogisticRegression(),
                                passthrough=True, cv=5, verbose=2, n_jobs=1)

        else: print("error type")
        clf.fit(d_train_att, d_train_gt50)
        #
        # xgb_test = xgb.DMatrix(d_test_att, d_test_gt50)
        pred = clf.predict(d_test_att)
        # for i in range(len(pred)):
        #     if pred[i] > 0.5:
        #         pred[i] = 1
        #     else:
        #         pred[i] = 0
        print(accuracy_score(pred, d_test_gt50))

        return clf

    def predict(self, X):
        pred = self.model.predict(X)
        # for i in range(len(pred)):
        #     if pred[i] > 0.5:
        #         pred[i] = 1
        #     else:
        #         pred[i] = 0
        return pred.tolist()