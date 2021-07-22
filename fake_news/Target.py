import pickle

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class Target(object):
    def __init__(self, type):
        self.accuracy = 0.99
        #self.model = self.load_model()
        self.model = self.train_model(type)

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction

    def load_model(self):
        model = joblib.load("target.pkl")
        print("load target model")
        vector_path = "vectorizer.pkl"
        tfidftransformer_path = 'tfidftransformer.pkl'
        # load vocabulary
        self.vectorizer = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(vector_path, "rb")))
        # load tfidftransformer
        self.transformer = pickle.load(open(tfidftransformer_path, "rb"))
        return model


    def train_model(self, type):
        '''df = pd.read_csv("data/processed_twitter.csv", encoding='latin-1').dropna(axis=0, how='any')
        df = df.sample(frac=1)
        df.to_csv("data/processed_twitter.csv")'''
        data_path = "data/fake_aug_pool/processed_news.csv"
        vector_path = "vectorizer.pkl"
        tfidftransformer_path = 'tfidftransformer.pkl'
        df = pd.read_csv(data_path).dropna(how='any')
        x_train = df['text']
        y_train = df['target']
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=33)

        self.vectorizer = CountVectorizer(decode_error='replace')
        self.tfidftransformer = TfidfTransformer()

        vectors = self.vectorizer.fit_transform(x_train)
        tfidf = self.tfidftransformer.fit_transform(vectors)

        # save vectorizer
        with open(vector_path, 'wb') as fw:
            pickle.dump(self.vectorizer.vocabulary_, fw)
        with open(tfidftransformer_path, 'wb') as fw:
            pickle.dump(self.tfidftransformer, fw)

        # # load vocabulary
        # loaded_vectorizer = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(vector_path, "rb")))
        # # load tfidftransformer
        # tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))
        # # test loaded transformer
        # test_tfidf = tfidftransformer.transform(loaded_vectorizer.transform(x_test))

        # clf = ensemble.RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=35, min_samples_split=15,
        #                                       random_state=100)
        if type == "XGB":
            clf = xgb.XGBClassifier(n_estimators=70, max_depth=6, learning_rate=0.1,
                                    subsample=.7, colsample_bytree=0.6, gamma=0.05)
        elif type == "Ada":
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=60, min_samples_split=10, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=100, learning_rate=0.8)
        elif type == "RF":
            clf = RandomForestClassifier(criterion='gini', n_estimators=100,
                                                  max_depth=35, min_samples_split=15,
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

        clf.fit(vectors, y_train)
        # test on testin set
        vector_test = self.vectorizer.transform(x_test)
        pred_test = clf.predict(vector_test)
        self.accuracy = accuracy_score(pred_test, y_test)
        print("testing set", self.accuracy)
        # test on training set
        pred_train = clf.predict(vectors)
        print("training set", accuracy_score(pred_train, y_train))

        joblib.dump(clf, "target.pkl")

        return clf