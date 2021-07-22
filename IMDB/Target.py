import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier as skstack
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from mlxtend.classifier import StackingClassifier

import bigml.api
from bigml.api import BigML


class Target(object):
    def __init__(self, type):
        '''accu of model is 87-88%'''
        #self.accuracy = 0.8731
        #self.model = self.load_model()
        self.model = self.train_model(type)
        self.vectorizer, self.tfidftransformer = self.load_transformer()

    def load_transformer(self):
        vector_path = "model/vectorizer.pkl"
        tfidftransformer_path = 'model/tfidftransformer.pkl'

        '''load vocabulary'''
        loaded_vectorizer = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(vector_path, "rb")))
        '''load tfidftransformer'''
        tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))

        return loaded_vectorizer, tfidftransformer

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction

    def load_model(self):
        model = joblib.load("model/target.pkl")
        print("load target model")
        return model

    def train_model(self, type):
        data_path = "data/processed_IMDB.csv"
        vector_path = "model/vectorizer.pkl"
        tfidftransformer_path = 'model/tfidftransformer.pkl'
        df = pd.read_csv(data_path)

        x_train = df['review']
        y_train = df['sentiment']
        # label_encoder = LabelEncoder()
        # y_train = label_encoder.fit_transform(y_train)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=33)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.cats = np.unique(self.y_test.tolist())
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
            clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                    subsample=.7, colsample_bytree=0.6, gamma=0.05)

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
                           ('svm', make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))),
                           ('knn', KNeighborsClassifier(n_neighbors=11)),
                           # ('dt', DecisionTreeClassifier(criterion='entropy',
                           #                               max_depth=None,min_samples_split=2,
                           #                               min_samples_leaf=1))
                           ]
            clf = skstack(estimators=base_models,
                                     final_estimator=LogisticRegression(),
                                     passthrough=True, cv=5, verbose=2, n_jobs=1)

        elif type == "Stack2":
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=200, random_state=33)),
                ('ada', AdaBoostClassifier(n_estimators=100, algorithm='SAMME', learning_rate=0.8)),
                ('knn', KNeighborsClassifier(n_neighbors=11))
            ]
            clf = skstack(estimators=base_models,
                          final_estimator=LogisticRegression(),
                          passthrough=True, cv=5, verbose=2, n_jobs=1)

        clf.fit(vectors, y_train)



        #test on testin set
        vector_test = self.vectorizer.transform(x_test)
        pred_test = clf.predict(vector_test)
        print("testing set", accuracy_score(pred_test, y_test))
        #test on training set
        pred_train = clf.predict(vectors)
        print("training set", accuracy_score(pred_train, y_train))

        joblib.dump(clf, "model/target.pkl")

        return clf
