import pickle

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import xgboost as xgb
import lightgbm as lgb



class Target(object):
    def __init__(self, type):
        self.accuracy = 0.97
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
        data_path = "data/spam_aug_pool/spam.csv"
        vector_path = "vectorizer.pkl"
        tfidftransformer_path = 'tfidftransformer.pkl'
        df = pd.read_csv(data_path).sample(frac=1)

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
            clf = xgb.XGBClassifier(learning_rate=0.1, subsample=0.5,
                                    colsample_bytree=0.7, gamma=0.05,
                                    max_depth=10, n_estimators=30)
        elif type == "RF":
            clf = RandomForestClassifier(criterion='gini', n_estimators=100,
                                                  max_depth=35, min_samples_split=15,
                                                  random_state=100)
        elif type == "LGB":
            clf = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=50, earning_rate=0.1)
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
