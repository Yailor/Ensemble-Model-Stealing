from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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
        #self.model = self.load_model()
        self.model = self.train_model(type)
        #self.cats = self.y_train['6'].unique()

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction

    def load_model(self):
        model = joblib.load("models/target.pkl")
        self.x_train = pd.read_csv("data/x_train.csv")
        self.y_train = pd.read_csv("data/y_train.csv")
        self.x_test = pd.read_csv("data/x_test.csv")
        self.y_test = pd.read_csv("data/y_test.csv")
        print("load target model")
        # test accu
        pred = model.predict(self.x_test)
        print(accuracy_score(pred, self.y_test))
        return model

    def train_model(self, type):
        df = pd.read_csv("data/GSShappiness.csv", sep=',').sample(frac=1)
        df = df.drop(columns=['id', 'year'])

        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(df)
        df = imp.transform(df)
        df = pd.DataFrame(df)

        y = df[6]
        X = df.drop(columns=[6])
        X[3] = X[3].astype(int)
        X = pd.get_dummies(X, columns=[0,1,2,4,5,7,8])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=33)
        self.cats = self.y_train.unique()
        #
        # self.x_test = x_test
        # self.y_test = y_test
        # self.x_train = x_train
        # self.y_train = y_train
        #
        # x_train.to_csv("data/x_train.csv", index=False)
        # y_train.to_csv("data/y_train.csv", index=False)
        # x_test.to_csv("data/x_test.csv", index=False)
        # y_test.to_csv("data/y_test.csv", index=False)

        # ###
        # self.x_train = pd.read_csv("data/x_train.csv")
        # self.y_train = pd.read_csv("data/y_train.csv")
        # self.x_test = pd.read_csv("data/x_test.csv")
        # self.y_test = pd.read_csv("data/y_test.csv")
        ###
        if type == "XGB":
            clf = xgb.XGBClassifier(learning_rate=0.1, subsample=0.5,
                                    colsample_bytree=0.7, gamma=0.3,
                                    max_depth=20, n_estimators=300)
        elif type == "Ada":
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=80, min_samples_split=2, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.8)
        elif type == "RF":
            clf = ensemble.RandomForestClassifier(criterion='gini', n_estimators=100,
                                                  max_depth=40, min_samples_split=2,
                                               random_state=100)
        elif type == "Stack1":
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

        clf.fit(self.x_train, self.y_train)

        pred_train = clf.predict(self.x_train).tolist()
        print("accu on training set", accuracy_score(pred_train, self.y_train))
        pred = clf.predict(self.x_test).tolist()
        print("accu on testing set", accuracy_score(self.y_test, pred))
        joblib.dump(clf, "models/target.pkl")
        return clf