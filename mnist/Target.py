from sklearn import decomposition, ensemble
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_digits
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier as skstack
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier


class Target(object):
    def __init__(self, type):
        #self.model = self.load_model()
        self.accuracy = 0.97
        self.model = self.train_model(type)

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction

    def load_model(self):
        model = joblib.load("target.pkl")
        print("load target model")
        return model

    def train_model(self, type):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        #reshape X
        n_train_samples, nx_tr, ny_tr = X_train.shape
        X_train = X_train.reshape((n_train_samples, nx_tr * ny_tr))
        n_test_samples, nx_te, ny_te = X_test.shape
        X_test = X_test.reshape((n_test_samples, nx_te * ny_te))

        if type == "XGB":
            clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                      subsample=.7, colsample_bytree=0.6, gamma=0.05, use_label_encoder=False)
        elif type == "Ada":
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=60, min_samples_split=5, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=100, learning_rate=0.8)
        elif type == "RF":
            clf = ensemble.RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=40, min_samples_split=15,
                                                  random_state=100)
        elif type == "Stack1":
            #clf = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=150, earning_rate=0.1)
            base_models = [('dt', DecisionTreeClassifier(criterion='entropy')),
                           ('svm', SVC()),
                           ('knn', KNeighborsClassifier(n_neighbors=11))]
            clf = skstack(estimators= base_models,
                                final_estimator=LogisticRegression(),
                                passthrough=True, cv=5, verbose=2, n_jobs=1)
        elif type == "Stack2":
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=33)),
                ('ada', AdaBoostClassifier(n_estimators=100, algorithm='SAMME', learning_rate=0.8)),
                ('knn', KNeighborsClassifier(n_neighbors=11))
            ]
            clf = skstack(estimators= base_models,
                                final_estimator=LogisticRegression(),
                                passthrough=True, cv=5, verbose=2, n_jobs=1)
        clf.fit(X_train, y_train)
        #elif type == "LGB":
        #test on testin set
        pred_test = clf.predict(X_test)
        self.accuracy = accuracy_score(pred_test, y_test)
        print("testing set", self.accuracy)
        #test on training set
        pred_train = clf.predict(X_train)
        print("training set", accuracy_score(pred_train, y_train))

        joblib.dump(clf, "target.pkl")

        return clf