import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb


class Target(object):
    def __init__(self):
        '''accu of target model is around 0.855'''
        #self.model = self.load_model()
        self.model = self.train_model()

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction

    def load_model(self):
        model = joblib.load("model/target.pkl")
        print("load target model")
        return model

    def train_model(self):
        df = pd.read_csv("data/steak-survey.csv", sep=',').sample(frac=1)
        df = df[(df.astype(str) != ' ?').all(axis=1)]
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(df)
        df = imp.transform(df)

        # transform back to dataframe
        df = pd.DataFrame(df)
        df = df.drop(columns=[0])

        # Use one-hot encoding on categorial columns
        df = pd.get_dummies(df, columns=[1,2,3,4,5,6,7,8,10,11,12,13,14])
        y = df[9]

        X = df.drop(columns=[9])

        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2, random_state=33)

        self.train = x_train

        self.x_test = x_test
        self.y_test = y_test

        x_train.to_csv("data/x_train.csv")
        y_train.to_csv("data/y_train.csv")

        clf = xgb.XGBClassifier(learning_rate=0.1, subsample=.7,
                                colsample_bytree=0.6, gamma=0.05, n_estimators=100)
        clf.fit(x_train, y_train)

        pred = clf.predict(x_test).tolist()
        print("tar acc", accuracy_score(y_test, pred))
        joblib.dump(clf, "models/target.pkl")
        return clf


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer


class Target(object):
    def __init__(self, type):
        '''accu of target model is around 0.54'''
        #self.model = self.load_model()
        self.model = self.train_model(type)
        self.cats = self.y_train['9'].unique()

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
        df = pd.read_csv("data/steak-survey.csv", sep=',').sample(frac=1)
        df = df[(df.astype(str) != ' ?').all(axis=1)]
        #df.dropna(how='any', axis=0, inplace=True)
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(df)
        df = imp.transform(df)

        # transform back to dataframe
        df = pd.DataFrame(df)
        df = df.drop(columns=[0])

        # Use one-hot encoding on categorial columns
        df = pd.get_dummies(df, columns=[1,2,3,4,5,6,7,8,10,11,12,13,14])
        y = df[9]

        X = df.drop(columns=[9])

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        #
        x_train.to_csv("data/x_train.csv", index=False)
        y_train.to_csv("data/y_train.csv", index=False)
        x_test.to_csv("data/x_test.csv", index=False)
        y_test.to_csv("data/y_test.csv", index=False)
        self.x_train = pd.read_csv("data/x_train.csv")
        self.y_train = pd.read_csv("data/y_train.csv")

        ###
        # self.x_train = pd.read_csv("data/x_train.csv")
        # self.y_train = pd.read_csv("data/y_train.csv")
        # self.x_test = pd.read_csv("data/x_test.csv")
        # self.y_test = pd.read_csv("data/y_test.csv")
        ###
        if type == "XGB":
            clf = xgb.XGBClassifier(learning_rate=0.1, subsample=0.5,
                                    colsample_bytree=0.7, gamma=0.3,
                                    max_depth=20, n_estimators=100)
        elif type == "RF":
            clf = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=1000, min_samples_split=10,
                                                  random_state=100, oob_score=True)
        elif type == "LGB":
            clf = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=150, earning_rate=0.1)
            clf.fit(self.x_train.squeeze, self.y_train)
        else: print("error type")
        clf.fit(self.x_train, self.y_train)

        pred_train = clf.predict(self.x_train).tolist()
        print("accu on training set", accuracy_score(pred_train, self.y_train))
        pred = clf.predict(self.x_test).tolist()
        print("accu on testing set", accuracy_score(self.y_test, pred))
        joblib.dump(clf, "models/target.pkl")
        return clf