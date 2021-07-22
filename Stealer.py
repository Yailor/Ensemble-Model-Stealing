import math
from collections import Counter
from itertools import combinations
#from sensor.rf import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import scipy.sparse as scip
import copy
import random
import numpy as np
import sympy as sp
from sklearn.datasets import fetch_20newsgroups

BASE = 10
c0 = c1 = c2 = 0.5


class Stealer():
    def __init__(self, target, pro):
        super(Stealer, self).__init__()
        self.models = []
        self.proportion = pro
        self.data = self.data_augmentation(target)
        #self.load_augmentation_data(target)
        self.row = 0
        self.queries = 0

    def load_augmentation_data(self, target):
        self.train_aug = pd.read_csv("data/augmented data.csv")
        self.train_aug_y = pd.read_csv("data/augmented data y.csv")
        self.test = target.x_test
        self.data = pd.read_csv("data/steak_augmentation_data.csv")


    def data_augmentation(self, target):
        train = target.x_train
        num = round(self.proportion/100 * len(train))
        x_train, self.train_aug, y_train, self.y_train_aug = train_test_split(target.x_train, target.y_train,
                                                                              test_size=num,
                                                                              random_state=33)
        # save augmented data
        self.train_aug.to_csv("data/augmented data.csv", index=False)
        self.y_train_aug.to_csv("data/augmented data y.csv", index=False)
        self.test = target.x_test
        '''augmentation procedure'''
        '''1.get all types for each column'''
        feature_list = []

        for col in self.train_aug.iteritems():
            vals = np.sort(col[1].unique())
            if len(vals) >= 2:
                mean = np.mean(vals)
                left = mean - vals[0]
                right = vals[len(vals)-1] - mean
                miu = min(left, right)
                feature_list.append([mean, miu])
            else:
                feature_list.append([0])

        sample_pool = pd.DataFrame(columns=x_train.columns)
        '''2.Gaussian sampling'''
        for i in range(5000):
            sample = []
            for fea in feature_list:
                if len(fea) == 2:
                    val = round(random.gauss(fea[0], fea[1]))
                    if val < 0: val = 0
                else:
                    val = 0
                sample.append(val)
            sample_pool.loc[i] = sample

        '''save as csv'''
        sample_pool.to_csv("data/steak_augmentation_data.csv", index=False)

        return sample_pool



    def flip_biased(self, p):
        return random.random() < p

    def forest_predict(self, data, rows):
        lbs = np.ones((len(self.models), rows))
        lbs = lbs.astype(np.str)
        i = 0
        for model in self.models:
            lb = model.predict(data)
            lbs[i, 0:rows] = lb
            i += 1
        max_lbs = []
        for j in range(rows):
            col = lbs[:, j]
            col = col.tolist()
            max_lb = Counter(col).most_common(1)[0][0]
            max_lbs.append(max_lb)
        max_lbs = list(max_lbs)
        return max_lbs


    def active_learning(self, clf, random_pool, target, err_set, tar_lb):
        err_set = err_set.reset_index(drop=True)
        flag_update = 0
        i = 1
        s_cols = self.cols_name.copy()
        s_cols.append('prob')
        S = pd.DataFrame(columns=s_cols)
        random_pool = random_pool.reset_index(drop=True)
        while i <= len(random_pool):
            xi = random_pool.iloc[i-1]
            xi_list = xi.tolist()
            #yi = clf.predict(xi_vec) #clf predict yi
            # S is a set of random samples in pool
            hn = self.get_hn_error(i, S, clf)
            hnn = self.get_hnn_error(i, S, xi_list, target, clf)      #modify
            if i == 1:
                u = float('inf')
            else:
                u = math.sqrt(c0 * math.log(i, BASE) / (i - 1)) + c0 * math.log(i, BASE) / (i - 1)
            gn = hnn - hn

            s = 1
            if i > 1:
                s = sp.symbols('s')
                sqrt1 = math.sqrt(c0 * math.log(i, BASE) / (i - 1))
                ss2 = c0 * math.log(i, BASE) / (i - 1)
                eq = gn - (c1 / (s - c1 + 1) * sqrt1 + c2 / (s - c2 + 1) * ss2)
                s = sp.solve(eq, s)

                if not len(s):
                    s = 0
                elif len(s):
                    s = max(s)
                    if s < 0:
                        s = 0
                s = math.sqrt(s)

            if gn < u:
                # add this sample to S
                p = 1
                flag_update = 1
                '''xgb cannot predict single sample'''
                y = target.model.predict(random_pool)[i-1]
                self.queries += 1
                dt = xi_list.copy()
                dt.append(y)
                dt.append(p)
                S.loc[len(S)] = dt

                err_set.loc[len(err_set)] = xi_list
                tar_lb.append(y)

            else:
                p = s
                toss = self.flip_biased(p)
                if toss:
                    flag_update = 1
                    y = target.model.predict(random_pool)[i-1]
                    self.queries += 1
                    dt = xi_list
                    dt.append(y)
                    dt.append(p)
                    S.loc[len(S)] = dt
                    err_set.loc[len(err_set)] = xi_list
                    tar_lb.append(y)

            if flag_update:
                clf = self.update_model(clf, err_set, tar_lb)

            #print("{} times of clf, gn={}, lenS={}, u={}, hn={}, hnn={}".format(i, round(gn, 4), len(S), round(u, 4), round(hn, 4), round(hnn, 4)))
            i += 1
        print("clf train end, lenS={}".format(len(S)))
        return clf, err_set, tar_lb


    def get_hn_error(self, n, S, clf):
        sum = 0
        label = S['target']
        prob = S['prob']
        data = S.drop(['target', 'prob'], axis=1)
        for i in range(len(data)):
            pt = data.iloc[i].tolist()
            lb = clf.predict(np.reshape(pt, (1, -1)))[0]
            if lb != label[i]:
                sum += 1 / prob[i]
        sum = (1 / n) * sum
        return sum

    def change_label(self, S_lb):
        if S_lb == 1:
            return 0
        else: return 1

    def flip_predict(self, X, j):
        lb, lbs = self.model.predict_single(X)
        for i in range(len(lbs)):
            if i in j:
                lbs[i] = self.change_label(lb)

        result = Counter(lbs)
        if result['1'] >= self.model.n_estimators/2: return '1', lbs
        else: return '2', lbs


    def get_hnn_error(self, n, S, xi, target, clf):
        sum_list = []
        for cat in target.cats:
            sum = 0
            if len(S):
                # get leaf_id of this new sample, if other samples in S have the same id, change target label
                leaf_id = clf.apply(np.reshape(xi, (1, -1)))

                label = S['target']
                prob = S['prob']
                data = S.drop(['target', 'prob'], axis=1)
                for i in range(len(S)):
                    pt = data.iloc[i].tolist()
                    S_leaf_id = clf.apply(np.reshape(pt, (1, -1)))[0]
                    if S_leaf_id == leaf_id:
                        clf_pred_pt = cat
                        if clf_pred_pt != label[i]:
                            sum += 1 / prob[i]
                sum = (1 / n) * sum
                sum_list.append(sum)
            else:
                return 0
        # print(sum_list)
        return max(sum_list)


    def update_model(self, model, err_set, tar_lb):
        model.fit(err_set, tar_lb)
        return model

    def train_base_clf(self, data, label):
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=40, random_state=60)
        tree.fit(data, label)

        return tree
