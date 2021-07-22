import math
from collections import Counter
from itertools import combinations
#from sensor.rf import RandomForestClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import scipy.sparse as scip

import random
import numpy as np
import sympy as sp
from sklearn.datasets import fetch_20newsgroups

BASE = 10
c0 = c1 = c2 = 0.5


class Stealer():
    def __init__(self, target, pro):
        super(Stealer, self).__init__()
        self.proportion = pro
        self.models = []
        # dataframe, 1 column
        self.data = self.load_pool()
        self.row = 0
        self.queries = 0


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
        return max_lbs

    def add_elm(self, err_set, tar_lb, elm):
        err_set.append(elm['text'])
        tar_lb.append(elm['target'])
        return err_set, tar_lb

    def active_learning(self, clf, random_pool, target, err_set, tar_lb):
        flag_update = 0
        i = 1
        S = pd.DataFrame(columns=['text', 'target', 'prob'])
        random_pool = random_pool.reset_index(drop=True)
        while i <= len(random_pool):
            xi = random_pool[i-1]
            xi_vec = target.vectorizer.transform([xi])
            #yi = clf.predict(xi_vec) #clf predict yi
            # S is a set of random samples in pool
            hn = self.get_hn_error(i, S, target, clf)
            hnn = self.get_hnn_error(i, S, xi_vec, target, clf)      #modify
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
                y = target.model.predict(xi_vec)[0]
                self.queries += 1
                '''add sample to S'''
                elm = {'text': xi, 'target': y, 'prob': p}
                S = S.append(elm, ignore_index=True)

            else:
                p = s
                toss = self.flip_biased(p)
                if toss:
                    flag_update = 1
                    y = target.model.predict(xi_vec)[0]
                    self.queries += 1
                    '''add sample to S'''
                    elm = {'text': xi, 'target': y, 'prob': p}
                    S = S.append(elm, ignore_index=True)
            if flag_update:
                err_set, tar_lb = self.add_elm(err_set, tar_lb, elm)
                clf = self.update_model(clf, err_set, tar_lb, target.vectorizer)

            #print("{} times of clf, gn={}, lenS={}, u={}, hn={}, hnn={}".format(i, round(gn, 4), len(S), round(u, 4), round(hn, 4), round(hnn, 4)))
            i += 1
        '''if flag_update:
            clf = self.update_model(clf, err_set, tar_lb, S, target.vectorizer)'''
        print("clf train end, lenS={}".format(len(S)))
        self.models.append(clf)
        return clf, err_set, tar_lb

    def load_pool(self):
        dd = pd.read_csv("data/spam_aug_pool/spam-{}%/spam-{}%.csv"
                         .format(self.proportion, self.proportion))
        l = int(len(dd)*2/3)
        data = dd.head(l)
        for i in range(40):
            data = data.append(pd.read_csv("data/spam_aug_pool/spam-{}%/spam_aug_pool{}.csv"
                                           .format(self.proportion, i)).head(l))
        data = data['text']
        #add some real samples
        data = data.sample(frac=1).reset_index(drop=True)
        return data

    def get_hn_error(self, n, S, target, clf):
        sum = 0
        if len(S):
            data = target.vectorizer.transform(S['text'])
            lbs = clf.predict(data)
            tar = S['target']
            prob = S['prob']
            for i in range(len(S)):
                if lbs[i] != tar[i]:
                    sum += 1 / prob[i]
            sum = (1 / n) * sum
        return sum

    def flip_predict(self, X, j):
        lb, lbs = self.model.predict_single(X)
        for i in range(len(lbs)):
            if i in j:
                lbs[i] = self.change_label(lb)

        result = Counter(lbs)
        if result['1'] >= self.model.n_estimators/2: return '1', lbs
        else: return '2', lbs


    def get_hnn_error(self, n, S, xi_vec, target, clf):
        #change label to other types in S and calculate err
        #use clfs in stealer to predict
        # when S not null, or it will encounter error
        err_list = []
        leaf_id = clf.apply(xi_vec)
        if len(S):
            # get leaf_id of this new sample, if other samples in S have the same id, change target label
            sum = 0
            text_vec = target.vectorizer.transform(S['text'])
            tar = S['target']
            prob = S['prob']
            S_leaf_id = clf.apply(text_vec)
            same_leaf_index = [i for i in range(len(S_leaf_id)) if S_leaf_id[i] == leaf_id]
            list = tar
            '''change target label into other 9 cats for the leaves which have the same id '''
            if len(same_leaf_index):
                for cat in ['ham', 'spam']:
                    list[same_leaf_index] = cat
                    # calculate error
                    err_index = [i for i in range(len(list)) if S_leaf_id[i] != tar[i]]
                    for e in err_index:
                        sum += 1 / prob[e]
                    sum = (1 / n) * sum
                    err_list.append(sum)
        if len(err_list):
            return min(err_list)
        else:
            return 0


    def update_model(self, model, err_set, tar_lb, vectorizer):
        err_set_vec = vectorizer.transform(err_set)
        model.fit(err_set_vec, tar_lb)
        return model

    def train_base_clf(self, data, label):
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=30, random_state=60)
        tree.fit(data, label)

        return tree
