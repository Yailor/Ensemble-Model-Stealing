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
from mnist.augment import augment
BASE = 10
c0 = c1 = c2 = 0.5


class Stealer():
    def __init__(self, target, pro):
        super(Stealer, self).__init__()
        self.models = []
        # dataframe, 1 column
        self.data, self.init_x, self.init_y = self.load_pool(pro=pro)
        self.row = 0
        self.queries = 0


    def flip_biased(self, p):
        return random.random() < p

    def forest_predict(self, data, rows):
        lbs = np.ones((len(self.models), rows))
        lbs = lbs.astype(np.str)    # result is str
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
            max_lbs.append(int(max_lb))
        return max_lbs

    def add_elm(self, err_set, tar_lb, elm):
        err_set.append(elm['pic'])
        tar_lb.append(elm['target'])
        return err_set, tar_lb

    def active_learning(self, clf, random_pool, target, err_set, tar_lb, categories):
        flag_update = 0
        i = 1
        S = pd.DataFrame(columns=['pic', 'target', 'prob'])
        while i <= len(random_pool):
            xi = random_pool[i-1]
            #yi = clf.predict(xi_vec) #clf predict yi
            # S is a set of random samples in pool
            hn = self.get_hn_error(i, S, target, clf)
            hnn = self.get_hnn_error(i, S, xi, target, clf, categories)      #modify
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
                y = target.model.predict(np.reshape(xi, (1, -1)))[0]
                self.queries += 1
                #print("xi:{} yi:{}, y:{}".format(xi, yi, y))
                #S.append([xi, y, p])
                '''add sample to S'''
                elm = {'pic': xi, 'target': y, 'prob': p}
                S = S.append(elm, ignore_index=True)
            else:
                p = s
                toss = self.flip_biased(p)
                if toss:
                    flag_update = 1
                    y = target.model.predict(np.reshape(xi, (1, -1)))[0]
                    self.queries += 1
                    #print("xi:{} yi:{}, y:{}".format(xi, yi, y))
                    #S.append([xi, y, p])
                    elm = {'pic': xi, 'target': y, 'prob': p}
                    S = S.append(elm, ignore_index=True)
            if flag_update:
                err_set, tar_lb = self.add_elm(err_set, tar_lb, elm)
                clf = self.update_model(clf, err_set, tar_lb)

            #print("{} times of clf, gn={}, lenS={}, u={}, hn={}, hnn={}".format(i, round(gn, 4), len(S), round(u, 4), round(hn, 4), round(hnn, 4)))
            i += 1
        '''if flag_update:
            clf = self.update_model(clf, err_set, tar_lb, S, target.vectorizer)'''
        #print("clf train end, lenS={}".format(len(S)))

        self.models.append(clf)
        return clf, err_set, tar_lb


    def load_pool(self, pro):
        return augment(proportion=pro)


    def get_hn_error(self, n, S, target, clf):
        sum = 0
        if len(S):
            data = S['pic'].values.tolist()
            lbs = clf.predict(data)
            tar = S['target']
            prob = S['prob']
            for i in range(len(S)):
                if lbs[i] != tar[i]:
                    sum += 1 / prob[i]
            sum = (1 / n) * sum
        return sum

    def change_label(self, S_lb):
        if S_lb == 'positive':
            return 'negative'
        else: return 'positive'

    def flip_predict(self, X, j):
        lb, lbs = self.model.predict_single(X)
        for i in range(len(lbs)):
            if i in j:
                lbs[i] = self.change_label(lb)

        result = Counter(lbs)
        if result['1'] >= self.model.n_estimators/2: return '1', lbs
        else: return '2', lbs


    def get_hnn_error(self, n, S, xi, target, clf, categories):
        #change label to other types in S and calculate err
        #use clfs in stealer to predict
        # when S not null, or it will encounter error
        err_list = []
        leaf_id = clf.apply([xi])[0]
        if len(S):
            #get leaf_id of this new sample, if other samples in S have the same id, change target label
            sum = 0
            pic = S['pic'].values.tolist()
            tar = S['target']
            prob = S['prob']
            S_leaf_id = clf.apply(pic)
            same_leaf_index = [i for i in range(len(S_leaf_id)) if S_leaf_id[i] == leaf_id]
            list = tar
            '''change target label into other 9 cats for the leaves which have the same id '''
            if len(same_leaf_index):
                for cat in categories:
                    list[same_leaf_index] = cat
                    # calculate error
                    err_index = [i for i in range(len(list)) if S_leaf_id[i] != tar[i]]
                    for e in err_index:
                        sum += 1 / prob[e]
                    sum = (1 / n) * sum
                    err_list.append(sum)
        if len(err_list):
            return min(err_list)
        else: return 0


    def update_model(self, model, err_set, tar_lb):
        model.fit(err_set, tar_lb)
        return model

    def train_base_clf(self, data, label):
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=60, random_state=60)
        tree.fit(data, label)

        return tree



