import time
from collections import Counter

from sklearn.tree import DecisionTreeClassifier

from steak.Stealer import Stealer
from steak.Target import Target
import pandas as pd
import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import scipy.sparse as scip
import matplotlib.pyplot as plt
import joblib
from matplotlib.animation import FuncAnimation

step = 200
err_set_num = 100
rgt_set_num = 100

def samples(data):
    # get 50 samples as random samples
    return random.sample(data, 50)

# generate h0 and err_set, rgt_set
def prepare(target, stealer):
    #h0_train = stealer.pool[0:stealer.pool.shape[0]-1]
    h0_train = stealer.data[stealer.row : stealer.row + step].squeeze()
    stealer.row = step
    target_pred_train = target.model.predict(h0_train)
    stealer.queries += step

    # train h0 by a small random dataset
    #h0_vectors = target.vectorizer.transform(h0_train)
    h0 = stealer.train_base_clf(h0_train, target_pred_train)
    stealer.models.append(h0)

    # test on training set and testing set
    h0_test = stealer.data[stealer.row : step + stealer.row]
    stealer.row += step
    h0_test = h0_test.reset_index(drop=True)  # "="
    target_pred_test = target.model.predict(h0_test)
    h0_pred = h0.predict(h0_test)
    print("testing on h0", accuracy_score(target_pred_test, h0_pred))
    print("training on h0", accuracy_score(target_pred_train, h0.predict(h0_train)))

    cols_name = target.x_test.columns.tolist()
    cols_name.append('target')
    stealer.cols_name = cols_name
    err_sample = pd.DataFrame(columns=cols_name)
    rgt_sample = pd.DataFrame(columns=cols_name)

    # pick error samples from pool as error_set0
    for i in range(len(target_pred_test)):
        if target_pred_test[i] != h0_pred[i]:
            x = h0_test.iloc[i].tolist()
            x.append(target_pred_test[i])
            err_sample.loc[len(err_sample)] = x
        else:
            x = h0_test.iloc[i].tolist()
            x.append(target_pred_test[i])
            rgt_sample.loc[len(rgt_sample)] = x
            #rgt_sample.append(h0_test.iloc[i].tolist().append(target_pred_test[i]))
    return err_sample, rgt_sample


def form_err_rgt_set(target, stealer, rd):
    add_new = 200
    test = stealer.data[stealer.row : stealer.row + add_new]
    test = test.reset_index(drop=True)
    stealer.row += add_new     #update row
    # pick 300 err samples from file rd-1, rd-2 and rd-3
    #if rd >= 3:
    j = rd - 1
    while j >= (rd - 3) and j >= 0:
        # err set
        err_data = pd.read_csv("err_set/err_set{}.csv".format(j))
        if len(err_data)>=err_set_num:
            err_data = err_data.sample(err_set_num)

        # rgt set
        rgt_data = pd.read_csv("rgt_set/rgt_set{}.csv".format(j))
        if len(rgt_data)>= rgt_set_num:
            rgt_data = rgt_data.sample(rgt_set_num)

        j -= 1

    return err_data, rgt_data


def merge(err_set):
    chars = err_set[0].shape[1]
    row = len(err_set)
    dd1 = err_set[0]
    i = 1
    while i < len(err_set):
        if i == 1:
            dd = err_set[i]
            dd = scip.vstack(dd1, dd)
        else:
            dd1 = err_set[i]
            dd = scip.vstack(dd, dd1)
        i += 1
    return dd

def store(err_sample, rgt_sample, rd):
    err_sample.to_csv("err_set/err_set{}.csv".format(rd), index=False)
    rgt_sample.to_csv("rgt_set/rgt_set{}.csv".format(rd), index=False)


def show_pic(rd, accuracy):
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.title('model stealing')
    plt.plot(rd, accuracy, "b--", linewidth=1)
    plt.show()
    time.sleep(3)
    plt.close()

def test_accu(stealer, target):
    x_test = target.x_test
    y_test = target.y_test
    pred = stealer.forest_predict(x_test, rows=len(x_test))

    target_pred = target.model.predict(x_test)
    return accuracy_score(target_pred, pred), accuracy_score(y_test, pred)

def update_clf(err_set, tar_lb, add_random, target, clf):
    add_random = add_random.tolist()
    lb = target.model.predict(target.vectorizer.transform(add_random)).tolist()
    err_set = err_set + add_random
    tar_lb = tar_lb + lb
    #clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60)
    clf.fit(target.vectorizer.transform(err_set), tar_lb)
    return clf


if __name__ == '__main__':
    target_type = "XGB"
    pro = 30
    print("type:{}, pro:{}, step:{}, errnum:{}, aux_num:{}".
          format(target_type, pro, step, err_set_num, rgt_set_num))
    target = Target(target_type)
    stealer = Stealer(target, pro)
    print("proportion of training set:{}".format(stealer.proportion))

    '''test aug data'''
    # sub_clfs = []
    # for i in range(9):
    #     clf = joblib.load("models/sub_clf{}-{}%.pkl".format(i, stealer.proportion))
    #     sub_clfs.append(clf)
    #
    # stealer.models = sub_clfs
    # # load aug data
    # test_aug = stealer.train_aug
    # test_aug_y = stealer.train_aug_y
    # ste_pred = stealer.forest_predict(test_aug, len(test_aug))
    # tar_pred = target.predict(test_aug)
    # print("agreement on aug data:", accuracy_score(ste_pred, tar_pred))
    # print("accuracy on aug data:", accuracy_score(ste_pred, test_aug_y))

    ''''''

    # train tree0 and err_set
    err_sample, rgt_sample = prepare(target, stealer)   #return dataframe
    rd = []
    accuracy1 = []
    accuracy2 = []
    i = 0
    accu1, accu2 = test_accu(stealer, target)
    print("agree {}, acc {}".format(accu1, accu2))
    accuracy1.append(accu1)
    accuracy2.append(accu2)
    pool_len = len(stealer.data)
    while stealer.row <= pool_len:
        if i >= 1:
            # store err_set and tar_lb in files
            err_sample, rgt_sample = form_err_rgt_set(target, stealer, i)

        err_data = err_sample.drop(['target'], axis=1)
        rgt_data = rgt_sample.drop(['target'], axis=1)
        err_label = err_sample['target'].tolist()
        rgt_label = rgt_sample['target'].tolist()
        frame = [err_data, rgt_data]
        both_data = pd.concat(frame)
        both_lb = err_label + rgt_label
        clf = stealer.train_base_clf(both_data, both_lb)

        add_random = stealer.data[stealer.row : stealer.row+step]
        stealer.row += step
        #clf = update_clf(err_set, tar_lb, add_random, target, clf)
        clf, err_data, err_label = stealer.active_learning(clf, add_random, target, both_data, both_lb)

        # test clf
        clf_pred = clf.predict(target.x_test)
        print("acc of clf:{}".format(accuracy_score(clf_pred, target.y_test)))

        err_label = pd.DataFrame(err_label, columns=['target'])
        err_data = err_data.reset_index(drop=True)
        err_sample = pd.concat([err_data, err_label], axis=1)

        # save clf in round i
        joblib.dump(clf, "models/sub_clf{}-{}%.pkl".format(i, stealer.proportion))

        store(err_sample, rgt_sample, i)
        stealer.models.append(clf)
        accu1, accu2 = test_accu(stealer, target)

        print("-------------------------------round {}, {} {}, queries={}-----------------------------------".format(i, accu1, accu2, stealer.queries))
        rd.append(stealer.queries)
        accuracy1.append(accu1)
        accuracy2.append(accu2)
        diff = abs(accuracy1[i] - accuracy1[i-1])
        print("diff".format(diff))
        if i >= 1 and diff <= 0.05:
            step += 500
            print("expand step to {}".format(step))
        i += 1
    plt.xlabel('queries')
    plt.ylabel('agreement')
    plt.title('{} extraction steak-{}%'.format(target_type, stealer.proportion))
    plt.plot(rd, accuracy1, "b--", linewidth=1)
    plt.show()

    plt.xlabel('queries')
    plt.ylabel('accuracy')
    plt.title('{} extraction steak-{}%'.format(target_type, stealer.proportion))
    plt.plot(rd, accuracy2, "b--", linewidth=1)
    plt.show()
