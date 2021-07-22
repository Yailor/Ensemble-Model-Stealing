import time
from collections import Counter

from sklearn.tree import DecisionTreeClassifier

from spam.Stealer import Stealer
from spam.Target import Target
import pandas as pd
import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import scipy.sparse as scip
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

step = 200
err_set_num = 100
rgt_set_num = 100

# generate h0 and err_set, rgt_set
def prepare(target, stealer):
    #h0_train = stealer.pool[0:stealer.pool.shape[0]-1]
    h0_train = stealer.data[stealer.row : stealer.row + step]
    stealer.row = step
    h0_train_vec = target.vectorizer.transform(h0_train)
    target_pred_train = target.model.predict(h0_train_vec)
    stealer.queries += step

    # train h0 by a small random dataset
    #h0_vectors = target.vectorizer.transform(h0_train)
    h0 = stealer.train_base_clf(h0_train_vec, target_pred_train)
    stealer.models.append(h0)

    # test on training set and testing set
    h0_test = stealer.data[stealer.row : step + stealer.row]
    stealer.row += step
    h0_test = h0_test.reset_index(drop=True)  # "="
    h0_test_vec = target.vectorizer.transform(h0_test)
    target_pred_test = target.model.predict(h0_test_vec)
    h0_pred = h0.predict(h0_test_vec)
    print("testing on h0", accuracy_score(target_pred_test, h0_pred))
    print("training on h0", accuracy_score(target_pred_train, h0.predict(h0_train_vec)))

    err_set = []
    target_label = []
    rgt_set = []
    rgt_lb = []

    # pick error samples from pool as error_set0
    for i in range(len(target_pred_test)):
        if target_pred_test[i] != h0_pred[i]:
            err_set.append(h0_test[i])
            target_label.append(target_pred_test[i])
        else:
            rgt_set.append(h0_test[i])
            rgt_lb.append(h0_pred[i])
    return err_set, target_label, rgt_set, rgt_lb


def test_accu(stealer, target):
    #test = stealer.data[len(stealer.data)-500 : len(stealer.data)]
    test = pd.read_csv("data/spam_aug_pool/spam.csv")
    true_lb = test['target'][:5000]
    test = test['text'][:5000]
    test_vector = target.vectorizer.transform(test)
    target_pred = target.predict(test_vector)
    pred = stealer.forest_predict(test_vector, rows=5000)
    # print("accu of target:{}".format(accuracy_score(target_pred, label)))
    return accuracy_score(target_pred, pred), accuracy_score(true_lb, pred)


def form_err_rgt_set(target, stealer, rd):
    err_set = []
    tar_lb = []
    rgt_set = []
    rgt_lb = []
    j = rd - 1
    while j >= (rd - 3) and j >= 0:
        # err set
        err_data = pd.read_csv("err_set/err_set{}.csv".format(j))
        if len(err_data) >= err_set_num: err_data = err_data.sample(err_set_num)
        review = err_data['text'].tolist()
        sentiment = err_data['target'].tolist()
        '''if not len(err_set):    #if err_set is null
            err_set = review
            tar_lb = sentiment
        else:'''
        err_set = err_set + review
        tar_lb = tar_lb + sentiment
        # rgt set
        rgt_data = pd.read_csv("rgt_set/rgt_set{}.csv".format(j))
        if len(rgt_data) >= rgt_set_num: rgt_data = rgt_data.sample(rgt_set_num)
        rgt_review = rgt_data['text'].tolist()
        rgt_sentiment = rgt_data['target'].tolist()
        rgt_set = rgt_set + rgt_review
        rgt_lb = rgt_lb + rgt_sentiment

        j -= 1

    return err_set, tar_lb, rgt_set, rgt_lb


def test_clf_accu(stealer, target, clf):
    #test = stealer.data[len(stealer.data)-500 : len(stealer.data)]
    test = pd.read_csv("data/spam_aug_pool/spam.csv")
    test = test['text'][:5000]
    test_vector = target.vectorizer.transform(test)
    target_pred = target.predict(test_vector)
    pred = clf.predict(test_vector)
    # print("accu of target:{}".format(accuracy_score(target_pred, label)))
    return accuracy_score(target_pred, pred)

def store(err_set, tar_lb, rgt_set, rgt_lb, rd):
    err_data = pd.DataFrame({"text": err_set,
            "target": tar_lb
            })
    err_data.to_csv("err_set/err_set{}.csv".format(rd))
    rgt_data = pd.DataFrame({
        "text": rgt_set,
        "target": rgt_lb
    })
    rgt_data.to_csv("rgt_set/rgt_set{}.csv".format(rd))


def update_clf(err_set, tar_lb, add_random, target, clf):
    add_random = add_random.tolist()
    lb = target.model.predict(target.vectorizer.transform(add_random)).tolist()
    err_set = err_set + add_random
    tar_lb = tar_lb + lb
    #clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60)
    clf.fit(target.vectorizer.transform(err_set), tar_lb)
    return clf

def test_aug_data(stealer, target):
    data = pd.read_csv("data/spam_aug_pool/spam-{}%/spam-{}%.csv".format(stealer.proportion, stealer.proportion))
    X = data['text']
    y = data['target']
    X_vec = target.vectorizer.transform(X)
    pred = stealer.forest_predict(X_vec, len(data))
    print(accuracy_score(pred, y))

if __name__ == '__main__':
    tar_type = "XGB"
    print("pro=20")
    pro = 30
    print("type:{}, pro:{}, step:{}, errnum:{}, aux_num:{}".
          format(tar_type, pro, step, err_set_num, rgt_set_num))
    target = Target(tar_type)
    stealer = Stealer(target, pro)
    # train tree0 and err_set
    err_set, tar_lb, rgt_set, rgt_lb = prepare(target, stealer)
    rd = []
    accuracy1 = []
    accuracy2 = []

    accu1, accu2 = test_accu(stealer, target)
    print("agree {}, acc {}".format(accu1, accu2))
    accuracy1.append(accu1)
    accuracy2.append(accu2)

    i = 0
    pool_len = len(stealer.data)
    while stealer.row < (pool_len-500):
    # while i < 200:
        # train clf in this round, first it should be trained on err_set
        if i >= 1:
            # store err_set and tar_lb in files
            err_set, tar_lb, rgt_set, rgt_lb = form_err_rgt_set(target, stealer, i)

        both_set = err_set + rgt_set
        both_lb = tar_lb + rgt_lb
        #transform
        both_set_vec = target.vectorizer.transform(both_set)
        clf = stealer.train_base_clf(both_set_vec, both_lb)
        #
        # err_set_vec = target.vectorizer.transform(err_set)
        # tar_err_pred = target.model.predict(err_set_vec)
        # print("accu on err_set training:{}".format(accuracy_score(clf.predict(err_set_vec),
        #                                                           tar_err_pred)))
        # print("test accu of clf:", test_clf_accu(stealer, target, clf))

        add_random = stealer.data[stealer.row : stealer.row + step]
        stealer.row += step
        #clf = update_clf(err_set, tar_lb, add_random, target, clf)
        clf, err_set, tar_lb = stealer.active_learning(clf, add_random, target, both_set, both_lb)
        store(err_set, tar_lb, rgt_set, rgt_lb, i)
        # # test accu of new clf
        # print("test accu of new clf:", test_clf_accu(stealer, target, clf))
        # # check if pred on err_set is correct
        # print("accu on err_set:{}".format(accuracy_score(clf.predict(err_set_vec),
        #                                                           tar_err_pred)))
        stealer.models.append(clf)
        accu1, accu2 = test_accu(stealer, target)

        print("-------------------------------round {}, {} {}, queries={}-----------------------------------".format(i, accu1, accu2, stealer.queries))
        rd.append(stealer.queries)
        accuracy1.append(accu1)
        accuracy2.append(accu2)

        '''judge whether to stop'''
        if abs(target.accuracy - accu2) <= 0.01:
            print("target_accu={}, accu2={}".format(target.accuracy, accu2))
            break
        '''when accu cannot rise, expand step'''
        if i >= 1 and abs(accuracy2[i] - accuracy2[i-1]) <= 0.005:
            step += 500
            print("expand step to {}".format(step))

        i += 1

    plt.xlabel('queries')
    plt.ylabel('agreement')
    plt.title('{} extraction on spam-{}%'.format(tar_type, stealer.proportion))
    plt.plot(rd, accuracy1, "b--", linewidth=1)
    plt.show()

    plt.xlabel('queries')
    plt.ylabel('accuracy')
    plt.title('{} extraction on spam-{}%'.format(tar_type, stealer.proportion))
    plt.plot(rd, accuracy2, "b--", linewidth=1)
    plt.show()

    print("err_set_num={}, rgt_set_num={}, step={}".format(err_set_num, rgt_set_num, step))
    '''test accu on augmented data'''
    test_aug_data(stealer, target)
