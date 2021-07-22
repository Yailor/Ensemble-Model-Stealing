import time

from mnist.Stealer import Stealer
from mnist.Target import Target
import pandas as pd
import random
import numpy as np
import scipy.sparse as scip
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

step = 500
err_set_num = 200
rgt_set_num = 200

(X_train, y_train), (X_test, y_test) = mnist.load_data()
n_train_samples, nx_tr, ny_tr = X_test.shape
X_test = X_test.reshape((n_train_samples, nx_tr * ny_tr))
categories = set(y_test)



def samples(data):
    # get 50 samples as random samples
    return random.sample(data, 50)


# generate h0 and err_set, rgt_set
def prepare(target, stealer):
    first_size = 5000
    h0_train = stealer.data[stealer.row : stealer.row + first_size]
    stealer.row += first_size
    target_pred_train = target.model.predict(h0_train)
    stealer.queries += first_size

    # train h0 by a small random dataset
    h0 = stealer.train_base_clf(h0_train, target_pred_train)
    # nsamples, nx, ny = stealer.init_x.shape
    # stealer.init_x = stealer.init_x.reshape((nsamples, nx * ny))
    # target_pred_train = target.model.predict(stealer.init_x)
    # h0 = stealer.train_base_clf(stealer.init_x, target_pred_train)
    stealer.models.append(h0)

    # test on training set and testing set
    h0_test = stealer.data[stealer.row : step + stealer.row]
    stealer.row += step
    stealer.queries += step
    target_pred_test = target.model.predict(h0_test)
    h0_pred = h0.predict(h0_test)

    #print("testing on h0", accuracy_score(target_pred_test, h0_pred))
    print("training on h0", accuracy_score(target_pred_train, h0.predict(h0_train)))

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
    target_pred = target.predict(X_test)
    pred = stealer.forest_predict(X_test, rows=X_test.shape[0])
    return accuracy_score(target_pred, pred), accuracy_score(y_test, pred)


def form_err_rgt_set(target, stealer, rd):
    '''1. add err_set from the last 3 files/rounds (was predicted wrongly)
       2. add rgt_set from the last 3 files/rounds (was predicted correctly)
    '''
    err_set = []
    tar_lb = []
    rgt_set = []
    rgt_lb = []
    # pick 300 err samples from file rd-1, rd-2 and rd-3
    #if rd >= 3:
    j = rd - 1
    while j >= (rd - 3) and j >= 0:
        # err set
        # err_data = pd.read_csv("err_set/err_set{}.csv".format(j))
        # if len(err_data)>=100: err_data = err_data.sample(100)
        # review = np.array(err_data['pic']).reshape(err_data['pic'].size, 1)
        # # sentiment = err_data['target'].tolist()
        pic = np.load("err_set/err_set_data{}.npy".format(j)).tolist()
        tar = pd.read_csv("err_set/err_set_label{}.csv".format(j))['target'].tolist()
        if len(tar) >= err_set_num:
            pic = pic[0:err_set_num]
            tar = tar[0:err_set_num]

        err_set = err_set + pic
        tar_lb = tar_lb + tar

        # rgt set
        rgt_data = np.load("rgt_set/rgt_set_data{}.npy".format(j)).tolist()[0:rgt_set_num]
        rgt_tar = pd.read_csv("rgt_set/rgt_set_label{}.csv".format(j))['target'].tolist()[0:rgt_set_num]
        rgt_set = rgt_set + rgt_data
        rgt_lb = rgt_lb + rgt_tar

        j -= 1
    #
    # # test stealer.models by test set
    # stealer_pred = stealer.forest_predict(test, rows=len(test))
    # stealer.queries += add_new      # query
    # target_pred = target.model.predict(test).tolist()
    #
    # for i in range(len(stealer_pred)):
    #     if stealer_pred[i] != target_pred[i]:
    #         err_set.append(test[i].tolist())
    #         tar_lb.append(target_pred[i])
    #     else:
    #         if random.random() >= 0.5:
    #             rgt_set.append(test[i].tolist())
    #             rgt_lb.append(stealer_pred[i])
    print("len rgt {}".format(len(rgt_lb)))
    return err_set, tar_lb, rgt_set, rgt_lb


# def test_clf_accu(target, clf):
#     target_pred = target.predict(X_test)
#     pred = clf.predict(X_test)
#     return accuracy_score(target_pred, pred)

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

def store(err_set, tar_lb, rgt_set, rgt_lb, rd):

    # save err data
    np.save("err_set/err_set_data{}.npy".format(rd), err_set)
    tar_lb = pd.DataFrame(columns=['target'], data=tar_lb)
    tar_lb.to_csv("err_set/err_set_label{}.csv".format(rd))
    #np.save("err_set/err_set_label{}.npy".format(rd), tar_lb)
    #err_set2 = np.load("err_set/err_set_data{}.npy".format(rd))
    #tar_lb2 = pd.read_csv("err_set/err_set_label{}.csv".format(rd))['target']

    # save rgt data
    np.save("rgt_set/rgt_set_data{}.npy".format(rd), rgt_set)
    rgt_lb = pd.DataFrame(columns=['target'], data=rgt_lb)
    rgt_lb.to_csv("rgt_set/rgt_set_label{}.csv".format(rd))

def update_clf(err_set, tar_lb, add_random, target, clf):
    #add_random = add_random.tolist()
    lb = target.model.predict(add_random).tolist()
    err_set = err_set + add_random
    tar_lb = tar_lb + lb
    #clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60)
    clf.fit(err_set, tar_lb)
    return clf

def test_augdata(stealer):
    n_train_samples, nx_tr, ny_tr = stealer.init_x.shape
    X = stealer.init_x.reshape((n_train_samples, nx_tr * ny_tr))
    pred = stealer.forest_predict(data=X, rows=n_train_samples)
    print("accuracy on init aug data:{}".format(accuracy_score(pred, stealer.init_y)))

if __name__ == '__main__':
    pro = 0.2
    target_type = "XGB"
    print("type:{}, pro:{}, step:{}, errnum:{}, aux_num:{}".
          format(target_type, pro, step, err_set_num, rgt_set_num))
    target = Target(target_type)
    stealer = Stealer(target, pro)
    print("type={}".format(target_type))
    # train tree0 and err_set
    err_set, tar_lb, rgt_set, rgt_lb = prepare(target, stealer)
    target_pred = target.predict(X_test)
    pred = stealer.forest_predict(X_test, rows=X_test.shape[0])
    print("h0 agree:{}, acc:{}".format(accuracy_score(target_pred, pred), accuracy_score(y_test, pred)))
    rd = []
    accuracy1 = []
    accuracy1.append(accuracy_score(target_pred, pred))
    accuracy2 = []
    accuracy2.append(accuracy_score(y_test, pred))
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

        clf = stealer.train_base_clf(both_set, both_lb)
        #
        # tar_err_pred = target.model.predict(err_set)
        # print("accu on err_set training:{}".format(accuracy_score(clf.predict(err_set),
        #                                                           tar_err_pred)))
        # print("test accu of clf:", test_clf_accu(target, clf))

        add_random = stealer.data[stealer.row : stealer.row + step]
        stealer.row += step
        #clf = update_clf(err_set, tar_lb, add_random, target, clf)
        clf, err_set, tar_lb = stealer.active_learning(clf, add_random, target, both_set, both_lb, categories)

        # store err_set and rgt_set after active learning
        store(err_set, tar_lb, rgt_set, rgt_lb, i)
        #
        # # test accu of new clf
        # print("test accu of new clf:", test_clf_accu(target, clf))
        # # check if pred on err_set is correct
        # print("accu on err_set:{}".format(accuracy_score(clf.predict(err_set), tar_lb)))
        stealer.models.append(clf)
        accu1, accu2 = test_accu(stealer, target)

        print("-------------------------------round {}, {} {},queries={}-----------------------------------".format(i, accu1, accu2, stealer.queries))
        rd.append(stealer.queries)
        accuracy1.append(accu1)
        accuracy2.append(accu2)

        '''judge whether to stop'''
        if abs(target.accuracy - accu2) <= 0.05: break
        '''when accu cannot rise, expand step'''
        if i >= 3 and ( (accuracy2[i] - accuracy2[i-1]) <= 0.005 or (accuracy2[i-1] - accuracy2[i-2]) <= 0.005 ):
            step += 500
            print("expand step to {}".format(step))

        i += 1

    plt.xlabel('queries')
    plt.ylabel('agreement')
    plt.title('{} extraction on MNIST-{}%'.format(target_type, pro))
    plt.plot(rd, accuracy1, "b--", linewidth=1)
    plt.show()

    plt.xlabel('queries')
    plt.ylabel('accuracy')
    plt.title('{} extraction on MNIST-{}%'.format(target_type, pro))
    plt.plot(rd, accuracy2, "b--", linewidth=1)
    plt.show()

    print("err_set_num={}, rgt_set_num={}, step={}".format(err_set_num, rgt_set_num, step))
    '''test augmentation data'''
    test_augdata(stealer)


