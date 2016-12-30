import datetime
import random
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from operator import itemgetter

random.seed(20160)


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    return sorted(output)


def run_single(train, test, train_answer, random_state=0):
    eta = 0.2
    max_depth = 6
    subsample = 0.8
    colsample_bytree = 0.8
    min_child_weight = 1
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'
          .format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "multi:softmax",
        "booster": "gbtree",
        "eval_metric": "merror",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
        "min_child_weight": min_child_weight,
        "num_class": 3
    }
    num_boost_round = 300
    early_stopping_rounds = 60
    test_size = 0.1
    train = hstack([train, csr_matrix(train_answer).transpose()])
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    y_train = X_train[:, -1:]
    y_valid = X_valid[:, -1:]
    X_train = X_train[:, :-1]
    X_valid = X_valid[:, :-1]
    print(X_train.shape)
    print(test.shape)

    dtrain = xgb.DMatrix(X_train.todense(), y_train.todense())
    dvalid = xgb.DMatrix(X_valid.todense(), y_valid.todense())

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid.todense()), ntree_limit=gbm.best_iteration+1)
    score = accuracy_score(y_valid.todense(), check.tolist())
    print('Check error value: {:.6f}'.format(score))

    #imp = get_importance(gbm, features)
    #print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test.todense()), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist()


def create_submission(score, test, answer):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    output = pd.DataFrame(data={"activity_id": test['activity_id'], "outcome": answer})
    output.to_csv(sub_file, index=False)