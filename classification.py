import os, sys
from itertools import product
import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data_locator import ret_grads, ret_losses, ret_zoo_weights, ret_embedded_weights
from data_locator import request_logger, log_it

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix,  accuracy_score, recall_score, precision_score, f1_score

logger = request_logger(__name__)

outlier = 1
inlier = 0

# digits = tuple(range(1,10,1))
digits = tuple([1])
epss = list(map(float,[f'{digit}e-{i}' for i in range(5,30) for digit in digits]))
epss.append(0)

stds = [0.5, 1, 2]

def load_grads(lsb=-1, zoo_name='mnist', msb=False, split=False, segregate=False, malware_name='malware_12584bytes'):
    ret = ret_grads(zoo_name=zoo_name, malware_name=malware_name, lsb=lsb, msb=msb)
    if ret is None:
        raise Exception('No grads found')
    
    if lsb == -1:
        return ret
    else:
        grads_unmodified, grads_modified = ret
        if segregate:
            return grads_unmodified, grads_modified
        
        X = np.concatenate([grads_unmodified, grads_modified])
        y = np.concatenate([np.full((grads_unmodified.shape[0]), inlier), np.full((grads_modified.shape[0]), outlier)])
        
        if split:
            return train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        else:
            return X, y

def load_losses(lsb=-1, zoo_name='mnist', msb=False, split=False, malware_name='malware_12584bytes'):
    ret = ret_losses(zoo_name=zoo_name, malware_name=malware_name, msb=msb)
    if ret is None:
        raise Exception('No losses found')
    
    assert lsb in (-1, 0, *range(1,24)), 'lsb must be -1 or 0 or in range(1,24)'
    
    if lsb==-1:
        return ret
    
    losses = ret['losses']
    if lsb==0:
        
        X = losses[0]
        y = np.full((len(X)), inlier)

    else:
        losses_unmodified = losses[0]
        losses_modified = losses[lsb]

        X = np.hstack([losses_unmodified, losses_modified]).reshape(-1, 1)
        y = np.concatenate([np.full((losses_unmodified.shape[0]), inlier), np.full((losses_modified.shape[0]), outlier)])
    
    if split:
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    else:
        return X, y

def load_weights(lsb=1, zoo_name='mnist', msb=False, split=False, malware_name='malware_12584bytes'):
    weights_unmodified = ret_zoo_weights(zoo_name=zoo_name)
    weights_modified = ret_embedded_weights(zoo_name=zoo_name, malware_name=malware_name, lsb=lsb, msb=msb)

    X = np.concatenate([weights_unmodified, weights_modified])
    y = np.concatenate([np.full((weights_unmodified.shape[0]), inlier), np.full((weights_modified.shape[0]), outlier)])

    if split:
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    else:
        return X, y

def generate_supervised_results(zoos=["mnist", "stl10", "cifar10", "svhn"], msb=False, n_jobs=-1):
    @log_it
    def generate_supervised_result(zoo_name='mnist', feature_type='grads'):
        def generate_result(lsb):
            assert 1 <= lsb <= 23

            if feature_type=='grads':
                X_train, X_test, y_train, y_test = load_grads(lsb, zoo_name=zoo_name, msb=msb, split=True)
            elif feature_type=='losses':
                X_train, X_test, y_train, y_test = load_losses(lsb, zoo_name=zoo_name, msb=msb, split=True)
            elif feature_type=='weights':
                X_train, X_test, y_train, y_test = load_weights(lsb, zoo_name=zoo_name, msb=msb, split=True)

            results_lsb = np.empty((len(clfs), 2+2+5), dtype=object)

            for iclf, clf_class in enumerate(clfs):
                clf_name = str(clf_class.__name__).replace("Classifier", "")

                clf = clf_class()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                y_true = y_test

                acc = accuracy_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                conf_mat = confusion_matrix(y_true, y_pred, labels=[inlier, outlier])

                results_lsb[iclf,:] = zoo_name, feature_type, lsb, clf_name, acc, recall, precision, f1, conf_mat
            return results_lsb
        # logger.info(f'Generating classification results for {zoo_name} | features: {features} | msb: {msb} | n_jobs: {n_jobs}')

        assert feature_type in ('grads', 'losses', 'weights')
        lsbs = range(1,24)
        clfs = [HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier]
        if feature_type=='weights':
            clfs = [HistGradientBoostingClassifier, RandomForestClassifier]
        
        results_l = Parallel(n_jobs=n_jobs)(delayed(generate_result)(i) for i in lsbs)
        results_curr = np.vstack(results_l)

        return results_curr
    
    results = []
    features = ['grads', 'losses', 'weights']
    for zoo_name, feature in product(zoos, features):
        results_curr = generate_supervised_result(zoo_name, feature)
        results.append(results_curr)

    df = pd.DataFrame(np.vstack(results), columns=['zoo', 'feature_type', 'lsb', 'clf', 'accuracy', 'recall', 'precision', 'f1', 'conf_mat'])
    os.makedirs('./results', exist_ok=True)
    df.to_csv(f'./results/results_supervised{"_msb" if msb else ""}.csv', index=False)

inlier_outlier_map = {False: inlier, True: outlier}
inlier_outlier_vectorize = np.vectorize(lambda x: inlier_outlier_map[x])

def generate_unsupervised_results(zoos=["mnist", "stl10", "cifar10", "svhn"], msb=False, malware_name='malware_12584bytes'):

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    results = []

    @log_it
    def generate_unsupervised_result(zoo_name='mnist', spacer_type="eps"):
        assert spacer_type in ("eps", "std")

        def threshold_plus_spacer(data, spacer):
            bigger_arr = data > (threshold + spacer)
            y_pred = inlier_outlier_vectorize(bigger_arr)
            return y_pred

        losses_z = ret_losses(zoo_name=zoo_name, malware_name=malware_name, msb=msb)
        losses = losses_z['losses']
        sizes = losses_z['sizes']

        train_size = int(sizes[0])
        test_size = int(sizes[1])

        threshold = losses_z['avg'][0]
        losses_z.close()

        X_unmodified_train, X_unmodified_test = losses[0][:train_size], losses[0][train_size:]
        y_unmodified_train, y_unmodified_test = np.full((train_size), inlier), np.full((test_size), inlier)

        threshold = np.mean(X_unmodified_train)
        train_std = np.std(X_unmodified_train)
        logger.info(f'finished STEP 1: threshold: {threshold}')
        def find_best_spacer():
            best_score = -1
            best_spacer = 0
            best_y_pred = None
            for spacer in spacer_list:
                curr_spacer = spacer if spacer_type == "eps" else spacer * train_std
                y_pred = threshold_plus_spacer(X_unmodified_train, curr_spacer)
                curr_score = accuracy_score(y_unmodified_train, y_pred)

                if curr_score > best_score:
                    best_score = curr_score
                    best_spacer = spacer
                    best_y_pred = y_pred

                if curr_score == best_score and spacer < best_spacer:
                    best_spacer = spacer
                    best_y_pred = y_pred

            logger.info(f'best score from train data: {best_score}')

            return best_spacer

        spacer_list = epss if spacer_type == "eps" else stds

        best_spacer = find_best_spacer()
        logger.info(f'finished STEP 2: best spacer from train data: {best_spacer}{"std" if spacer_type == "std" else ""}')
        if spacer_type == "std":
            spacer_value = best_spacer
            best_spacer *= train_std
        else:
            spacer_value = best_spacer
        
        y_test = np.concatenate([y_unmodified_test, np.full((len(losses[1])), outlier)])
        def classify_lsb(lsb):
            X_test = np.hstack([X_unmodified_test, losses[lsb]])
            y_pred = threshold_plus_spacer(X_test, best_spacer)

            y_true = y_test
            acc = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            results.append((zoo_name, spacer_type, spacer_value, lsb, acc, recall, precision, f1))

        for i in range(1,24):
            classify_lsb(i)

    for zoo_name, spacer_type in product(zoos, ("eps", "std")):
        generate_unsupervised_result(zoo_name, spacer_type)

    df = pd.DataFrame(results, columns=['zoo', 'spacer_type', 'spacer_value', 'lsb', 'accuracy', 'recall', 'precision', 'f1'])
    df.to_csv(f'./results/results_unsupervised{"_msb" if msb else ""}.csv', index=False)

if __name__ == '__main__':
    generate_supervised_results(zoo_name='mnist', msb=False, n_jobs=-1)
