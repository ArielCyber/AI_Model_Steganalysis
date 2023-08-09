import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data_locator import ret_grads

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix,  accuracy_score, recall_score, precision_score, f1_score

outlier = 1
inlier = 0

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
        
def generate_results(zoo_name='mnist', msb=False, n_jobs=-1):
    def generate_result(lsb, override_results=True):
        assert 1 <= lsb <= 23

        X_train, X_test, y_train, y_test = load_grads(lsb, zoo_name=zoo_name, msb=msb, split=True)

        results_lsb = np.empty((len(clfs), 2+5), dtype=object)

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

            results_lsb[iclf,:] = lsb, clf_name, acc, recall, precision, f1, conf_mat
        return results_lsb

    lsbs = range(1,24)
    clfs = [HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier]
    
    results_l = Parallel(n_jobs=n_jobs)(delayed(generate_result)(i) for i in lsbs)
    results = np.vstack(results_l)
    
    df = pd.DataFrame(results, columns=['lsb', 'clf', 'accuracy', 'recall', 'precision', 'f1', 'conf_mat'])
    os.makedirs('./results', exist_ok=True)
    df.to_csv(f'./results/results_{zoo_name}{"_msb" if msb else ""}.csv', index=False)

if __name__ == '__main__':
    generate_results(zoo_name='mnist', msb=False, n_jobs=-1)
