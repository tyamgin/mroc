import _mroc
import numpy as np

def mean_roc_auc(labels, actual, pred):
    labels = np.array(labels, dtype='int32')
    actual = np.array(actual, dtype='int32')
    pred = np.array(pred, dtype='double')

    assert len(labels.shape) == 1
    assert len(actual.shape) == 1
    assert len(pred.shape) == 1
    assert labels.shape[0] == actual.shape[0] == pred.shape[0]

    return _mroc.mean_roc_auc(labels, actual, pred)