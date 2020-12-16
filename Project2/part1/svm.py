import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    clf = LinearSVC(random_state=0, C=0.1)
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    """
    pred_test_y = np.zeros(test_x.shape[0])
    for label in np.unique(train_y):
        current_label = train_y - label
        current_y = np.where(current_label == 0, 0, 1)
        current_pred = one_vs_rest_svm(train_x, current_y, test_x)
        pred_test_y = np.where(current_pred == 0, label, pred_test_y)
    return pred_test_y
    """
    clf = LinearSVC(random_state=0, C=0.1)
    clf.fit(train_x, train_y)
    return clf.predict(test_x)

def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

