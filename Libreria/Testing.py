from sklearn.model_selection import train_test_split, cross_val_score
from Libreria.ConfusionMatrix import ConfusionMatrix
from Libreria.Dataset import Dataset
from statistics import stdev
import math
def testClassifier(clf, data, label, split):
    '''
    tests the performace of a classifier "clf" against the "data" samples.
    "label" is the list of the corresponding labels of elements in list "data".
    "split" is the fraction (from 0 to 1) of the dataset used for testing.
    '''
    #divide il dataset
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=split)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return ConfusionMatrix(y_test, predicted)

def kFoldCrossValidation(datasets, class_index, clf):
    """
    Apply K-fold Cross validation with a given split of the dataset.
        - Datasets: array of Dataset objects
        - class_index: the index of the class attribute
        - clf a scikit-learn classifier
    """
    n = len(datasets)
    results = []
    for i in range(n):
        print("DEBUG --> K validation, fold {}".format(i))
        x_test, y_test = datasets[i].divideFeaturesAndLabels(class_index)
        train = Dataset()
        for j in range(n):
            if j != i:
                train.appendDataset(datasets[j])
        #perform testing
        x_train, y_train = train.divideFeaturesAndLabels(class_index)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        cm = ConfusionMatrix(y_test, pred)
        results += [cm]
    return results

def ninetyFiveConfidenceInterval(values):
    """
    Find the 95 percent confidence interval.
    """
    mean_value = sum(values) / len(values)
    std_value = stdev(values) / math.sqrt(len(values) - 1)

    return (mean_value, 2.08*std_value)
