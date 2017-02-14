from Libreria.Dataset import Dataset
import Libreria.CSVLoader as csv
import Libreria.DatasetAnalysis as da
from Libreria.Testing import testClassifier, kFoldCrossValidation, ninetyFiveConfidenceInterval
from Libreria.ConfusionMatrix import ConfusionMatrix
from Libreria.Plotting import  plotFMeasure, plotAccuracyInterval, plotDatasetSizeVsAccuracy
from Libreria.Latex import printTableSizeVsAccuracy, printTableAccuracies, printTablePrecisionAndRecall, printConfusionMatrix


def startTest(main_dataset, dataset_parts, clf, id_string):
    """
    Starts the test campaign.
    """
    #percentage of the dataset used as training set (used later)
    test_split = 0.37
    n_inc_datasets = 10

    print(" ========= SIMPLE TESTING ================== \n\n")
    features, labels = main_dataset.divideFeaturesAndLabels(44)
    cm = testClassifier(clf, features, labels, test_split)

    print("Stats: accuracy = {}\n".format(cm.getAccuracy()))
    print("Detailed stats for each class: ")
    n_labels = len(cm.labels)
    precision = [0]*n_labels
    recall = [0]*n_labels
    fmeasure = [0]*n_labels

    for l in cm.labels:
        precision[int(l-1)] = cm.getPrecision(l)
        recall[int(l-1)] = cm.getRecall(l)
        fmeasure[int(l-1)] = cm.getFMeasure(l)

    print(" == LAtex precision recall ===\n")
    printTablePrecisionAndRecall(precision, recall, fmeasure)

    print(" == confusion matrix === \n")
    printConfusionMatrix(cm)

    #perform kFoldCrossValidation
    cms = kFoldCrossValidation(dataset_parts, 44, clf)

    #Compute statistics resulting from kfold cross validation
    accuracies  = []
    precision = [0]*len(cms[0].labels)
    recall = [0]*len(cms[0].labels)
    fmeasure = [0]*len(cms[0].labels)

    for c in cms:
        accuracies += [c.getAccuracy()]
        for l in c.labels: #labels are numbers so we can use them as indexes
            recall[int(l-1)] += c.getRecall(l)
            precision[int(l-1)] += c.getPrecision(l)
            fmeasure[int(l-1)] += c.getFMeasure(l)

    #compute the mean now
    for i in range(len(precision)):
        precision[i] = precision[i] / len(cms)
        recall[i] = recall[i] / len(cms)
        fmeasure[i] = fmeasure[i] / len(cms)

    print("\n========== KFOLD VALIDATION ===========")
    print("\nLatex code for accuracy table\n")
    printTableAccuracies(accuracies)
    #now compute confidence interval
    m_accuracy, std_accuracy = ninetyFiveConfidenceInterval(accuracies)

    print("\n====== General stats =======\nMean accuracy: {:.3f} (95 perc. conf interval: +- {})\
        Precision: {:.3f}   Recall: {:.3f}   FMeasure: {:.3f}  ".format(m_accuracy, std_accuracy, \
        sum(precision)/len(precision), sum(recall)/len(recall), sum(fmeasure)/len(fmeasure)))

    #show a plot of confidence interval
    plotAccuracyInterval(accuracies, id_string)

    print("\n====== Detailed stats for each class ==========\n")

    print("\nName    Precision    Recall    FMeasure")
    for l in range(len(precision)):
        print("{}    {:.3f}    {:.3f}     {:.3f}\n".format(int(l+1), precision[l], recall[l], fmeasure[l]))

    print("\n==== Latex code for precision and recall =======\n")
    printTablePrecisionAndRecall(precision, recall, fmeasure)

    plotFMeasure(fmeasure, id_string)

    print("============ DATASET SIZES=========================\n")
    print("Now we test performance on various dataset sizes.")
    datasets = main_dataset.incrementalDatasets(n_inc_datasets)
    X = []
    Y = []

    for dataset in datasets:

        features, labels = dataset.divideFeaturesAndLabels(44)
        cm = testClassifier(clf, features, labels, test_split)
        print("size: {}, accuracy: {:.3f}".format(dataset.size(), cm.getAccuracy()))
        X.append(len(dataset))
        Y.append(cm.getAccuracy())

    plotDatasetSizeVsAccuracy(X, Y, id_string)

    print("=== Latex code====\n")
    printTableSizeVsAccuracy(X, Y)
