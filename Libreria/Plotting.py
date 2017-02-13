import numpy
from matplotlib import pyplot as plt
from Libreria.Testing import ninetyFiveConfidenceInterval

def plotFMeasure(fmeasure, id_string):
    """
    Plot precision and recall for each class
    """
    n_groups = len(fmeasure)
    plt.figure()
    index = numpy.arange(n_groups)
    opacity = 1

    plt.bar(index, fmeasure,alpha=opacity,
                     color='#3a6dc4',
                     label='F-Measure')

    plt.xlabel('Class')
    plt.ylabel('F-Measure')
    plt.title('FMeasure of each class')
    plt.xticks(index , ('1', '2', '3', '4', '5', '6', '7'))
    plt.legend()
    plt.savefig("Report/pictures/" +id_string + "_fmeasure.eps")

def plotAccuracyInterval(accuracies, id_string):
    m, std = ninetyFiveConfidenceInterval(accuracies)

    #compute the maximum and minimum value to show on the plot
    ma_v = max(accuracies)
    mi_v = min(accuracies)
    diff = ma_v - mi_v
    ma_v = ma_v + 0.2*diff
    mi_v = mi_v - 0.2*diff

    #start the plot
    plt.figure()
    #plot accuracy values for each kfold
    plt.plot(accuracies, "k+")
    #now plot the mean
    ma = [m]*(len(accuracies) + 1)
    maPlus = [m + std]
    maMinus = [m - std]
    x = [-1]
    for i in range(len(ma)):
        maPlus += [m + std]
        maMinus += [m - std]
        x += [i]

    plt.plot(x, [m]+ma, 'b-')
    plt.fill_between(x, maPlus, maMinus, color='#a0c3ff')
    plt.axis([-1, len(accuracies), mi_v, ma_v])
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend(['Accuracies', 'Mean accuracy'])
    plt.title('95 Percent confidence interval for accuracy')
    plt.savefig("Report/pictures/" + id_string + "_accuracy_interval.eps")
    return

def plotDatasetSizeVsAccuracy(sizes, accs, id_string):
    plt.figure()
    plt.plot(sizes, accs, 'bo-')
    v = (max(sizes) - min(sizes))*0.1
    b = (max(accs) - min(accs)) * 0.1
    plt.axis([min(sizes)-v, max(sizes)+v, min(accs) -b, max(accs) + b])
    plt.xlabel('Dataset size')
    plt.ylabel('Accuracy')
    plt.title('Dataset size vs Accuracy')
    plt.savefig("Report/pictures/" + id_string + "_size_vs_accuracy.eps")
