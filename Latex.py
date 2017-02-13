"""
This module contains functions to generate latex code to display data (using tables and plots)
"""
from Libreria.ConfusionMatrix import ConfusionMatrix

def printTableAccuracies(accuracies):
    print("\\begin{table}[h!]")
    print("\\centering\n\\begin{tabular}{|l|l|}\n\hline")
    print("\\textbf{Fold Index} & \\textbf{Accuracy}\\\\\hline")
    for i in range(len(accuracies)):
        print("{} & {:.3f}\\\\\hline".format(i, accuracies[i]))
    print("\end{tabular}\n\end{table}")
    return

def printTablePrecisionAndRecall(precision, recall, fmeasure):
    print("\\begin{table}[h!]")
    print("\\centering\n\\begin{tabular}{|l|l|l|l|}\n\hline")
    print("\\textbf{Class Index} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F-Measure}\\\\\hline")
    for i in range(len(precision)):
        print("{} & {:.3f} & {:.3f}& {:.3f}\\\\\hline".format(i+1, precision[i], recall[i], fmeasure[i]))
    print("\end{tabular}\n\end{table}")

def printTableSizeVsAccuracy(sizes, accs):
    print("\\begin{table}[h!]")
    print("\\centering\n\\begin{tabular}{|l|l|}\n\hline")
    print("\\textbf{Dataset Size} & \\textbf{Accuracy}\\\\\hline")
    for i in range(len(accs)):
        print("{} & {:.3f}\\\\\hline".format(sizes[i], accs[i]))
    print("\end{tabular}\n\end{table}")

def printConfusionMatrix(cm):
    #compute first the percentage form of comfusion matrix
    perc_matrix = []
    for i in range(len(cm.confusionMatrix)):
        row = []
        s = sum(cm.confusionMatrix[i])
        for j in range(len(cm.confusionMatrix[0])):
            row += [(cm.confusionMatrix[i][j] / s)*100]
        perc_matrix += [row]

    print("\\begin{table}\n\centering\n\\begin{tabular}{|*{" + str(len(cm.labels))+ "}{c|}r|}\n\hline\n")
    for i in range(len(cm.indexesLabel)-1):
        print("{:d} &".format(int(cm.indexesLabel[i])), end="")
    print("{:d} & classified as \\\\\hline\n".format(int(cm.indexesLabel[len(cm.indexesLabel)-1])))
    for i in range(len(cm.indexesLabel)):
        for j in range(len(cm.indexesLabel)-1):
            cl=""
            if i == j:
                cl = "\cellcolor{black!15}"
            print(cl + "{:.2f} &".format(perc_matrix[i][j]), end="")
        cl=""
        j = len(cm.indexesLabel)-1
        if i == j:
            cl = "\cellcolor{black!15}"
        print(cl + "{:.2f} & {:d} \\\\\hline\n".format(perc_matrix[i][j],int(cm.indexesLabel[i])))
    print("\end{tabular}\n\end{table}")
