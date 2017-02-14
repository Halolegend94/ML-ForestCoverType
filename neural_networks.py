from DataHandler import importDatasetAndSplit
from TestBase import startTest
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
#load datasets
main_dataset, dataset_parts = importDatasetAndSplit()

#parameters
epoch = 300

# Some further test and parameter tuning specific for neural networks
#
# Step one: choose the number of neurons in the hidden layer. To do so, we use a subsample
# of the entire dataset.
feature_vectors, labels= main_dataset.subsample(0.05, 44).divideFeaturesAndLabels(44)
hidden_layers = [10, 15, 18, 20, 23, 25, 27, 30, 35, 40, 45]

best = -1
best_hl = 0
scores = []

print("Hidden layer that will be tested: {}".format(hidden_layers))
for hl in hidden_layers:
    clf = MLPClassifier(hidden_layer_sizes=(hl), max_iter=epoch)
    score = cross_val_score(clf, feature_vectors, labels, cv=5)
    mean_val = sum(score) / len(score)
    scores += [mean_val]
    if mean_val > best:
        best = mean_val
        best_hl = hl

print("Step 1: results\n# Hidden layers    score")
for i in range(len(hidden_layers)):
    print("{}                 {:.2f}".format(hidden_layers[i], scores[i]))

#show graphics
plt.plot(hidden_layers, scores, "bo-")
plt.xlabel('Number of neurons in hidden layer')
plt.ylabel('Accuracy')
plt.savefig("Report/pictures/nn_hidden_layers_scores.eps")

#Step 2: now lets test on the full dataset with kfold cross validation

clf = MLPClassifier(hidden_layer_sizes=(40), max_iter=epoch)
startTest(main_dataset, dataset_parts, clf, "nn" )
