from DataHandler import importDatasetAndSplit
from TestBase import startTest
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

#load datasets
main_dataset, dataset_parts = importDatasetAndSplit()

# Now we try different parameters to have the best configuration.
# To do so, we use a subsample of the dataset.
# Parameters to test:
C = [0.5, 1, 2, 3, 4, 6, 8]
Knl = [2]

feature_vectors, labels = main_dataset.subsample(0.03, 44).divideFeaturesAndLabels(44)

best_c = 0
best_k = 0
best_score = -1
all_scores = []

for c in C:
    for d in Knl:
        clf = SVC(C=c, degree=d)
        scores = cross_val_score(clf, feature_vectors, labels, cv=2)
        score = sum(scores) / len(scores) * 100
        all_scores += [score]
        if score > best_score:
            best_score = score
            best_c = c
            best_k = d

plt.figure()
plt.plot(all_scores, "bo-")
plt.title("Best parameters search")
plt.xlabel("Configuration")
plt.ylabel("Score")
v = (max(all_scores) - min(all_scores))*0.1
plt.axis([0, 8, min(all_scores)-v, max(all_scores) + v])
plt.savefig("Report/pictures/svm_config.eps")
print("Best score: {} with d = {} and C = {}".format(best_score, best_k, best_c))
clf = SVC(degree=2, C=best_c)
startTest(main_dataset, dataset_parts, clf, "svm")
