from DataHandler import importDatasetAndSplit
from TestBase import startTest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from Libreria.Testing import testClassifier
from Latex import printConfusionMatrix
#load datasets
main_dataset, dataset_parts = importDatasetAndSplit()

#try different params
print("Choose different params")
feature_vectors, labels = main_dataset.subsample(0.07, 44).divideFeaturesAndLabels(44)

criterions = ['gini', 'entropy']
min_saples_splits = [2, 5, 10, 30, 50]
min_saples_leaf = [5, 7, 15, 20, 50]
max_features = [10, 15, 30, 35, 40]

best_cri = None
best_spli = 0
best_leaf = 0
best_score = -1
best_f = 0

all_scores = []
for c in criterions:
    for m in min_saples_splits:
        for j in min_saples_leaf:
            for h in max_features:

                clf = RandomForestClassifier(criterion=c, min_samples_leaf=j, max_features=h, min_samples_split=m)
                scores = cross_val_score(clf, feature_vectors, labels, cv=5)
                score = sum(scores)/len(scores)*100
                all_scores += [score]
                if score > best_score:
                    print("New best score {}".format(score))
                    best_cri = c
                    best_spli = m
                    best_leaf = j
                    best_f = h
                    best_score = score

print("Best score of {:.3f} with cri = {}, split = {}, max feature = {},  and leaf {}".format(best_score, best_cri, best_spli, best_f, best_leaf))
plt.figure()
plt.plot(all_scores, 'b-o')
plt.axis([0, len(all_scores)-1, min(all_scores)-10, max(all_scores)+10])
plt.title("Best random forest configuration")
plt.xlabel("Configuration")
plt.ylabel("Score")
plt.savefig("Report/pictures/random_forest_config.eps")
# clf = RandomForestClassifier(criterion=best_cri, max_features=best_f, min_samples_leaf=best_leaf, min_samples_split=best_spli)
# startTest(main_dataset, dataset_parts, clf,"random_forest")
