from DataHandler import importDatasetAndSplit
from TestBase import startTest
from sklearn.naive_bayes import GaussianNB, MultinomialNB

#load datasets
main_dataset, dataset_parts = importDatasetAndSplit()

#create the classifiers and test them
clf = GaussianNB()
startTest(main_dataset, dataset_parts, clf, "nb_gaussian")

print("\n ================== MULTINOMIAL ================== \n")

clf = MultinomialNB()
startTest(main_dataset, dataset_parts, clf, "nb_multi")
