from Libreria.Dataset import Dataset
import Libreria.CSVLoader as csv

def importDatasetAndSplit():
    #first of all load data in memory
    main_dataset = Dataset(csv.readCSV('Data/preprocessed.csv', ',', True))

    #the main dataset has been splitted in 7 parts for cross validation. These parts have been saved
    #on disk because will be used to do a comparation amongst classifiers (so we want the same split)
    dataset_parts = [None]*30
    for i in range(30):
        dataset_parts[i] = Dataset(csv.readCSV('Data/part{}.csv'.format(i), ',', True))

    return (main_dataset, dataset_parts)
