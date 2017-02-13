from Libreria.Dataset import Dataset
from Libreria.CSVLoader import readCSV, writeCSV

d = Dataset(readCSV('Data/preprocessed.csv', convertToFloat=True)).randomSplit(30)

counter = 0
for i in d:
    writeCSV(i.data, 'Data/part{}.csv'.format(counter))
    counter += 1
