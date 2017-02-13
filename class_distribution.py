from Libreria.DatasetAnalysis import getClassDistribution
from Libreria.CSVLoader import readCSV
from Libreria.Dataset import Dataset
from matplotlib import pyplot as plt
import numpy

main_dataset = Dataset(readCSV('Data/preprocessed.csv', ',', True))

d = getClassDistribution(main_dataset, 44)

vals = [0]*7
for k in d.keys():
    vals[k-1] = d[k]
    print("class :{} - examples: {:.5f}".format(k, d[k]))

plt.bar(numpy.arange(len(vals)), vals)

plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Probability of each class')
plt.xticks(numpy.arange(len(vals)) , ('1', '2', '3', '4', '5', '6', '7'))
plt.show()
