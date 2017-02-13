import random
import math
class Dataset:
    """
    Represents a wrapper arround a matrix that represents a dataset and gives some functions to
    operate with it.
    """

    def __init__(self, data = None):
        """
        Intitializes a new instace of CSVDataset loading data from a file.
        """
        if data == None:
            data = []
        self.data = data
        return

    def getColumn(self, index):
        """
        Retuns the index-column in the dataset
        """
        if len(self.data) == 0:
            return []
        elif index < 0 or index >= len(self.data[0]):
            print("getColumn: index not valid.")
            return
        col = []
        for row in self.data:
            col.append([row[index]])
        return col

    def getRow(self, index):
        """
        Retrieves a row from the dataset.
        """
        if(index < 0 or index >= len(self.data)):
            print("getRow: index out of range.")
            return
        return self.data[index]

    def getElement(self, i, j):
        """
        Retrieves a specific element from the dataset.
        """
        if i < 0 or i >= len(self.data):
            print("getElement: index i out of range.")
            return
        elif j < 0 or j >= len(self.data[0]):
            print("getElement: index j out of range.")
        return self.data[i][j]

    def setElement(self, i, j, val):
        """
        Sets the value for a specific element from the dataset.
        """
        if i < 0 or i >= len(self.data):
            print("getElement: index i out of range.")
            return
        elif j < 0 or j >= len(self.data[0]):
            print("getElement: index j out of range.")
        self.data[i][j] = val
        return

    def createCopy(self):
        """
        Creates a new CSVDataset object that is a copy of the CSVDataset object passed as argument.
        """
        if self.data == None:
            return Dataset()

        data = []
        currRow = []
        for row in self.data:
            for element in row:
                currRow += [element]
            data += [currRow]
            currRow = []
        return Dataset(data)

    def appendDataset(self, dataset):
        """
        Appends to the current dataset data from another CSVDataset
        """
        if(dataset == None):
            print("appendDataset: dataset was NoneType.")
            return
        dataToAppend = dataset.data

        if len(self.data) > 0 and len(dataToAppend[0]) != len(self.data[0]):
            print("appendDataset: datasets differs in number of columns.")
            return
        self.data += dataToAppend
        return self

    def addRow(self, row, pos = None):
        """
        Adds a new row to the current dataset. If "pos" is not specified, row will be added as last
        element, otherwise it will be inserted at "pos" position
        """
        datLen = len(self.data)
        if datLen == 0:
            #this is the first row
            self.data += [row]
        elif len(row) == len(self.data[0]):
            if pos == None:
                self.data += [row]
            elif pos >= 0 and pos < datLen:
                self.data.insert(pos, row)
            else:
                print("addRow: pos not in valid range.")
        else:
            print("addRow: row size doesn't match.")
        return self

    def removeRows(self, indexes):
        """
        Removes specified rows from the dataset.
        Params:
            - indexes: indexes of the rows to be removed.
        """
        indexes = sorted(indexes)[::-1]
        if len(self.data) == 0:
            return self
        for index in indexes:
            if(index < 0 or index >= len(self.data)):
                print("removeRow: index out of range.")
                return
            del self.data[index]
        return self

    def removeColumns(self, indexes):
        """
        Removes specified rows from the dataset.
        Params:
            - indexes: index of the rows to be removed.
        """
        indexes = sorted(indexes)[::-1]
        if len(self.data) == 0:
            print("removeColumn: dataset is empty.")
            return

        if max(indexes) >= len(self.data[0]) or min(indexes) < 0:
            print("Indexes not valid.")
            return

        for i in range(len(self.data)):
            for j in indexes:
                del self.data[i][j]
        return self

    def subsample(self, N, class_index):
        """
        Returns a smaller dataset with almost the same class distribution.
        - N, between 0 and 1, specifies the size of the sample dataset with respect to the
        integral one.
        - class_index: is the index of the class attribute.
        """
        N = math.floor(len(self.data) * N)
        #First, go through the database and partitions
        partition = dict()
        for row in self.data:
            c = row[class_index]
            if not (c in partition.keys()):
                partition[c] = [row]
            else:
                partition[c] += [row]

        newDataset = Dataset()
        for c in partition.keys():
            #compute the number of instances to insert in the new Dataset
            samples = partition[c]
            perc = len(samples) / self.size()
            numb = math.floor(N*perc)
            for i in range(numb):
                newDataset.addRow(samples[i])

        return newDataset

    def randomSplit(self, N):
        """
        Splits the set in N subsets with approximately the same number of elements. The splits are
        generated randomly.
        """
        newDatasets = list()
        for i in range(N):
            newDatasets.append(Dataset())

        indexes = list(range(len(self.data)))
        currentSet = 0
        indexesSize = len(indexes)
        while(indexesSize > 0):
            rowIndex = random.randint(0, indexesSize-1)
            selectedRow = indexes[rowIndex]
            newDatasets[currentSet].addRow(self.data[selectedRow])
            del indexes[rowIndex]
            indexesSize -= 1
            currentSet = (currentSet + 1) % N

        return newDatasets

    def incrementalDatasets(self, N):
        """
        From a single datasets, create new datasets of with incremental sizes.
        """
        datasets = self.randomSplit(N)
        incDatasets = []
        for d in datasets:
            if len(incDatasets) == 0:
                incDatasets += [d]
            else:
                last_d = incDatasets[-1]
                incDatasets += [last_d.createCopy().appendDataset(d)]
        return incDatasets

    def split(self, N):
        """
        Splits the set in N subsets with approximately the same number of elements.
        """
        newDatasets = list()
        for i in range(N):
            newDatasets.append(Dataset())

        dataSize = len(self.data)
        currentSet = 0
        setSize = math.floor(dataSize / N)
        dataSize -= 1

        while(dataSize >= 0):
            currentSet = dataSize % N
            newDatasets[currentSet].addRow(self.data[dataSize])
            dataSize -= 1
        return newDatasets

    def size(self):
        """
        Returns the size of the dataset.
        """
        return len(self.data)

    def divideFeaturesAndLabels(self, class_index):
        etichette = []
        features = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data[0])):
                if(j != class_index):
                    row += [self.data[i][j]]
                else:
                    etichette += [self.data[i][j]]
            features += [row]
        return features, etichette

    def numColumns(self):
        """
        Returns the number of colums in the dataset.
        """
        return len(self.data[0])


    def __len__(self):
        """
        Returns the number of instances in the dataset.
        """
        return len(self.data)

    def __str__(self):
        """
        Returns a string representation of the dataset
        """
        strin = ""
        for row in self.data:
            for element in row[0:-1]:
                strin += "{},".format(element)
            strin += "{}\n".format(row[-1])
        return strin[0:-1]
