"""
Name: DatasetAnalysis

This library contains a set of functions that operate on a dataset in order to perform some kind
of analysis.
"""
import math
import numpy as np

def getEntropy(dataset, class_col=-1):
    """
    Returns the entropy of the dataset.
    """
    n_rows = len(dataset)  #number of examples
    if n_rows == 0:
        print("getEntropy: the dataset is empty.")
        return

    cl_freq = dict()                     #class frequencies

    for r in range(n_rows):              #for each example
        k = dataset[r][class_col]        #get the class and update the frequency
        if k not in cl_freq.keys():
            cl_freq[k] = 1
        else:
            cl_freq[k] = cl_freq[k] + 1

    Entropy = 0
    for key in cl_freq.keys():
        p = cl_freq[key] / n_rows
        Entropy += (p * math.log2(p))
    Entropy = -Entropy

    #coumpute the maximum entropy possible
    mp = 1/len(cl_freq.keys()) #all the classes have the same probability
    maxEntr = -(mp*math.log2(mp)*len(cl_freq.keys()))
    #so now we can express entropy as a percentage
    if maxEntr > 0:
        EntropyPercentage = (Entropy * 100) / maxEntr
    else:
        EntropyPercentage = 0
    return (Entropy, EntropyPercentage)

def getInformationGain(dataset, class_col=-1):
    """
    Returns an array of elements, where each element is an array whose first element is the index
    of an attribute, the second is the absolute value of its info gain, the this reports the info gain
    as a percentage. Elements are returned in descending order.
    """
    n_rows = len(dataset)
    if n_rows > 0:
        n_cols = len(dataset[0])
    else:
        print("getInformationGain: the dataset is empty.")
        return []

    E = getEntropy(dataset, class_col)[0]

    #this dictionary will contain the information gain of each attribute
    ig = []
    #for each attribute
    for i in range(n_cols):
        #for each attribute value, we need to compute the entropy of the dataset's subset that
        #present that value for feature i
        att_vals = dict()
        for r in range(n_rows):
            subset = att_vals.get(dataset[r][i])
            if subset == None:
                subset = []
            subset += [dataset[r]]
            att_vals[dataset[r][i]] = subset

        val = 0
        for key in att_vals.keys():
            subset = att_vals[key]
            e = (len(subset) / n_rows) * getEntropy(subset, class_col)[0]
            val += e

        ig += [[i, (E -val), (E - val)/E * 100]] #attribute index, info gain, info gain percentage

    ig = np.array(ig)
    ig = ig[ig[:, 2].argsort()[::-1]] #sort in descending order
    return ig

def __parseIndexes(indexes):
    """
    This function is used to parse the indexes param in discretize function.
    """
    ind = []
    for i in indexes:
        if type(i) is str:
            for t in range(int(i[0]), int(i[-1]) + 1):
                ind += [t]
        else:
            ind += [int(i)]
    return ind

def discretize(dataset, indexes = None, nbins = 10):
    """
    Returns the dataset with discretized values for the columns specified in indexes.
    Params:
        -   indexes: an array that contains indexes and/or ranges eg [1, 2, "4-9"]
        -   nbins: the number of bins in which we want to partition the data
    """
    n_rows = len(dataset)
    if n_rows == 0:
        print("discretize: the dataset is empty.")
        return
    myIndexes = __parseIndexes(indexes)
    n_cols = len(dataset[0])
    if n_cols <= max(myIndexes):
        print("discretize: index out of bound.")
        return
    dataset = np.array(dataset)
    print(dataset)
    for ind in myIndexes:
        print(ind)
        v_max = max(dataset[:, ind]) +  1
        v_min = min(dataset[:, ind])

        d = (v_max - v_min) / (nbins)

        for r in range(n_rows):
            v = dataset[r][ind]
            mybin = math.ceil(v / d)
            dataset[r][ind] = mybin
    return dataset

def getClassDistribution(dataset, class_index):
    """
    Returns the distribution over the classes.
    """
    if (len(dataset)) == 0:
        print("getClassDistribution: dataset is empty.")
        return
    if class_index >= len(dataset.data[0]):
        print ("getClassDistribution: class_index not valid.")
        return None
    classes = dict()

    for row in dataset.data:
        if not(int(row[class_index]) in classes.keys()):
            classes[int(row[class_index])] = 1
        else:
            classes[int(row[class_index])] += 1
    for k in classes.keys():
        classes[k] /= len(dataset)
    return classes
