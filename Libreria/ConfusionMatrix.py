class ConfusionMatrix:

    def __init__(self, truth, predicted):
        '''
        truth is an array of labels where the i-th element represents the ground truth
        for the i-th sample. The i-th element in "predicted" is the label assigned to
        the sample by a classifier. With this informations we build a confusion matrix.
        '''

        #first, find all the class labels
        s_labels = set()
        for i in truth:
            s_labels.add(i)
        for i in predicted:
            s_labels.add(i)

        #now we assign an index to each class label
        #we will need a way to recover one given the other
        counter = 0
        self.indexesLabel = list()
        self.labelIndexes = dict()
        for label in s_labels:
            self.labelIndexes[label] = counter
            self.indexesLabel += [label]
            counter += 1

        n_labels = len(s_labels)
        self.confusionMatrix = []
        for i in range(n_labels):
            self.confusionMatrix += [[0]*n_labels]

        #now compute the confusion matrix. Meanwhile we can compute the accuracy
        #of the classifier
        self.accuracy = 0;
        for i in range(len(truth)):
            if truth[i] == predicted[i]:
                self.accuracy += 1

            j = self.labelIndexes[truth[i]]
            k = self.labelIndexes[predicted[i]]
            self.confusionMatrix[j][k] += 1

        self.accuracy = (float(self.accuracy) / len(truth))*100
        self.labels = s_labels
        return

    def getAccuracy(self):
        '''
        returns the percentage of correctly classified instances.
        '''
        return self.accuracy

    def getError(self):
        '''
        returns the percentage of misclassified instances.
        '''
        return 100 - self.accuracy

    def getPrecision(self, attrName):
        #first, get the index of the attribute
        ind = self.labelIndexes[attrName]
        TP = self.confusionMatrix[ind][ind]
        TPFP = 0
        for i in range(len(self.confusionMatrix)):
            TPFP += self.confusionMatrix[i][ind]
        if TPFP == 0:
            return 0
        else:
            return (TP/TPFP)*100

    def getRecall(self, attrName):
        #first, get the index of the attribute
        ind = self.labelIndexes[attrName]
        TP = self.confusionMatrix[ind][ind]
        TPFN = 0
        for i in range(len(self.confusionMatrix)):
            TPFN += self.confusionMatrix[ind][i]
        if TPFN == 0:
            return 0
        else:
            return (TP/TPFN)*100

    def getFMeasure(self, attrName):
        p = self.getPrecision(attrName)
        r = self.getRecall(attrName)
        if p + r == 0:
            return 0
        else:
            return 2*((p*r)/(p+r))

    def getMeanPrecision(self):
        val = 0
        for label in self.labels:
            val += self.getPrecision(label)
        return (val / len(self.labels))

    def getMeanRecall(self):
        val = 0
        for label in self.labels:
            val += self.getRecall(label)
        return (val / len(self.labels))

    def __str__(self):
        s = ""
        n = len(self.labels)
        for row in self.confusionMatrix:
            for i in range(n):
                s+= " ---"
            s+="\n|"
            for val in row:
                s+=" {} |".format(val)
            s+="\n"
        for i in range(n):
            s+= " ---"
        s+="\n"
        return s
