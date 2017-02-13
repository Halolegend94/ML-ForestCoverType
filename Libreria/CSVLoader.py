"""
This module contains functions to interact with csv files.
"""
from Libreria.Dataset import Dataset
def writeCSV(data, filename):
    """
    Writes a file containing the dataset "data".
    Parameters:
        -   data: a dataset of type "Dataset"
        -   filename: name of the file to be written
    """
    f = open(filename, "w+")
    d = Dataset(data)
    f.write(str(d))
    f.close()
    return

def readCSV(filename, sep=',', convertToFloat = False):
    """
    Parses a csv dataset from a text file
    """
    myFile = open(filename, "r")
    text = myFile.read()
    myFile.close()
    currentLine = 1
    data = []
    currentRow = []
    firstRowSize = -1
    index = 0
    currentItem = ""
    while(index < len(text)):
        if (currentItem == "" and text[index] == sep):
            print("Syntax error: unexpected token \""+ str(sep) +"\" (line {})".format(currentLine))
            return

        elif text[index] == sep:
            currentRow += [currentItem]
            currentItem = ""

        elif text[index] == '\n':
            currentRow += [currentItem]
            currentItem = ""

            if firstRowSize == -1:
                #the first row sets the size of each other row
                firstRowSize = len(currentRow)

            elif len(currentRow) != firstRowSize:
                print("Syntax error: row at line {} has a different size.".format(currentLine))
                return

            currentLine += 1
            data += [currentRow]
            currentRow = []
        else:
            currentItem += text[index]
        index += 1
    if convertToFloat:
        data = __convertToFloat(data)
    return data

def __convertToFloat(data):
    """
    converts all the elements of the matrix into float numbers.
    """
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = float(data[i][j])
    return data
