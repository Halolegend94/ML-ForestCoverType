from Libreria.Dataset import Dataset #dalla nostra libreria personale
import Libreria.CSVLoader as csv
import numpy #per gestire i dati
from sklearn import neural_network  #per il machine learning
import Libreria.DatasetAnalysis as da
from sklearn import preprocessing as pp
###################################################################

# prima di tutto, carica il file csv. Per farlo usiamo una libreria creata apposta per manipolare i
# file CSV. Il parametro convertToFloat a true converte i dati da stringa a numeri reali
integralCopy  = Dataset(csv.readCSV('covtype.csv', convertToFloat=True))

c = numpy.array(da.getInformationGain(integralCopy.data))

for row in c:
    print("{:.2f} &".format(row[0]) + "  {:.4f} &".format(row[1]) + "  {:.2f} \\\\\\hline".format(row[2]))

# #we remove the columns that have not enough info gain
# indexes = []
#
# for row in c:
#     if row[2] < 0.1:
#         indexes += [int(row[0])]
#
# newcopy = integralCopy.removeColumns(indexes)
#
# csv.writeCSV(newcopy.data, "gioco.csv")
