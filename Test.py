import pickle
import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print ("Chargement du modèle... \n")

clf_loaded = pickle.load(open('groupeE.pkl', 'rb'))

movieComments = pandas.read_csv('Data/test_data.csv', sep='\t', header = None, encoding = "utf8")
movieCommentsLabels = pandas.read_csv('Data/test_labels.csv', sep='\t', header = None, encoding = "utf8")

movieCommentsArray = movieComments.values
movieCommentsLabelsArray = movieCommentsLabels.values

result = clf_loaded.predict(movieCommentsArray[:, 0])

print("Accuracy:", accuracy_score(result, movieCommentsLabelsArray[:, 0]),'\n')

matrix = confusion_matrix(movieCommentsLabelsArray[:, 0], result)
print('\nMatrice de confusion: \n', matrix, "\n")

print('\n', classification_report(movieCommentsLabelsArray[:, 0], result), "\n")
