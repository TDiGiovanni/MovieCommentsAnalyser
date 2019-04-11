import sklearn
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from IPython.display import display
import contractions
import unicodedata
import nltk
import pandas
import inflect

movieComments = pandas.read_csv(
    'Data/dataset.csv', sep='\t', header=None, encoding="utf8")
movieCommentsLabels = pandas.read_csv(
    'Data/labels.csv', sep='\t', header=None, encoding="utf8")


nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')


def cleanText(commentString):
    # Removing non ASCII characters
    # commentString = unicodedata.normalize('NFKD', commentString).encode("ascii", "ignore")

    # Removing contractions
    commentString = contractions.fix(commentString, slang=True)

    # Tokenizing
    tokenizedText = word_tokenize(commentString)

    # Putting all words in lowercase
    tokenizedText = [word.lower() for word in tokenizedText]

    # Deleting ponctuations
    tokenizedText = [word for word in tokenizedText if word.isalpha()]

    # Removing stop words
    tokenizedText = [
        word for word in tokenizedText if not word in stopwords.words('english')]

    # Converting numbers into text
    inflectEngine = inflect.engine()
    newWords = []
    for word in tokenizedText:
        if word.isdigit():
            newWords.append(inflectEngine.number_to_words(word))
        else:
            newWords.append(word)
    tokenizedText = newWords

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    commentString = [lemmatizer.lemmatize(
        word, pos='v') for word in tokenizedText]

    # Turning it back into a string
    commentString = "".join([" " + i for i in tokenizedText]).strip()

    return commentString


# Sickit learn met régulièrement à jour des versions et indique des futurs warnings.
# Ces deux lignes permettent de ne pas les afficher.
warnings.filterwarnings("ignore", category=FutureWarning)

movieCommentsArray = movieComments.values
data = movieCommentsArray[4500:5500, 0]  # X

movieCommentsLabelsArray = movieCommentsLabels.values
dataLabels = movieCommentsLabelsArray[4500:5500, 0]  # Y

print("Pre-processing... \n")
for i in range(len(data)):
    data[i] = cleanText(data[i])

vectorizer = TfidfVectorizer(ngram_range = (1, 2))
vectors = vectorizer.fit_transform(data)

data = vectors.toarray()

trainingSize = 0.7
testingSize = 0.3

trainingData, testingData, trainingDataLabels, testingDataLabels = train_test_split(data, dataLabels, train_size=trainingSize, test_size=testingSize)
# X_train,    X_test,      Y_train,            Y_test


# GRIDSEARCH

classifiers = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'SVC': SVC()
    }

parameters = {
    'KNeighborsClassifier': [
        {'n_neighbors': list(range(1,15))},
        {'metric': ['minkowski', 'euclidean', 'manhattan']}
        ],

    'DecisionTreeClassifier': [
        {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {'criterion': ['gini', 'entropy']},
        {'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        ],

    'SVC': [
        {'C': [0.001, 0.01, 0.1, 1, 10]},
        {'gamma': [0.001, 0.01, 0.1, 1]},
        {'kernel': ['linear', 'rbf']}
    ]
}

class Result:
    def __init__(self, name, score, parameters):
        self.name = name
        self.score = score
        self.parameters = parameters

    def __repr__(self):
        return repr((self.name, self.score, self.parameters))

results = []
print("Performing grid search...")
for key, value in classifiers.items():
    gridSearch = GridSearchCV(
        estimator = value,
        param_grid = parameters[key],
        scoring = 'accuracy',
        cv = 5,
        n_jobs = -1,
        iid = True)

    gridSearch.fit(trainingData, trainingDataLabels)

    result = Result(key, gridSearch.best_score_, gridSearch.best_estimator_)
    results.append(result)

results = sorted(results, key = lambda result: result.score, reverse = True)

print('Results from best to worst: \n')
for result in results:
    print ("Classifier: ", result.name,
    " score %0.2f " %result.score,
    "with ", result.parameters,'\n')


# PIPELINE

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

print ('Creating pipeline... \n')
pipeline = Pipeline([('preprocessing', cleanText()),
    ("vectorizer", TfidfVectorizer(ngram_range = (1, 2))),
    ('classifier', classifiers[results[0].name])])

# SAUVEGARDE MODELE

import pickle

print("Saving model... \n")
pickle.dump(results[0], open('groupeE.pkl', 'wb'))