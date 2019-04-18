import nltk
import unicodedata
import contractions
import inflect
from IPython.display import display
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

GoWords = ['not', 'nor', 'up', 'out', 'can']
global OurStopWords
OurStopWords = ['movie', 'popcorn']

for word in stopwords.words('english'):
	if GoWords.count(word) == 0:
		OurStopWords.append(word)
        
def clean_text(commentString):   
    # Removing non ASCII characters
    commentString = unicodedata.normalize('NFKD', commentString).encode("ascii", "ignore").decode("utf-8", 'ignore')

    # Removing contractions
    commentString = contractions.fix(commentString, slang = True)

    # Tokenizing
    tokenizedText = word_tokenize(commentString)

    # Putting all words in lowercase
    tokenizedText = [word.lower() for word in tokenizedText]

    # Deleting ponctuations
    tokenizedText = [word for word in tokenizedText if word.isalpha()]

    # Removing stop words
    tokenizedText = [word for word in tokenizedText if not word in OurStopWords]
    
    # Converting numbers
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
    commentString = [lemmatizer.lemmatize(word, pos = 'v') for word in tokenizedText]

    # Turning back tokens into a string
    commentString = "".join([" " + i for i in tokenizedText]).strip()
    
    return commentString
