# Clean up the text, normalizing it
def normalize(cardText):
    if (cardText is not None):
        # Removing non ASCII characters (not needed)
        #cardText = unicodedata.normalize(
        #    'NFKD', cardText).encode("ascii", "ignore")

        # Removing contractions
        cardText = contractions.fix(cardText, slang=False)

        # Tokenizing
        tokenizedText = word_tokenize(cardText)

        # Putting all words in lowercase
        tokenizedText = [word.lower() for word in tokenizedText]

        # Deleting ponctuations (not needed)
        #tokenizedText = [word for word in tokenizedText if word.isalpha()]

        # Removing stop words (not needed)
        #tokenizedText = [
        #    word for word in tokenizedText if not word in stopwords.words('english')]

        # Lemmatization (not needed)
        #lemmatizer = WordNetLemmatizer()
        #cardText = [lemmatizer.lemmatize(word, pos = 'v') for word in tokenizedText]

        # Turning it back into a string
        cardText = "".join([" " + i for i in tokenizedText]).strip()

        return cardText

    else:  # If the text is empty
        return ""