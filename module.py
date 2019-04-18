import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import nltk
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
import unicodedata
import contractions
import re
import inflect
import pickle
from nltk.corpus import stopwords

GoWords = ['not', 'nor', 'up', 'out', 'can']
global OurStopWords
OurStopWords = ['movie', 'popcorn']

for word in stopwords.words('english'):
	if GoWords.count(word) == 0:
		OurStopWords.append(word)
		pass
	pass

def remove_non_ascii(words):
	new_words = []
	for word in words:
		new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		new_words.append(new_word)
	return new_words

def remove_contraction(words):
	return contractions.fix(words, slang=True)

def to_lowercase(words):
	new_words = []
	for word in words:
		new_word = word.lower()
		new_words.append(new_word)
	return new_words

def remove_punctuation(words):
	new_words = []
	for word in words:
		new_word = re.sub(r'[^\w\s]', '', word)
		if new_word != '':
			new_words.append(new_word)
	return new_words

def replace_numbers(words):
	p = inflect.engine()
	new_words = []
	for word in words:
		if word.isdigit():
			new_word = p.number_to_words(word)
			new_words.append(new_word)
		else:
			new_words.append(word)
	return new_words

def remove_stopwords(words):
	new_words = []
	for word in words:
		if word not in stopwords.words('english'):
			new_words.append(word)
	return new_words

def lemmatizee(words):
	new_words = []
	lemmatizer = WordNetLemmatizer()
	return [lemmatizer.lemmatize(word, pos='v') for word in words]


def normalize(words):
	words = remove_non_ascii(words)
	words = to_lowercase(words)
	words = replace_numbers(words)
	words = remove_punctuation(words)
	words = remove_stopwords(words)
	words = lemmatizee(words)
	return words

def delete_words(words):
	tags = pos_tag(words)
	#print(tags)
	new_words = []
	for tag in tags:
		if(tag[1]=="VB" 
			or tag[1] == "NN" 
			or tag[1] == "VBP" 
			or tag[1] == "NNS" 
			or tag[1] == "VBD" 
			or tag[1] == "VBZ"
			or tag[1] == "JJ"):
			del words[words.index(tag[0])]
	#print(pos_tag(words))
	return words
	

def clean_text(text):
	text = remove_contraction(text)
	tokens = word_tokenize(text)
	tokens = normalize(tokens)
	#tokens = delete_words(tokens)
	text="".join([" "+i for i in tokens]).strip()
	return text