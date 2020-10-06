import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

def extractParas(text):
	paragraphs = []
	para = ""
	i = 0
	while i < len(text):
		if i < len(text) and text[i] == '\n':
			if para != "":
				paragraphs.append(para)
			para = ""
		else:
			para += text[i]
		i += 1
	if para != "\n" and para != "":
		paragraphs.append(para)

	return paragraphs

def tokenize(text):
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)

	#Convert to lower case
	for i, token in enumerate(tokens):
		tokens[i] = token.lower()
	return tokens

def stopwordRemoval(tokens):
	#Remove stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	return tokens

def stemmer(tokens):
	#Perform stemming
	stemmer = PorterStemmer()
	for i, token in enumerate(tokens):
		tokens[i] = stemmer.stem(token)
	return tokens

if __name__ == '__main__':
	documents=[]
	files = os.listdir('TRAIN')
	for i,file in enumerate(files):
		with open('TRAIN/' + file, encoding="utf8", errors='ignore') as f:
			data = f.read()
			paras = extractParas(data)
			paragraphs = []
			for para in paras:
				###preprocessing###
				tokens = tokenize(para)
				stoplesstokens = stopwordRemoval(tokens)
				finalpara = stemmer(stoplesstokens)
				###
				paragraphs.append(finalpara)
			documents.append(paragraphs)
	