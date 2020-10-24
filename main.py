import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import sys
import math
import numpy as np

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
	vocab = set()
	files = os.listdir('TRAIN')
	for i,file in enumerate(files):
		with open('TRAIN/' + file, encoding="utf8", errors='ignore') as f:
			data = f.read()
			paras = extractParas(data)
			paragraphs = []
			for j,para in enumerate(paras):
				###preprocessing###
				tokens = tokenize(para)
				stoplesstokens = stopwordRemoval(tokens)
				finaltokens = stemmer(stoplesstokens)
				###
				paragraphs.append(finaltokens)
				for term in finaltokens:
					vocab.add(term)
			documents.append(paragraphs)
	vocablen = len(vocab)

	#creating the inverted index
	# indexer = Indexer(documents)

	inverted_index = {}
	for i,doc in enumerate(documents):
		for j, para in enumerate(doc):
			for word in para:
				# print(word)
				if inverted_index.get(word,False):
					if i not in inverted_index[word]:
						inverted_index[word].append(i)
				else:
					inverted_index[word] = [i]

	# print(len(inverted_index))

	doclen = [] #length of the docs
	totcount = 0
	for doc in documents:
		totcount = 0
		for para in doc:
			totcount += len(para)
		doclen.append(totcount)

	tf_idf = {}
	for word in vocab:
		for i,doc in enumerate(documents):
			if i in inverted_index[word]:
				for para in doc:
					for w in para:
						if(w == word):
							count+=1
				# count/=doclen[i]
			else:
				count = 0

			if (i != 0):
				tf_idf[word].append(count)
			else:
				tf_idf[word] = [count]
			count = 0
	#tf_idf currently stores only tf values for words in dictionary format

	#convert to 1 + log(tf)
	for word in vocab:
		tf_idf[word] = [(1 + math.log(x+1)) for x in tf_idf[word]]

	#add idf weighting
	totaldocs = len(documents)
	for word in vocab:
		idfval = math.log(totaldocs/len(inverted_index[word]))
		tf_idf[word] = [round(x * idfval, 3) for x in tf_idf[word]]

	
	# print(tf_idf["inherit"]) #to check whether index is properly created

	#take testfile name from cl arguments
	# testDocument = str(sys.argv[1])