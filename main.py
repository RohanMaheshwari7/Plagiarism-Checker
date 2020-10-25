import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import sys
import math
import numpy as np
import time
#ensure you have Python 3.6+

def extractParas(text):
	"""
	Extract paras from the raw text provided

	Parameters:  
	str: raw text data from the input file  

	Returns:  
	list: List of paragraphs from the input document  
	"""

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
	"""
	Tokenize the given paragraph

	Parameters:  
	list: List of paragraphs from the input document  

	Returns:  
	list: List of tokens from the input paragraph  
	"""

	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)

	#Convert to lower case
	for i, token in enumerate(tokens):
		tokens[i] = token.lower()
	return tokens

def stopwordRemoval(tokens):
	"""
	Remove common English stopwords from the tokens
	
	Parameters:  
	list: List of tokens  

	Returns:  
	list: List of tokens without the stop words  
	"""

	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	return tokens

def stemmer(tokens):
	"""
	Stemming function

	Parameters:  
	list: List of tokens without stopwords  

	Returns:  
	list: List of stemmed tokens  
	"""
	stemmer = PorterStemmer()
	for i, token in enumerate(tokens):
		tokens[i] = stemmer.stem(token)
	return tokens

def createInvertedIndex(documents):
	"""
	Creating the inverted index

	Parameters:  
	documents(list): Preprocessed list containing content of the documents  

	Returns:  
	Dictionary: The inverted index  
	&nbsp;&nbsp; Key: word  
	&nbsp;&nbsp; Value: posting list  
	"""

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
	return inverted_index


def calculate_TfIdf_Weights(vocab, inverted_index, documents):
	"""
	Calculate the tfidf scores 

	Parameters:  
	(arg1)vocab(set): Vocabulary of training data  
	(arg2)inverted_index(dict): Inverted Index  
	(arg3)documents(list): Preprocessed list containing content of the documents  


	Returns:  
	Dictionary: TfIdf scores  
	&nbsp;&nbsp; Key: word  
	&nbsp;&nbsp; Value: list of tf * idf scores for all the documents  
	"""
	
	tf_idf = {}
	count = 0
	for word in vocab:
		for i,doc in enumerate(documents):
			if i in inverted_index[word]:
				for para in doc:
					for w in para:
						if(w == word):
							count+=1
			else:
				count = 0

			if (i != 0):
				tf_idf[word].append(count)
			else:
				tf_idf[word] = [count]
			count = 0

	#convert to 1 + log(tf)
	for word in vocab:
		tf_idf[word] = [(1 + math.log(x+1)) for x in tf_idf[word]]

	#add idf weighting
	totaldocs = len(documents)
	for word in vocab:
		idfval = math.log(totaldocs/len(inverted_index[word]))
		tf_idf[word] = [round(x * idfval, 3) for x in tf_idf[word]]

	return tf_idf


def rankDocsByMatchingSimilarity(documents, testparagraphs, tf_idf):
	"""
	Calculate the ranking of the docs wrt matching similarity to testdoc 

	Parameters:  
	(arg1)documents(list): Preprocessed list containing content of the documents  
	(arg2)testparagraphs(list): paragraphs extracted from test document  
	(arg3)tf_idf(dict): TfIdf scores  
	&nbsp;&nbsp; Key: word  
	&nbsp;&nbsp; Value: list of tf * idf scores for all the documents  

	Returns:  
	Dictionary: Ranking  
	&nbsp;&nbsp; Key: doc number  
	&nbsp;&nbsp; Value: Similarity score  
	"""

	ranking = {} 

	for i,doc in enumerate(documents):
		ranking[i] = 0
		for j,para in enumerate(testparagraphs):
			for word in para:
				ranking[i] += tf_idf[word][i]
		ranking[i] = round(ranking[i],3)
	
	#sort ranking dict in reverse order by keys
	ranking = {k: v for k, v in sorted(ranking.items(),reverse=True, key=lambda item: item[1])}
	return ranking


def calculateQueryWeights(testvocab, inverted_index, testparagraphs, totaldocs):
	"""
	Calculate weight of the query passed - here query is a document 

	Parameters:  
	arg1: testvocab(set): Preprocessed list containing content of the documents  
	arg2: inverted_index(dict): Inverted Index  
	arg3: testparagraphs(list): paragraphs extracted from test document  
	arg4: totaldocs(int): Number of documents in Training corpus  

	Returns:  
	weights(dictionary)  
	&nbsp;&nbsp; Key: word in test document vocabulary  
	&nbsp;&nbsp; Value: TfIdf score  
	"""
	weights = {}
	
	for word in testvocab:
		count = 0
		for para in testparagraphs:
			for w in para:
				if(w == word):
					count += 1
		weights[word] = count

	for word in testvocab:
		weights[word] = 1 + math.log(weights[word]+1)

	
	for word in testvocab:
		if word not in inverted_index:
			inverted_index[word] = []
			tf_idf[word] = [0]*totaldocs
		idfval = math.log((totaldocs+1)/(len(inverted_index[word])+1))
		
		weights[word] = round(weights[word] * idfval, 3)

	return weights

def calculateParaWeights(para, vocab, inverted_index, totaldocs):
	"""
	Calculate weight of the query passed - here query is a paragraph 

	Parameters:  
	arg1: para(list): List of tokens in the query paragraph  
	arg2: testvocab(set): Preprocessed list containing content of the documents  
	arg3: inverted_index(dict): Inverted Index  
	arg4: totaldocs(int): Number of documents in Training corpus  

	Returns:  
	weights(numpy array) - contains weights of the words in the test para vocabulary for the passed para    
	"""

	weights = np.zeros(len(vocab))
	for i,word in enumerate(vocab):
		count = 0
		for w in para:
			if(w == word):
				count += 1
		weights[i] = count

		for i in range(len(vocab)):
			if(weights[i] != 0):
				weights[i] = 1 + math.log(weights[i]+1)

		for i,word in enumerate(vocab):
			if word not in inverted_index:
				inverted_index[word] = []
				tf_idf[word] = [0]*totaldocs
			idfval = math.log((totaldocs+1)/(len(inverted_index[word])+1))
			weights[i] = round(weights[i] * idfval, 3)
	return weights


def rankDocsByCosineSimilarity(documents, testparagraphs, tf_idf, inverted_index):
	"""
	Calculate the ranking of the docs wrt cosine similarity to testdoc 

	Parameters:  
	arg1: documents(list): Preprocessed list containing content of the documents  
	arg2: testparagraphs(list): paragraphs extracted from test document  
	arg3: tf_idf(dict): TfIdf scores  
	&nbsp;&nbsp; Key: word  
	&nbsp;&nbsp; Value: list of tf * idf scores for all the documents  
	arg4: inverted_index(dict): Inverted Index  

	Returns:  
	Dictionary: Ranking  
	&nbsp;&nbsp; Key: doc number  
	&nbsp;&nbsp; Value: Cosine Similarity score  
	"""
	ranking = {}
	testvocab = set()
	for para in testparagraphs:
		for word in para:
			testvocab.add(word)
	testdoc = []
	testdoc.append(testparagraphs)
	query_tfidf = calculateQueryWeights(testvocab, inverted_index, testparagraphs, len(documents))
	a = np.zeros(len(testvocab))
	b = np.zeros(len(testvocab))

	# print(query_tfidf)
	
	for j,doc in enumerate(documents):
		i = 0
		for word in testvocab:
			if word in inverted_index and j in inverted_index[word]:
				a[i] = tf_idf[word][j]
			else:
				a[i] = 0
			b[i] = query_tfidf[word]
			i += 1
		ranking[j] = cosine_sim(a,b)

	#sort ranking dict in reverse order by keys
	ranking = {k: v for k, v in sorted(ranking.items(),reverse = True, key=lambda item: item[1])}
	# print(ranking)
	return ranking


def cosine_sim(a,b):
	"""
	Calculate cosine of two vectors
	
	Parameters:  
	arg1: a(list) - 1st vector
	arg2: b(list) - 2nd vector

	Return:
	cos_sim(float) - Computed cosine
	"""

	cos_sim = 0
	if np.linalg.norm(a)!=0 and np.linalg.norm(b)!=0:
		cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
	return cos_sim




if __name__ == '__main__':
	start = time.time()
	documents=[]
	filename = {}
	vocab = set()
	files = os.listdir('TRAIN')
	for i,file in enumerate(files):
		filename[i] = file
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

	doclen = [] #length of the docs
	totcount = 0
	for doc in documents:
		totcount = 0
		for para in doc:
			totcount += len(para)
		doclen.append(totcount)



	inverted_index = createInvertedIndex(documents)	
	

	tf_idf = calculate_TfIdf_Weights(vocab, inverted_index, documents)



	#take testfile name from cl arguments
	testDocument = str(sys.argv[1])

	with open(testDocument, encoding="utf8", errors="ignore") as input_file:
		testdata = input_file.read()

	#process test document
	paras = extractParas(testdata)
	testparagraphs = []
	for j, para in enumerate(paras):
		###preprocessing###
		tokens = tokenize(para)
		stoplesstokens = stopwordRemoval(tokens)
		finaltokens = stemmer(stoplesstokens)
		###
		testparagraphs.append(finaltokens)
	
	
	
	

	# ranking = rankDocsByMatchingSimilarity(documents, testparagraphs, tf_idf)
	# Use when Matching Similarity is to be used to rank docs

	ranking = rankDocsByCosineSimilarity(documents, testparagraphs, tf_idf, inverted_index)

	#print top 10 matching documents
	print("\nTop 10 documents matching the given test document in ranked order are: ")
	print("Rank","	-	","Doc No", "	-	","Doc Name","		-	","Cosine Score")
	cnt = 0
	for i in ranking:
		cnt += 1
		print(cnt,"	-	",i, "		-	",filename[i],"	-	",ranking[i])
		if(cnt>9):
			break

	#para as a query part
	print("\n")
	print("Calculating document uniqueness...")
	
	totaldocs = len(documents)
	totalmatches = 0
	for k, tpara in enumerate(testparagraphs):
		testparavocab = set()
		for term in tpara:
			testparavocab.add(term)
		testparaweights = calculateParaWeights(tpara, testparavocab, inverted_index, totaldocs)
		matchfound = 0
		for i,doc in enumerate(documents):
			for j,para in enumerate(doc):
				paraweights = calculateParaWeights(para, testparavocab, inverted_index, totaldocs)
				cos_sim = cosine_sim(testparaweights, paraweights)
				if(cos_sim > 0.95):
					matchfound = 1
					totalmatches += 1
					print("Paragraph ",k," from test document matches with paragraph ",j," from document ",i," - ",filename[i])
					break
			if(matchfound == 1):
				break
	print("Document uniqueness = ", round(100 - (totalmatches*100/len(testparagraphs)), 2),"%")

	print("Total time taken = ", time.time() - start, "s")