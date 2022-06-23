from operator import delitem
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

VOC_SIZE = 8520
DOC_COUNT = 3

## @package prep
#  
#  Some utility scripts for pre-processing the data.
#

## @ brief Create a vocabulary(dict) of type str(number): int(number).
#
#  @return voc The vocabulary.
#
def get_voc():
	ints = (list(range(0, 8520)))
	strings = [str(x) for x in ints]
	voc = dict(zip(strings, ints))
	return voc

## Function that creates a standardised BoW.
#
#  @param path The path to the .dat file.
#  @param voc The vocabulary to be used by the CountVectorizer
#  
#  @return bow The standardised BoW returned as a numpy ndarray.
#
def dat_to_bow_2(path):
	## Create a vocabulary.
	voc = get_voc()
	## Read dat file to a pandas dataframe.
	df = pd.read_csv(path, header=None, names=['text'])
	## Remove all sentence and word counters using regex.
	df = df.replace(to_replace='<[0-9]*> ', value='', regex=True)
	## Create a Vectorizer object with given vocabulary.
	cv = CountVectorizer(vocabulary=voc)
	# Set fixed_vocabulary boolean to True.
	cv.fixed_vocabulary_ = True
	## Create a sparse matrix where:
	# - columns: Every word found in data.
	# - rows: one for each document (line) in .dat file.
	doc_vec = cv.fit_transform(df['text'])
	return doc_vec.toarray()

## @brief Function that pads the .dat file contents with 0s at the end.
#
#  @param path The path to the .dat file.
#
#  @return embeddings The word embeddings returned as a numpy ndarray.
#
def dat_to_pad(path):
	## Read dat file to a pandas dataframe.
	df = pd.read_csv(path, header=None, names=['text'])
	## Remove all sentence and word counters using regex.
	df = df.replace(to_replace='<[0-9]*> ', value='', regex=True)
	## Split each word in its own column.
	df = df['text'].str.split(' ', expand=True)
	## Fill empty spots with zeros.
	df = df.fillna(0)
	## Return data as numpy matrix.
	arr = df.to_numpy(dtype='int')
	return arr

## @brief Function that creates a population of 8520 by 1 vectors.
#
#  @param count The population count.
#
#  @param prob The probability to pick a word for the current individual.
#
#  @return pop A 8520 by count matrix of the population.
def create_population(count, prob):

	## Create an empty population.
	pop = np.empty((0, VOC_SIZE), dtype=int)
	print(pop)
	## Create 'count' individuals and add them to the population.
	for i in range (0, count):
		# Repeat until created individual has more than 1000 ones.
		while True:
			individual = np.random.choice([0, 1], size=(1, VOC_SIZE), 
											p=[1-prob, prob])
			print(individual)
			## If individual has at least 1000 ones, break 
			# and add it to the population.						
			if np.sum(individual) >= 1000:
				break

		# Add new individual to population.
		pop = np.append(pop, individual, axis=0)

	## Return population.
	return pop

## @brief Function that calculates the term frequency of an individual.
#
#  @param count The population count.
#
#  @param prob The probability to pick a word for the current individual.
#
#  @return pop A 8520 by count matrix of the population.
def calc_tf(pop, bow):

	## Get row count of pop.
	rows, _ = pop.shape
	## Create a matrix to store the cumulative tf values of the population.
	pop_tf = np.empty((0 , VOC_SIZE))
	## For each individual in population (every row).
	for i in range(0, rows):
		## Create a vector to store the cumulative tf values of each word.
		doc_tf = np.zeros((1, VOC_SIZE))
		## For each document in vector.
		for j in range(0, DOC_COUNT):
			## Get frequency vector of each word selected in 
			# the current individual.
			freq = np.multiply(pop[i], bow[j])
			## Add the accumulated term frequency to the vector.
			doc_tf = doc_tf + freq

		## Calculate tf.
		doc_tf = doc_tf/np.count_nonzero(bow[j])
		print(doc_tf.max())
		## Add the calculated tf for the current word in 
		# the tf matrix for the population.
		pop_tf = np.append(pop_tf, doc_tf, axis=0)
	
	## Return the tf matrix.
	return pop_tf

def calc_idf():
	pass

