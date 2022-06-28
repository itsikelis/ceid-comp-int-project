from cmath import inf
from operator import delitem
from cv2 import mean
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

VOC_SIZE = 8520
DOC_COUNT = 8250

SCALAR = 1000


## @package prep
#  
#  Some utility scripts for pre-processing the data. Enriched with functions 
#  used by the Genetic Algorithm in Part B.
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
def dat_to_bow_std(path):

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
	## Return the standardized matrix.
	bow = StandardScaler().fit_transform(doc_vec.toarray())
	return bow

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
	## Create 'count' individuals and add them to the population.
	for i in range (0, count):
		# Repeat until created individual has more than 1000 ones.
		while True:
			individual = np.random.choice([0, 1], size=(1, VOC_SIZE), 
											p=[1-prob, prob])
			## If individual has at least 1000 ones, break 
			# and add it to the population.						
			if np.sum(individual) >= 1000:
				break

		# Add new individual to population.
		pop = np.vstack([pop, individual])

	## Return population.
	return pop


## @brief Function that calculates the term frequency of an individual.
#
#  @param count The population count.
#
#  @param prob The probability to pick a word for the current individual.
#
#  @return bow_tf A DOC_COUNT by WORD_COUNT ndarray with the tf values 
#  of each word.
def calc_tf(bow):

	## Add bow row-wise to calculate total words in each document.
	total_words_row_wise = np.sum(bow, axis=1)

	## Create a copy of the BoW with dtype=float.
	tf_values = bow.astype(dtype=float)

	## Divide each row with the corresponding word count to get the tf matrix.
	for i in range(0, DOC_COUNT):
		tf_values[i] = tf_values[i] / total_words_row_wise[i]

	## Sum all tf values of each word and divide bu DOC_COUNT
	mean_tf_values = tf_values.sum(axis=0)
	mean_tf_values = np.divide(mean_tf_values, DOC_COUNT)
	## Return the mean tf value for each word found in all documents.
	return mean_tf_values


## @brief Function that calculates the inverse document frequency of each word.
#
#  @param bow The BoW representation of the test data.
#
#  @return idf A (1, VOC_SIZE) array with the idf of each word.
def calc_idf(bow):
	## Replace all non-zero elements of bow with 1.
	bow[bow>0] = 1
	## Sum all ones to get each word's appearence in every document.
	word_appearence = np.sum(bow, axis=0, dtype=int)
	## Create an array of floats to store the calculated idf values.
	idf_values = word_appearence.astype(dtype=float)
	## Calculate the element-wise inverse of the array.
	idf_values = np.reciprocal(idf_values)
	## Infs occur when dividing by zero, so we replace them with 0 again.
	idf_values[idf_values==inf] = 0
	## Multiply with DOC_COUNT.
	idf_values = np.multiply(idf_values, DOC_COUNT)
	## Get element-wise lof of array.
	idf_values = np.log10(idf_values)
	## And again, remove infs.
	## Infs occur when dividing by zero, so we replace them with 0 again.
	idf_values[idf_values == -inf] = 0
	## Return idf_values
	return idf_values

def calc_tf_idf(mean_tf_array, idf_array):
	## Multiply each row of the matrix with the idf values for each word.
	tf_idf = np.multiply(mean_tf_array, idf_array)
	
	return tf_idf

def evaluate(pop, tf_idf):
	## Create a matrix with only the selected words for each individual.
	selected = np.multiply(pop, tf_idf)

	## Divide with word count of each individual to get the final score.
	score = np.sum(selected, axis=1)
	score = np.divide(score, np.count_nonzero(selected, axis=1)-1000)*SCALAR
	return score

def roulette(pop, scores, pop_size):

	## Calculate total score.
	total_score = np.sum(scores)
	## Calculate selection probabilities.
	probs = np.divide(scores, total_score)

	number_of_rows = pop.shape[0]
	random_indices = np.random.choice(number_of_rows, size=pop_size, p=probs)
	
	new_pop = pop[random_indices, :]

	return new_pop

def cross(pop, cross_probability):
	## Initialize an array with the selected individuals.
	cross_pop = np.empty((0, 8520), dtype=int)
	## Initialize an array to store the indices of the selected individuals.
	cross_indices = np.empty(0, dtype=int)

	## Select the individuals to be crossed from the population matrix.
	for i in range(0, pop.shape[0]):

		## Calculate a random number between 0 and 1.
		rand = np.random.uniform()
		## If number is less than the cross probability, select
		#  individual for crossing.
		if(rand < cross_probability):
			## Append 
			cross_pop = np.vstack([cross_pop, pop[i]])
			## Also store index of selected individual.
			cross_indices = np.append(cross_indices, i)

	## Make sure that only an even number of individuals is selected.
	if (np.mod(cross_pop.shape[0], 2)!=0):
		## Select and remove a random individual from the cross set.
		index_to_remove = np.random.randint(low=0, high=cross_pop.shape[0])
		cross_pop = np.delete(cross_pop, index_to_remove, axis=0)

	## Uniformly cross pairs of individuals.
	for i in range(0, cross_pop.shape[0]-1, 2):
		for j in range(0, VOC_SIZE):
			## Determine if a cross is going to happen.
			rand = np.random.uniform()
			if (rand<=0.5):
				## Swap the two genes.
				temp = cross_pop[i][j]
				cross_pop[i][j] = cross_pop[i+1][j]
				cross_pop[i+1][j] = temp

	## Replace the selected individuals with their offspring.
	np.put(pop, cross_indices, cross_pop, mode='raise')

	return pop
	
def mutate(pop, mut_prob):
	## Calculate a random probability for each gene.
	r = np.random.random(size=pop.shape)	
	## Replace every gene with a smaller probability than mut_prob with 1 or 0.
	m = r < mut_prob
	pop[m] = np.random.randint(0, 2, size=np.count_nonzero(m))

	return pop
