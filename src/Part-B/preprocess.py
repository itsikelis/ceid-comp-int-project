from operator import delitem
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


## @package prep
#  
#  Some utility scripts for pre-processing the data.
#

## Function that creates a standardised BoW.
#
#  @param path The path to the .dat file.
#  @param voc The vocabulary to be used by the CountVectorizer
#  
#  @return bow The standardised BoW returned as a numpy ndarray.
#
def dat_to_bow_std(path, voc):
	## Read dat file to a pandas dataframe.
	df = pd.read_csv(path, header=None, names=['text'])
	## Remove all sentence and word counters using regex.
	df = df.replace(to_replace='<[0-9]*> ', value='', regex=True)
	## Create a Vectorizer object with given vocabulary.
	# cv = CountVectorizer(vocabulary=voc)
	# Set fixed_vocabulary boolean to True.
	# cv.fixed_vocabulary_ = True     	
	## Create a sparse matrix where:
	# - columns: Every word found in data.
	# - rows: one for each document (line) in .dat file.
	# doc_vec = cv.fit_transform(df['text'])
	## Return the standardized matrix.
	# bow = StandardScaler().fit_transform(doc_vec.toarray())
	return df

## @ brief Create a vocabulary(dict) of type str(number): int(number).
#
#  @return voc The vocabulary.
#
def get_voc():
	ints = (list(range(0, 8520)))
	strings = [str(x) for x in ints]
	voc = dict(zip(strings, ints))
	return voc


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
