# Import numpy and pandas
import numpy as np
import pandas as pd
import preprocess as prep

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

import os
import pickle


print('Loading Data.')
## Get vocabulary for count vectorizer.
vocab = prep.get_voc()
## Get BoW dataframe for train dataset.
X = prep.dat_to_bow_std(r'Part-B/data/dat/train-data.dat', vocab)
## Get BoW dataframe for test dataset
X_eval = prep.dat_to_bow_std(r'Part-B/data/dat/test-data.dat', vocab)
# Load the test labels.
Y = np.genfromtxt(r'Part-B/data/dat/train-label.dat', delimiter=' ',
                  dtype='int')
Y_eval = np.genfromtxt(r'Part-B/data/dat/test-label.dat', delimiter=' ',
                       dtype='int')

print('Labels Loaded!')

print(X.head())