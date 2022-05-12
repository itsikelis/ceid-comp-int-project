# Import numpy and pandas
import numpy as np
import pandas as pd
import preprocess as prep

from sklearn.model_selection import KFold

import tensorflow as tf
import keras

import os
import pickle


def main():

	print('Loading Data.')
	## Get vocabulary for count vectorizer.
	vocab = prep.get_voc()
	## Get BoW dataframe for train dataset.
	X = prep.dat_to_pad(r'src/data/dat/train-data.dat')
	## Get BoW dataframe for test dataset
	X_eval = prep.dat_to_pad(r'src/data/dat/test-data.dat')
	# Load the test labels.
	Y = np.genfromtxt(r'src/data/dat/train-label.dat', delimiter=' ',
	                  dtype='int')
	Y_eval = np.genfromtxt(r'src/data/dat/test-label.dat', delimiter=' ',
	                       dtype='int')
	print('Labels Loaded!')
	## Create a callback for early stopping
	callback = tf.keras.callbacks.EarlyStopping(
	    monitor='accuracy', min_delta=0.0005, patience=1)
	# Split Data To Training And Testing Data 5-Fold
	kfold = KFold(n_splits=2, shuffle=True)
	prev_loss = 100
	for curr_fold, (train, test) in enumerate(kfold.split(X)):
		## Create Model.
		model = keras.Sequential()
		## Add Embedding Layer.
		model.add(keras.layers.Embedding(input_dim=8521, output_dim=32, input_length=270))
		## Add a Flatten Layer.
		model.add(keras.layers.Flatten())
		## Create One Hidden Layer.
		model.add(keras.layers.Dense(units=4270, activation='relu'))
		## Specify Model's Output.
		model.add(keras.layers.Dense(units=20, activation='sigmoid'))
		## Compile Model.
		model.compile(loss=keras.losses.BinaryCrossentropy(),
		              optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
		              metrics=['mse', 'accuracy'])
		## Print Model's Summary.
		model.summary()
		## Fit Model and store each epoch's metrics in history.
		history = model.fit(X[train], Y[train], epochs=50,
		                    verbose=1,
		                    callbacks=[callback]
		                    )	
		## Evaluate the model and store results to file.
		eval = model.evaluate(X[test], Y[test], verbose=1)	
		# Save history and evaluation results of best fold.
		if(eval[0] < prev_loss):
			## Store current model as best.
			prev_loss = eval[0]
			## Store History.
			history_path = os.path.join(
			    'src', 'saves', 'a5_b', 'history', 'history.pkl')
			f = open(history_path, 'wb')
			pickle.dump(history.history, f)
			f.close()
			## Store Evaluation Results.
			eval_path = os.path.join(
			    'src', 'saves', 'a5_b', 'evaluation', 'eval.pkl')
			f = open(eval_path, 'wb')
			pickle.dump(eval, f)
			f.close()
		# End If.
	# End For.	


if __name__ == '__main__':
	main()
