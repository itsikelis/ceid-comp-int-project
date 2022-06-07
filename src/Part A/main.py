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


def main():

    print('Loading Data.')
    ## Get vocabulary for count vectorizer.
    vocab = prep.get_voc()
    ## Get BoW dataframe for train dataset.
    X = prep.dat_to_bow_std(r'src/data/dat/train-data.dat', vocab)
    ## Get BoW dataframe for test dataset
    X_eval = prep.dat_to_bow_std(r'src/data/dat/test-data.dat', vocab)
    # Load the test labels.
    Y = np.genfromtxt(r'src/data/dat/train-label.dat', delimiter=' ', 
        dtype='int')
    Y_eval = np.genfromtxt(r'src/data/dat/test-label.dat', delimiter=' ',
        dtype='int')

    print('Labels Loaded!')

    ## Create a callback for early stopping
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy', min_delta=0.0003, patience=3)

    ## Split Data To Training And Testing Data 5-Fold
    kfold = KFold(n_splits=5, shuffle=True)
    prev_loss = 100
    for curr_fold, (train, test) in enumerate(kfold.split(X)):

        ## Specify Model's Input.
        inputs = keras.Input(shape=(8520, ))
        ## Create One Hidden Layer.
        x = keras.layers.Dense(units=4270, activation='relu', 
            activity_regularizer=keras.regularizers.L2(0.9))(inputs)
        ## Add second Hidden Layer
        x = keras.layers.Dense(units=8540, activation='relu',
            activity_regularizer=keras.regularizers.L2(0.9))(x)
        ## Specify Model's Output.
        outputs = keras.layers.Dense(units=20, activation='sigmoid', 
            activity_regularizer=keras.regularizers.L2(0.9))(x)
        ## Create Model.
        model = keras.Model(inputs=inputs, outputs=outputs)

        ## Compile Model.
        model.compile(loss=keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.2),
            metrics=['mse', 'accuracy'])
        ## Print Model's Summary.
        model.summary()
        ## Fit Model and store each epoch's metrics in history.
        history = model.fit(X, Y, epochs=150, 
            verbose=1, 
            callbacks=[callback]
            )

        ## Evaluate the model and store results to file.
        eval = model.evaluate(X_eval, Y_eval, verbose=1)

        # Save history, eavluation results and model
        if(eval[0] < prev_loss):
            ## Store History.
            history_path = os.path.join(
                'src', 'saves', 'a4_3', 'history', 'history.pkl')
            f = open(history_path, 'wb')
            pickle.dump(history.history, f)
            f.close()
            ## Store Evaluation Results.        
            eval_path = os.path.join(
                'src', 'saves', 'a4_3', 'evaluation', 'eval.pkl')
            f = open(eval_path, 'wb')
            pickle.dump(eval, f)
            f.close()
        # End If.
    # End For.


if __name__ == '__main__':
    main()
