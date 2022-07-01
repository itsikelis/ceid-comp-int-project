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

import matplotlib.pyplot as plt

## Repeat algorithm 10 times in each case.
GA_ITERATIONS = 4

## Maximum generations allowed.
MAX_ITER = 500
## Score improvement threshold.
THRESHOLD = 2e-05

## Population size.
POP_SIZE = 200
## Crossover probability.
CROSS_PROB = .1
## Mutation probability.
MUTATION_PROB = .01

def main():

    # Get the BoW model from the previous part.
    X = prep.dat_to_bow_2(r'src/Part-B/data/dat/train-data.dat')
    tf_idf = prep.calc_tf_idf(prep.calc_tf(X), prep.calc_idf(X))

    ## Create an array to store the best individual scores each time.
    best_scores = np.zeros(1, dtype=float)
    total_generations = np.empty(0, dtype=int)

    for i in range(0, GA_ITERATIONS):

        # Create a population.
        pop = prep.create_population(POP_SIZE, 1.6/3)

        ## Reset initial metric values.
        best_score_prev = .00001
        generation_count = 0
        ## To record each generation's's scores.
        generation_scores = np.empty(0, dtype=float)
        while (generation_count<MAX_ITER):
            ## Get scores for current population.
            scores = prep.evaluate(pop, tf_idf)
            ## Compare the current best individual's score with the previous one.
            best_score_curr = np.max(scores)
            score_increase = ((best_score_curr/best_score_prev)-1)
            ## Make sure algorithm runs at least 30 times.
            if((np.absolute(score_increase) < THRESHOLD) & (generation_count >= 30)):
                break

            ## Create new population
            pop = prep.roulette(pop, scores, POP_SIZE)
            pop = prep.cross(pop, CROSS_PROB)
            pop = prep.mutate(pop, MUTATION_PROB)

            ## Set current mean score as previous and loop.
            best_score_prev = best_score_curr
            generation_count += 1 

            ## Record each generation's best score.
            generation_scores = np.append(generation_scores, best_score_curr)
            print('Generation:', generation_count)
            print('Score:', best_score_curr)

        ## If current iteration had better mean score, update best scores
        if (np.mean(generation_scores) > np.mean(best_scores)):
            print('Updated')
            best_scores = generation_scores
            best_individual = pop[np.argmax(scores)]

        ## Record generation count.
        total_generations = np.append(total_generations, generation_count)

    ## Create plots and show mean data.
    plt.figure()
    plt.xlabel('Generations')
    plt.ylabel('Mean Score')
    plt.plot(best_scores, label='Score/Generation')
    plt.legend(loc='best')
    print('Mean generation count: ', np.mean(total_generations))
    print('Mean best score: ', np.mean(best_scores))

    plt.show()


    ############  Artificial Neural Network - Code Taken from Part-A.  ############


    print('Loading Data.')
    ## Get BoW dataframe for train dataset.
    X = prep.dat_to_bow_std(r'src/Part-B/data/dat/train-data.dat')
    ## Get BoW dataframe for test dataset
    X_eval = prep.dat_to_bow_std(r'src/Part-B/data/dat/test-data.dat')
    # Load the test labels.
    Y = np.genfromtxt(r'src/Part-B/data/dat/train-label.dat', delimiter=' ',
                      dtype='int')
    Y_eval = np.genfromtxt(r'src/Part-B/data/dat/test-label.dat', delimiter=' ',
                           dtype='int')


    ## Delete all zero values from X and X_eval, as instructed by the best individual.
    X = np.delete(X, np.where(best_individual == 0), axis=1)
    X_eval = np.delete(X_eval, np.where(best_individual == 0)[0], axis=1)

    print('Labels Loaded!')

    ## Create a callback for early stopping
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy', min_delta=0.0003, patience=3)

    ## Split Data To Training And Testing Data 5-Fold
    kfold = KFold(n_splits=2, shuffle=True)
    prev_loss = 100
    for curr_fold, (train, test) in enumerate(kfold.split(X)):

        ## Specify Model's Input.
        inputs = keras.Input(shape=(X.shape[1], ))
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
                      optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.001, momentum=0.2),
            metrics=['mse', 'accuracy'])
        ## Print Model's Summary.
        model.summary()
        ## Fit Model and store each epoch's metrics in history.
        history = model.fit(X, Y, epochs=100,
                            verbose=1,
                            callbacks=[callback]
                            )

        ## Evaluate the model and store results to file.
        eval = model.evaluate(X_eval, Y_eval, verbose=1)

        # Save history, eavluation results and model
        if(eval[0] < prev_loss):
            ## Store History.
            history_path = os.path.join(
                'src', 'Part-B', 'saves', 'history', 'history.pkl')
            f = open(history_path, 'wb')
            pickle.dump(history.history, f)
            f.close()
            ## Store Evaluation Results.
            eval_path = os.path.join(
                'src', 'Part-B', 'saves', 'evaluation', 'eval.pkl')
            f = open(eval_path, 'wb')
            pickle.dump(eval, f)
            f.close()
            # End If.
        # End For.


if __name__ == '__main__':
    main()

