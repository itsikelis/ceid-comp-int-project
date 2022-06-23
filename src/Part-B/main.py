# Import numpy and pandas
import numpy as np
import pandas as pd
import preprocess as prep

POP_COUNT = 5

def main():
    # Create a population.
    pop = prep.create_population(POP_COUNT, 1.6/3)

    # Get the BoW model from the previous part.
    X = prep.dat_to_bow_2(r'src/Part-B/data/dat/mini-test.dat')
    
    tf_matrix = prep.calc_tf(pop, X)

    print("FINAL SUM")
    print(tf_matrix.sum())


if __name__ == '__main__':
    for i in range(0,5):
        main()
