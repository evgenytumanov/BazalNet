import tensorflow as tf
from gen_dataset import *


def main():

    print('\n##############################\n')
    print('TESTING THE DATASET GENERATION')

    print('Small dataset example')
    dataset = gen_dataset(lambda x, y: x+y, '+', 2, 3)
    #print(dataset)

    #print('\n X_train, X_val, y_train, y_val shapes')
    #print(list(map(lambda arr: arr.shape, dataset)))

    print('\n##############################\n')


if __name__ == "__main__":
    main()