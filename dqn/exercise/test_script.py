import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def main():
    ones = np.ones([2, 2])
    var = tf.Variable(ones, dtype='float32')
    print(ones)
    print(var)
    var = K.variable(value=ones)
    print(var)

if __name__=='__main__':
    main()