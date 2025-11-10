import numpy as np
import gnumpy as gnp

#convolution of square matrix with 3x3 kernel and samepadding
def conv2D(input,kernel):
    Inp = np.pad(input,pad_width=1)
    i_size = Inp.shape[0]

    out = np.zeros((i_size,i_size))

    for i in range(i_size):
        for j in range(i_size):
            region = Inp[i:i+3,j:j+3]
            out[i,j] = np.sum(region * kernel)
    
    return out




    

def relu(input):
    input[input < 0] = 0
    return input