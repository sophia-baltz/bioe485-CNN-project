import numpy as np
import gnumpy as gnp

class ConvLayer:
    def __init__(self, inputSize, kernelNumber, kernelSize):
        self.inputSize = inputSize
        self.kernelNumber = kernelNumber
        self.kernelSize = kernelSize
        self.kernels = self.initializeKernels(kernelNumber, kernelSize)
        self.bias = self.initializeBias(kernelNumber)
        self.kernelsGradients = np.zeros((kernelNumber, kernelSize, kernelSize))
        self.biasGradients = np.zeros(kernelNumber)
        self.momentum = 0.9
    
    #assigns random initial weights to kernels (which have shape kernelSize x kernelSize)
    def initializeKernels(self, kernelNumber, kernelSize):
        return np.random.randn(kernelNumber, kernelSize, kernelSize)
    
    def initializeBias(self, kernelNumber):
        return np.random.randn(kernelNumber)
    
    #activation function
    def relu(self,input):
        input[input < 0] = 0
        return input
    
    #derivative of relu function
    def drelu(self,input):
        input[input>0] = 1
        input[input<=0] = 0
        return input
    
    
    #takes in an nxn matrix of inputs, outputs kernelNumber of n+3-(kernelSize) square matricies
    def forward(self,input):
        self.inputs = np.copy(input)
        self.outputs = np.zeros((self.kernelNumber, self.inputSize, self.inputSize))

        #add padding
        self.inputs = np.pad(self.inputs,pad_width=1)

        #convolution function
        for idx_k, kernel in enumerate(self.kernels):
            for i in range(self.inputSize+2):
                for j in range(self.inputSize+2):
                    region = input[i:i+self.kernelSize, j:j+self.kernelSize]
                    self.outputs[idx_k,i, j] = gnp.sum(region * kernel) + self.bias[idx_k]

        self.outputs = self.relu(self.outputs) 
        return self.outputs                                  

