import numpy as np
import gnumpy as gnp

class ConvLayer:
    def __init__(self, kernelNumber, kernelSize, outputSize):
        self.kernelNumber = kernelNumber
        self.kernelSize = kernelSize
        self.kernels = self.initializeKernels(kernelNumber, kernelSize)
        self.kernelsGradients = np.zeros((kernelNumber, kernelSize, kernelSize))
        self.outputSize = outputSize
        self.momentum = 0.9

    def initializeKernels(self, kernelNumber, kernelSize):
        #assigns random initial weights to kernels (which have shape kernelSize x kernelSize)
        return np.random.randn(kernelNumber, kernelSize, kernelSize)
    
    def forward(self,input):
        self.inputs = np.copy(input)

        self.outputs = np.zeros((self.kernelNumber, self.outputSize, self.outputSize))

        sumOfAllInputs = np.sum(inputs, axis=0)
        for idx_k, kernel in enumerate(self.kernels):
            inputSum = np.copy(sumOfAllInputs)
            for input_idx in self.combinations[idx_k]:
                inputSum -= inputs[input_idx]             
            self.outputs[idx_k] += signal.correlate2d(inputSum, kernel, "valid")

        self.outputs = self.activation.forward(self.outputs) #pass the outputs in the activation function
        return self.outputs                                  # and then return


