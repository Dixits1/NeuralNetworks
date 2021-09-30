import random
import math
import copy
from jitspec import *

RANDOM_VALUE_RANGE = [-1.0, 1.5]
N_OUTPUTS = 1
N_LAYERS = 2 # number of layers (not including inputs)

"""
"""
@jitspec(spec)
class Network:
    """
    """
    def __init__(self, nInputs, nHidden, weights=None):        
        self.nInputs = nInputs
        self.nHidden = nHidden

        # input, activation, outputs
        self.inputs = self.initArray(nInputs) # input array
        self.thetaj = self.initArray(nHidden) # activations for hidden layer
        self.hj = self.initArray(nHidden) # outputted values for hidden layer
        self.theta0 = 0.0 # activation for output layer
        self.F0 = 0.0 # outputted value of network
        self.output = 0.0 # true output value

        self.trainingInputs = self.initArray(1) # all input data used for training
        self.trainingOutputs = self.initArray(1) # all corresponding output data used for training
        self.trainingPos = -1 # current position within the training data

        # if no weights are provided, initialize the weight array with 0s;
        # else, initialize the weight array using the weights provided
        if(weights == None):
            self.weights = self.initRandomWeights(nInputs, nHidden, RANDOM_VALUE_RANGE[0], RANDOM_VALUE_RANGE[1])
            
        else:
            self.weights = weights

    """
    Initializes the weight array as a 3D array:
    - D1: the layer
    - D2: the input node of the next layer
    - D3: the output node in the previous layer

    The array is initialized with random values between min (inclusive) and 
    max (inclusive).

    nInputs specifies the number of input nodes in the network.
    nHidden specifies the number of hidden nodes in the network.
    min specifies the minimum random value of the randomly generated values.
    max specifies the maximum random value of the randomly generated values.

    Returns the weights array.
    """
    def initRandomWeights(self, nInputs, nHidden, min, max):
        weights = [[[] for i in range(nInputs)],[[] for i in range(nHidden)]]
        for i in range(nInputs):
            for j in range(nHidden):
                weights[0][i].append(self.getRandomValue(min, max))
            
        for i in range(nHidden):
            for j in range(N_OUTPUTS):
                weights[1][i].append(self.getRandomValue(min, max))

        return weights

    """
    Returns a 1D array of length n with values of 0.
    """
    def initArray(self, n):
        return [0.0 for i in range(n)]

    """
    Propogate inputs through network by running computeLayer twice.

    Returns the output value of the network.
    """
    def run(self):
        self.zeroNetwork()

        for i in range(2):
            self.computeLayer(i)

        return self.F0           

    """
    Calculate the activation/output value for the layer specified by nLayer.

    nLayer of 0 is the hidden layer.
    nLayer of 1 is the output layer.

    Precondition: nLayer is either 0 or 1.
    Nothing is returned.
    """
    def computeLayer(self, nLayer):

        if nLayer == 0:
            for j in range(self.nHidden):
                for i in range(self.nInputs):
                    self.thetaj[j] += self.weights[0][i][j]*self.inputs[i]
                self.hj[j] = self.f(self.thetaj[j])
        elif nLayer == 1:
            for i in range(self.nHidden):
                self.theta0 += self.weights[1][i][0]*self.hj[i]
            self.F0 = self.f(self.theta0)

    """
    Returns a random value between min (inclusive) and max (inclusive) to each element.
    
    min specifies the minimum random value
    max specifies the maximum random value
    """
    def getRandomValue(self, min, max):
        return random.uniform(min, max)

    """
    The output function of each node in the network.
    """
    def f(self, x):
        return 1.0/(1.0 + math.exp(-x))

    """
    The derivative of the output function of each node in the network.
    """
    def fDeriv(self, x):
        return self.f(x)*(1.0-self.f(x))

    """
    Returns the error of network's most recent output using the following equation:
    ((network output - true output)^2)/2
    """
    def getError(self):
        return ((self.F0 - self.output)**2.0)/2.0

    """
    Trains the network given a set of inputs and outputs for training data.

    Returns the weights.
    """
    def train(self, inputs, outputs, maxIterations, lr, epochs=100):
        # confirm that length of input and output training data arrays are of same length
        assert len(inputs) == len(outputs), "The input and output sets are not the same length. input: " + len(inputs) + ", output: " + len(outputs)
        
        # set input and output training arrays
        self.trainingInputs = inputs
        self.trainingOutputs = outputs

        # initialize epoch and iterations variables
        # epochSize = int(maxIterations/epochs)
        # epochError = 0
        iterations = 0

        # initialize arrays and values used during training calculations
        omega0 = 0.0
        psi0 = 0.0
        
        omegaj0 = self.initArray(self.nHidden)
        psij0 = self.initArray(self.nHidden)

        partialEwkj = copy.deepcopy(self.weights[0])
        partialEwj0 = copy.deepcopy(self.weights[1])

        delwkj = copy.deepcopy(self.weights[0])
        delwj0 = copy.deepcopy(self.weights[1])

        # boolean used to determine whether training is finished or not
        finished = False

        # training loop
        while not finished:
            self.inputs, self.output = self.getNextTrainingMember()

            self.run()

            # prints the average error per iteration for all iterations in the epoch
            # at the end of each epoch
            iterations += 1
            # epochError += self.getError()
            # if iterations % epochSize == 0:
            #     print("Iteration", iterations, "- Error:", epochError/epochSize)
            #     epochError = 0.0

            omega0 = (self.output - self.F0)
            psi0 = omega0 * self.fDeriv(self.theta0)

            for j in range(self.nHidden):
                partialEwj0[j][0] = -self.hj[j]*psi0
                delwj0[j][0] = -lr*partialEwj0[j][0]
                self.weights[1][j][0] += delwj0[j][0]

                omegaj0[j] = psi0 * self.weights[1][j][0]
                
                psij0[j] = omegaj0[j]*self.fDeriv(self.thetaj[j])

            for k in range(self.nInputs):
                for j in range(self.nHidden):
                    partialEwkj[k][j] = -self.inputs[k]*psij0[j]        
                    delwkj[k][j] = -lr*partialEwkj[k][j]
            for k in range(self.nInputs):
                for j in range(self.nHidden):
                    self.weights[0][k][j] += delwkj[k][j]

            # end condition
            finished = iterations >= maxIterations

        print("\nNetwork has reached maximum iterations.")
        print("Max Iterations:", maxIterations)
        print("Learning Rate:", lr)
        print("# of Inputs:", self.nInputs)
        print("# of Hidden Nodes:", self.nHidden)
        print("# of Outputs:", N_OUTPUTS)
        print("Random Value Range:", RANDOM_VALUE_RANGE)

        self.runOverTrainingData()
        
        return self.weights
            
    """
    Zeroes the thetaj array and hj array as well as the theta0 and F0 values.
    """
    def zeroNetwork(self):
        for i in range(self.nHidden):
            self.thetaj[i] = 0.0
            self.hj[i] = 0.0
        self.theta0 = 0.0
        self.F0 = 0.0

    """
    Retrieves the next training member in the training set sequentially. Once the end of 
    the training set is reached, it loops back to the first training member and continues.

    Returns a tuple in the following format:
    (next training member's inputs, next training member's true outputs)
    """
    def getNextTrainingMember(self):
        self.trainingPos += 1
        if(self.trainingPos == len(self.trainingInputs)):
            self.trainingPos = 0

        return (self.trainingInputs[self.trainingPos], self.trainingOutputs[self.trainingPos])

    """
    Runs the network over each of the values in the training data and prints the results
    in the following format:

    Network Inputs  Network Output  True Output     Error
    [0.0, 0.0]       0.01857         0.0             0.0001723342
    [1.0, 0.0]       0.98305         1.0             0.0001437082
    [0.0, 1.0]       0.9837          1.0             0.0001328677
    [1.0, 1.0]       0.01679         0.0             0.0001409572
    """
    def runOverTrainingData(self):
        print("\nNetwork Inputs\tNetwork Output\tTrue Output\tError\t")
        for input, output in zip(self.trainingInputs, self.trainingOutputs):
            self.inputs = input
            self.output = output
            
            self.run()

            print(self.inputs, "\t", round(self.F0, 5), "\t", round(self.output, 5), "\t\t", round(self.getError(), 10))