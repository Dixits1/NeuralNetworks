"""
" Arjun Dixit
" September 10, 2021
"
" A class which represents a feed-foward, fully-connected neural network with the following specifications:
" any number of inputs, 1 hidden layer with any number of nodes, and any number of output values.
" Contains the following methods:
"
" __init__(nInputs, nHidden, nOutputs, weights = None)
" initRandomWeights(randMin, randMax)
" initArray(n)
" run()
" computeLayer(nLayer)
" getRandomValue(randMin, randMax)
" printNetworkSpecs()
" f(x)
" fDeriv(x)
" getError()
" train(inputs, outputs, maxIterations, errorThreshold, lr)
" getNextTrainingMember()
" runOverTrainingData()
"""
import random
from math import exp
import copy
from itertools import accumulate
from operator import mul

RANDOM_VALUE_RANGE = [-1.0, 1.5]
N_LAYERS = 2         # number of layers (not including inputs)
N_DIGITS_DEC = 8     # number of digits in the decimal after rounding (used when printing values like error)
GET_ERROR_MULT = 0.5 # used in getError

# TODO: make sure raise Exception in constructor works correctly
# TODO: fix set the set weights when running functionality??

class Network:
   """
   " Constructor which initializes the instance variables of the network:
   " 
   " nInputs specifies the number of inputs
   " nHidden specifies the number of hidden nodes
   " nOutputs specifies the number of outputs
   " weights specifies the weights array to use; if none is provided, weights
   "         are randomly initialized. 
   """
   def __init__(self, nInputs, nHidden, nOutputs, weights = None):      
      self.nInputs = nInputs
      self.nHidden = nHidden
      self.nOutputs = nOutputs

      self.nInputsR = range(self.nInputs)
      self.nHiddenR = range(self.nHidden)
      self.nOutputsR = range(self.nOutputs)

      # input, activation, outputs
      self.inputs = self.initArray(nInputs)    # input array
      self.thetaj = self.initArray(nHidden)    # activations for hidden layer
      self.hj = self.initArray(nHidden)        # outputted values for hidden layer
      self.thetai = self.initArray(nOutputs)   # activations for output layer
      self.Fi = self.initArray(nOutputs)       # outputted values of output layer
      self.Ti = self.initArray(nOutputs)       # true output values

      self.Esum = 0.0                          # total error of network

      self.trainingInputs = self.initArray(1)  # all input data used for training
      self.trainingOutputs = self.initArray(1) # all corresponding output data used for training
      self.trainingPos = -1                    # current position within the training data

      # if no weights are provided, initialize the weight array with 0s;
      # else, initialize the weight array using the weights provided
      if weights == None:
         self.weights = self.initRandomWeights(RANDOM_VALUE_RANGE[0], RANDOM_VALUE_RANGE[1])
      else:
         print(len(weights[0]))
         print(nHidden)
         print(len(weights[N_LAYERS - 1]))
         print(nOutputs)
         if not (len(weights[0]) == nHidden and len(weights[N_LAYERS - 1]) == nOutputs):
            raise Exception("The dimensions of the weights file does not match the provided dimensions.")
         self.weights = weights

      return
   # def __init__(self, nInputs, nHidden, nOutputs, weights = None)

   """
   " Initializes the weight array as a 3D array:
   " - D1: the layer
   " - D2: the input node of the next layer
   " - D3: the output node in the previous layer
   " 
   " The array is initialized with random values between min (inclusive) and 
   " max (inclusive).
   "
   " randMin specifies the minimum random value of the randomly generated values.
   " randMax specifies the maximum random value of the randomly generated values.
   " 
   " Returns the weights array.
   """
   def initRandomWeights(self, randMin, randMax):
      weights = [[[] for i in range(self.nInputs)],[[] for i in range(self.nHidden)]]


      for i in range(self.nInputs):
         for j in range(self.nHidden):
            weights[0][i].append(self.getRandomValue(randMin, randMax))
         
      for i in range(self.nHidden):
         for j in range(self.nOutputs):
            weights[1][i].append(self.getRandomValue(randMin, randMax))

      return weights
   # def initRandomWeights(self, randMin, randMax)

   """
   " Returns a 1D array of length n with values of 0.
   """
   def initArray(self, n):
      return [0.0 for i in range(n)]

   """
   " The output function of each node in the network; in this case, a sigmoid is used.
   "
   " Returns the value of the output function at x.
   """
   def f(self, x):
      return 1.0 / (1.0 + exp(-x))

   """
   " The derivative of the output function of each node in the network.
   "
   " Returns the value of the output function's derivative at x.
   """
   def fDeriv(self, x):
      fx = self.f(x)
      return fx * (1.0 - fx)

   """
   " Propogate inputs through network by running computeLayer twice.
   "
   " Returns the output values of the network.
   """
   def run(self):
      self.Esum = 0.0

      for j in self.nHiddenR:
         self.thetaj[j] = 0.0

         for k in self.nInputsR:
            self.thetaj[j] += self.weights[0][k][j] * self.inputs[k]
         
         self.hj[j] = self.f(self.thetaj[j])

      for i in self.nOutputsR:
         self.thetai[i] = 0.0

         for j in self.nHiddenR:
            self.thetai[i] += self.weights[1][j][i] * self.hj[j]
      
         self.Fi[i] = self.f(self.thetai[i])

         self.Esum += (self.Ti[i] - self.Fi[i]) * (self.Ti[i] - self.Fi[i])

      return self.Fi
   # def run(self)

   """
   " Returns a random value between min (inclusive) and max (inclusive) to each element.
   "
   " randMin specifies the minimum random value
   " randMax specifies the maximum random value
   """
   def getRandomValue(self, randMin, randMax):
      return random.uniform(randMin, randMax)

   """
   " Prints the network specifications in the following format:
   "
   " # of Inputs: 2
   " # of Hidden Nodes: 5
   " # of Outputs: 3
   " Random Value Range: [-1.0, 1.5]
   """
   def printNetworkSpecs(self):
      print("\nNumber of Inputs:", self.nInputs)
      print("Number of Hidden Nodes:", self.nHidden)
      print("Number of Outputs:", self.nOutputs)
      print("Random Value Range:", RANDOM_VALUE_RANGE)

      return

   """
   " Returns the error of the network.
   """
   def getError(self):
      return GET_ERROR_MULT * self.Esum

   """
   " Trains the network given a set of inputs and outputs for training data as well as the max iterations,
   " error threshold, and learning rate.
   " 
   " inputs specifies the input training data
   " outputs specifies the output training data
   " maxIterations specifies the maximum iterations for training
   " errorThreshold specifies the threshold which the training set error
   "                must reach before exiting training
   " lr specifies lambda, or the learning rate
   "
   " Precondition: input and output arrays are the same length.
   "  
   " Returns the weights after training is completed.
   """
   def train(self, inputs, outputs, maxIterations, errorThreshold, lr):
      # initialize arrays and values used during training calculations
      omegaj = self.initArray(self.nHidden)
      psij = self.initArray(self.nHidden)
      
      omegai = self.initArray(self.nOutputs)
      psii = self.initArray(self.nOutputs)

      partialEwkj = copy.deepcopy(self.weights[0]) # creates new array with same structure as weights for input -> hidden
      partialEwji = copy.deepcopy(self.weights[1]) # creates new array with same structure as weights for hidden -> output

      delwkj = copy.deepcopy(self.weights[0])      # creates new array with same structure as weights for input -> hidden
      delwji = copy.deepcopy(self.weights[1])      # creates new array with same structure as weights for hidden -> output

      iterations = 0
      
      totalError = 0.0                             # sum of error used for end condition
      trainingLen = len(inputs)                    # length of training set
      
      finished = False                             # used to determine whether training is finished or not
      errorThresholdReached = False
      maxIterationsReached = False

      # set input and output training arrays
      self.trainingInputs = inputs
      self.trainingOutputs = outputs

      # training loop
      while not finished:
         self.inputs, self.Ti = self.getNextTrainingMember()

         self.run()

         for i in self.nOutputsR:
            omegai[i] = self.Ti[i] - self.Fi[i]
            psii[i] = omegai[i] * self.fDeriv(self.thetai[i])

            for j in self.nHiddenR:
               partialEwji[j][i] = -self.hj[j] * psii[i]
               delwji[j][i] = -lr * partialEwji[j][i]

         for j in self.nHiddenR:
            omegaj[j] = sum(map(mul, psii, self.weights[1][j]))

         for j in self.nHiddenR:
            psij[j] = omegaj[j] * self.fDeriv(self.thetaj[j])

            for k in self.nInputsR:
               partialEwkj[k][j] = -self.inputs[k] * psij[j]      
               delwkj[k][j] = -lr * partialEwkj[k][j]
         
         # applying the weights
         for k in self.nInputsR:
            for j in self.nHiddenR:
               self.weights[0][k][j] += delwkj[k][j]
         
         for j in self.nHiddenR:
            for i in self.nOutputsR:
               self.weights[1][j][i] += delwji[j][i]
         
         iterations += 1
         totalError += self.getError()
         if self.trainingPos == trainingLen - 1: # occurs once everytime it finishes looping over all training members
            # check if average error across training members is less than errorThreshold
            errorThresholdReached = totalError <= errorThreshold
            totalError = 0.0


         maxIterationsReached = iterations >= maxIterations

         # end condition
         finished = maxIterationsReached or errorThresholdReached
      # while not finished
      
      self.runOverTrainingData()

      if errorThresholdReached:
         print("Network has reached the error threshold.")
      if maxIterationsReached:
         print("Network has reached maximum iterations.")

      print("\nMax Iterations:", maxIterations)
      print("Error Threshold:", errorThreshold)
      print("Learning Rate:", lr)

      self.printNetworkSpecs()
      
      return self.weights
   # def train(self, inputs, outputs, maxIterations, errorThreshold, lr)

   """
   " Retrieves the next training member in the training set sequentially. Once the end of 
   " the training set is reached, it loops back to the first training member and continues.
   " 
   " Returns a tuple in the following format:
   " (next training member's inputs, next training member's true outputs)
   """
   def getNextTrainingMember(self):
      self.trainingPos += 1
      if self.trainingPos == len(self.trainingInputs):
         self.trainingPos = 0

      return (self.trainingInputs[self.trainingPos], self.trainingOutputs[self.trainingPos])

   """
   " Runs the network over each of the values in the training data and prints the results
   " in the following format:
   " 
   " Network Inputs Network Output True Output Error 
   " [0.0, 0.0]   [0.02027223, 0.00016129, 0.01689993]   [0.0, 0.0, 0.0]      0.0003483
   " [1.0, 0.0]   [0.98145447, 0.01045743, 0.98651243]   [1.0, 0.0, 1.0]      0.0003176
   " [0.0, 1.0]   [0.98074885, 0.01256923, 0.98889583]   [1.0, 0.0, 1.0]      0.00032595
   " [1.0, 1.0]   [0.02159712, 0.98290588, 0.99978671]   [0.0, 1.0, 1.0]      0.00037935
   " 
   " Total Error: 
   """
   def runOverTrainingData(self):
      TiRounded = []
      FiRounded = []
      errorRounded = 0.0
      totalError = 0.0


      self.trainingPos = -1

      print("\nNetwork Inputs\tNetwork Output\tTrue Output\tError\t")
      
      for i in range(len(self.trainingInputs)):
         self.inputs, self.Ti = self.getNextTrainingMember()

         self.run()
         
         TiRounded = [round(i, N_DIGITS_DEC) for i in self.Ti]
         FiRounded = [round(i, N_DIGITS_DEC) for i in self.Fi]
         errorRounded = round(self.getError(), N_DIGITS_DEC)

         totalError += errorRounded

         print(self.inputs, "\t", FiRounded, "\t", TiRounded, "\t\t", errorRounded)
      # for i in range(len(self.trainingInputs))

      print("\nTotal Error: ", round(totalError, N_DIGITS_DEC))

      return
   # def runOverTrainingData(self)
# class Network
