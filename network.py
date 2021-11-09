"""
" Arjun Dixit
" September 10, 2021
"
" A class which represents a feed-foward, fully-connected neural network with the following specifications:
" any number of inputs, 1 hidden layer with any number of nodes, and any number of output values.
" Contains the following methods:
"
" __init__(nInputs, nHidden, nOutputs, weights = None)
" verifyWeights(weights)
" initRandomWeights(randMin, randMax)
" initArray(n)
" f(x)
" fDeriv(x)
" run()
" getRandomValue(randMin, randMax)
" printNetworkSpecs()
" getError()
" train(inputs, outputs, maxIterations, errorThreshold, lr)
" getNextTrainingMember()
" runOverTrainingData()
"""
import random
from math import exp
import time

RANDOM_VALUE_RANGE = [-1.0, 1.5]
N_LAYERS = 2         # number of layers (not including inputs)
N_DIGITS_DEC = 8     # number of digits in the decimal after rounding (used when printing values like error)
GET_ERROR_MULT = 0.5 # used in getError
MS_IN_S = 1000.0     # milliseconds in seconds

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
      self.layerSpec = [nInputs, nHidden, nOutputs]

      # pre-made loop arrays to optimize performance
      self.nInputsR = range(self.nInputs)
      self.nHiddenR = range(self.nHidden)
      self.nOutputsR = range(self.nOutputs)
      self.layerSpecR = [range(i) for i in self.layerSpec]

      # input, activation, outputs
      self.inputs = self.initArray(nInputs)    # input array
      self.thetaj = self.initArray(nHidden)    # activations for hidden layer
      self.hj = self.initArray(nHidden)        # outputted values for hidden layer
      self.thetai = self.initArray(nOutputs)   # activations for output layer
      self.Fi = self.initArray(nOutputs)       # outputted values of output layer
      self.Ti = self.initArray(nOutputs)       # true output values

      self.omegai = self.initArray(nOutputs)
      self.psii = self.initArray(nOutputs)

      self.Esum = 0.0                          # total error of network

      self.trainingInputs = self.initArray(1)  # all input data used for training
      self.trainingOutputs = self.initArray(1) # all corresponding output data used for training
      self.trainingPos = -1                    # current position within the training data

      # if no weights are provided, initialize the weight array with random values;
      # else, initialize the weight array using the weights provided
      if weights == None:
         self.weights = self.initRandomWeights(RANDOM_VALUE_RANGE[0], RANDOM_VALUE_RANGE[1])
      else:
         if not self.verifyWeights(weights):
            raise Exception("The dimensions of the weights file does not match the network dimensions.")
         self.weights = weights

      return
   # def __init__(self, nInputs, nHidden, nOutputs, weights = None)

   """
   " Verifies the dimensions of the given weights object.
   "
   " weights specifies the weights object of which to verify the dimensions.
   "
   " Returns True if weights match the network's layerSpec in dimensions; otherwise
   " returns False.
   """
   def verifyWeights(self, weights):
      matches = True

      matches = matches and len(weights) == len(self.layerSpec) - 1

      for i in range(len(self.layerSpec) - 1):
         matches = matches and len(weights[i]) == self.layerSpec[i]
         matches = matches and len(weights[i][0]) == self.layerSpec[i + 1]
      
      return matches
   # def verifyWeights(self, weights)


   """
   " Initializes the weight array as a 3D array:
   " - D1: the layer
   " - D2: the input node of the next layer
   " - D3: the output node in the previous layer
   " 
   " The array is initialized with random values between randMin (inclusive) and 
   " randMax (inclusive).
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

      for i in self.nOutputsR:
         self.thetai[i] = 0.0

         for j in self.nHiddenR:

            self.thetaj[j] = 0.0
            for k in self.nInputsR:
               self.thetaj[j] += self.inputs[k] * self.weights[0][k][j]
         
            self.hj[j] = self.f(self.thetaj[j])
            self.thetai[i] += self.weights[1][j][i] * self.hj[j]
         # for j in self.nHiddenR
      
         self.Fi[i] = self.f(self.thetai[i])
         self.omegai[i] = self.Ti[i] - self.Fi[i]
         self.psii[i] = self.omegai[i] * self.fDeriv(self.thetai[i])

         self.Esum += self.omegai[i] * self.omegai[i]
      # for i in self.nOutputsR

      return
   # def run(self)

   """
   " Returns a random value between randMin (inclusive) and randMax (inclusive) to each element.
   "
   " randMin specifies the minimum random value
   " randMax specifies the maximum random value
   """
   def getRandomValue(self, randMin, randMax):
      return random.uniform(randMin, randMax)

   """
   " Prints the network specifications in the following format:
   "
   " Number of Inputs: 2
   " Number of Hidden Nodes: 5
   " Number of Outputs: 3
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

      iterations = 0
      
      totalError = 0.0                             # sum of error used for end condition
      trainingLen = len(inputs)                    # length of training set
      
      finished = False                             # used to determine whether training is finished or not
      errorThresholdReached = False
      maxIterationsReached = False

      # set input and output training arrays
      self.trainingInputs = inputs
      self.trainingOutputs = outputs

      trainingTime = time.time() 

      # training loop
      while not finished:
         self.inputs, self.Ti = self.getNextTrainingMember()

         self.run()

         for k in self.nInputsR:
            for j in self.nHiddenR:
               omegaj[j] = 0.0
               for i in self.nOutputsR:
                  omegaj[j] += self.psii[i] * self.weights[1][j][i]
                  self.weights[1][j][i] += lr * self.hj[j] * self.psii[i]

               psij[j] = omegaj[j] * self.fDeriv(self.thetaj[j])
               self.weights[0][k][j] += lr * self.inputs[k] * psij[j]

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

      print("\nTraining time: " + str(int((time.time() - trainingTime) * MS_IN_S)) + " ms")

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
   " [0.0, 0.0]   [0.01949439, 0.00020501, 0.0173927]    [0.0, 0.0, 0.0]      0.00034129
   " [1.0, 0.0]   [0.98150611, 0.01207658, 0.98872078]   [1.0, 0.0, 1.0]      0.00030754
   " [0.0, 1.0]   [0.98378772, 0.01133334, 0.98821289]   [1.0, 0.0, 1.0]      0.00026511
   " [1.0, 1.0]   [0.0179882, 0.98426464, 0.99965449]    [0.0, 1.0, 1.0]      0.00028565
   "
   " Total Error:  0.00119959
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
