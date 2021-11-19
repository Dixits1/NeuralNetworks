"""
" Arjun Dixit
" September 10, 2021
"
" A class which represents a feed-foward, fully-connected neural network with the following specifications:
" any number of inputs, 2 hidden layers with any number of nodes, and any number of output values. The network
" trains using the backpropagation algorithm.
" Contains the following methods:
"
" __init__(nInputs, nHidden1, nHidden2, nOutputs, randRange, training, weights = None)
" verifyWeights(weights)
" initRandomWeights(randMin, randMax)
" initArray(n)
" f(x)
" fDeriv(x)
" runTraining()
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

N_LAYERS = 3         # number of layers (not including inputs)
N_DIGITS_DEC = 8     # number of digits in the decimal after rounding (used when printing values like error)
GET_ERROR_MULT = 0.5 # used in getError
MS_IN_S = 1000.0     # milliseconds in seconds

class Network:
   """
   " Constructor which initializes the instance variables of the network:
   " 
   " nInputs specifies the number of inputs
   " nHidden specifies the number of hidden nodes
   " nOutputs specifies the number of outputs
   " randRange specifies the random number range used to randomize the weights for training
   " training specifies if the network will be trained or not
   " weights specifies the weights array to use; if none are provided, weights
   "         are randomly initialized. 
   """
   def __init__(self, nInputs, nHidden1, nHidden2, nOutputs, randRange, training, weights = None):      
      self.nInputs = nInputs
      self.nHidden1 = nHidden1
      self.nHidden2 = nHidden2
      self.nOutputs = nOutputs
      self.layerSpec = [nInputs, nHidden1, nHidden2, nOutputs]

      self.randRange = randRange
      self.preloadedWeights = weights != None

      # pre-made loop arrays to optimize performance
      self.nInputsR = range(self.nInputs)
      self.nHidden1R = range(self.nHidden1)
      self.nHidden2R = range(self.nHidden2)
      self.nOutputsR = range(self.nOutputs)

      # input, activation, outputs
      self.am = self.initArray(nInputs)    # input array

      # only initializes variables when training
      if training:
         self.omegak = self.initArray(nHidden1)
         self.omegaj = self.initArray(nHidden2)

         self.psik = self.initArray(nHidden1)
         self.psij = self.initArray(nHidden2)
         self.psii = self.initArray(nOutputs)
         
      self.thetak = self.initArray(nHidden1)
      self.thetaj = self.initArray(nHidden2)
      self.thetai = self.initArray(nOutputs)

      self.ak = self.initArray(nHidden1)
      self.aj = self.initArray(nHidden2)
      self.ai = self.initArray(nOutputs)
      
      self.Ti = self.initArray(nOutputs)       # true output values

      self.Esum = 0.0                          # total error of network

      self.inputSet = self.initArray(1)        # all input data used for training
      self.outputSet = self.initArray(1)       # all corresponding output data used for training
      self.trainingPos = -1                    # current position within the training data

      # if no weights are provided, initialize the weight array with random values;
      # else, initialize the weight array using the weights provided
      if weights == None:
         self.weights = self.initRandomWeights(randRange[0], randRange[1])
      else:
         if not self.verifyWeights(weights):
            raise Exception("The dimensions of the weights file does not match the network dimensions.")
         self.weights = weights

      return
   # def __init__(self, nInputs, nHidden1, nHidden2, nOutputs, randRange, training, weights = None)

   """
   " Verifies the dimensions of the given weights object.
   "
   " weights specifies the weights array of which to verify the dimensions.
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
      weights = [[[] for i in range(self.nInputs)],[[] for i in self.nHidden1R], [[] for i in self.nHidden2R]]

      for layer in range(N_LAYERS):
         for i in range(self.layerSpec[layer]):
            for j in range(self.layerSpec[layer + 1]):
               weights[layer][i].append(self.getRandomValue(randMin, randMax))

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
   " Propogate inputs through network by running computeLayer twice. Only used when training.
   """
   def runTraining(self):
      self.Esum = 0.0
      omegai = 0.0

      for i in self.nOutputsR:
         self.thetai[i] = 0.0

         for j in self.nHidden2R:

            self.thetaj[j] = 0.0
            for k in self.nHidden1R:
                  
               self.thetak[k] = 0.0
               for m in self.nInputsR:
                  self.thetak[k] += self.am[m] * self.weights[0][m][k]

               self.ak[k] = self.f(self.thetak[k])
               self.thetaj[j] += self.ak[k] * self.weights[1][k][j]
         
            self.aj[j] = self.f(self.thetaj[j])
            self.thetai[i] += self.aj[j] * self.weights[2][j][i]
         # for j in self.nHidden2R
         
         self.ai[i] = self.f(self.thetai[i])
         omegai = self.Ti[i] - self.ai[i]
         self.psii[i] = omegai * self.fDeriv(self.thetai[i])

         self.Esum += omegai * omegai
      # for i in self.nOutputsR

      return
   # def runTraining(self)

   """
   " Propogate inputs through network by running computeLayer twice. Only used when running the network.
   """
   def run(self):
      self.Esum = 0.0
      omegai = 0.0
      thetai = 0.0
      thetaj = 0.0
      thetak = 0.0

      for i in self.nOutputsR:
         thetai = 0.0

         for j in self.nHidden2R:

            thetaj = 0.0
            for k in self.nHidden1R:
                  
               thetak = 0.0
               for m in self.nInputsR:
                  thetak += self.am[m] * self.weights[0][m][k]

               self.ak[k] = self.f(thetak)
               thetaj += self.ak[k] * self.weights[1][k][j]
         
            self.aj[j] = self.f(thetaj)
            thetai += self.aj[j] * self.weights[2][j][i]
         # for j in self.nHidden2R
         
         self.ai[i] = self.f(thetai)
         omegai = self.Ti[i] - self.ai[i]

         self.Esum += omegai * omegai
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
      print("Number of Nodes in Hidden Layer 1:", self.nHidden1)
      print("Number of Nodes in Hidden Layer 2:", self.nHidden2)
      print("Number of Outputs:", self.nOutputs)
      print("Random Value Range:", self.randRange)

      return

   """
   " Returns the error of the network.
   """
   def getError(self):
      return GET_ERROR_MULT * self.Esum

   """
   " Trains the network given a set of inputs and outputs for training data as well as the max iterations,
   " error threshold, and learning rate. Uses backpropagation to optimize the training.
   " 
   " inputs specifies the input training data
   " outputs specifies the output training data
   " maxIterations specifies the maximum iterations for training
   " errorThreshold specifies the threshold which the training set error must reach before exiting training
   " lr specifies lambda, or the learning rate
   "
   " Precondition: input and output arrays are the same length.
   "  
   " Returns the weights after training is completed.
   """
   def train(self, inputs, outputs, maxIterations, errorThreshold, lr):
      iterations = 0
      
      totalError = 0.0                             # sum of error used for end condition
      trainingLen = len(inputs)                    # length of training set
      
      finished = False                             # used to determine whether training is finished or not
      errorThresholdReached = False
      maxIterationsReached = False

      trainingTime = time.time()

      # set input and output training arrays
      self.inputSet = inputs
      self.outputSet = outputs

      # training loop
      while not finished:
         self.am, self.Ti = self.getNextTrainingMember()

         self.runTraining()

         for j in self.nHidden2R:
            self.omegaj[j] = 0.0
            for i in self.nOutputsR:
               self.omegaj[j] += self.psii[i] * self.weights[2][j][i]
               self.weights[2][j][i] += lr * self.aj[j] * self.psii[i]

            self.psij[j] = self.omegaj[j] * self.fDeriv(self.thetaj[j])
            
         for k in self.nHidden1R:
            self.omegak[k] = 0.0
            for j in self.nHidden2R:
               self.omegak[k] += self.psij[j] * self.weights[1][k][j]
               self.weights[1][k][j] += lr * self.ak[k] * self.psij[j]

            self.psik[k] = self.omegak[k] * self.fDeriv(self.thetak[k])

            for m in self.nInputsR:
               self.weights[0][m][k] += lr * self.am[m] * self.psik[k]
         # for k in self.nHidden1R

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

      trainingTime = time.time() - trainingTime
      
      self.runOverTrainingData()

      if errorThresholdReached:
         print("Network has reached the error threshold.")
      if maxIterationsReached:
         print("Network has reached maximum iterations.")

      print("\nMax Iterations:", maxIterations)
      print("Error Threshold:", errorThreshold)
      print("Learning Rate:", lr)
      print("Use Preloaded Weights: " + str(self.preloadedWeights))

      print("\nTraining time: " + str(int(trainingTime * MS_IN_S)) + " ms")

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
      if self.trainingPos == len(self.inputSet):
         self.trainingPos = 0

      return (self.inputSet[self.trainingPos], self.outputSet[self.trainingPos])

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
      
      for i in range(len(self.inputSet)):
         self.am, self.Ti = self.getNextTrainingMember()

         self.run()
         
         TiRounded = [round(i, N_DIGITS_DEC) for i in self.Ti]
         FiRounded = [round(i, N_DIGITS_DEC) for i in self.ai]
         errorRounded = round(self.getError(), N_DIGITS_DEC)

         totalError += errorRounded

         print(self.am, "\t", FiRounded, "\t", TiRounded, "\t\t", errorRounded)
      # for i in range(len(self.inputSet))

      print("\nTotal Error: ", round(totalError, N_DIGITS_DEC))

      return
   # def runOverTrainingData(self)
# class Network
