"""
" Arjun Dixit
" September 10, 2021
"
" The main file used to run the network with parameters specified by the config file of CONFIG_FILE_NAME.
" Contains the following methods:
"
" genBooleanTrainingData(booleanFnStrs)
"""

from network import Network
import operator
from fileio import *
import sys

DEFAULT_CONFIG_NAME = "config.json" 
N_ARGS_CONFIG = 2  # the number of args needed for the config to be in args

"""
" Generates training input and output data given the name of a boolean operator as a string in lowercase.
" 
" Returns a tuple in the following format:
" (array of training input arrays, array of training output arrays)
"""
def genBooleanTrainingData(booleanFnStrs):
   inputs = [
      [0.0, 0.0], 
      [1.0, 0.0], 
      [0.0, 1.0], 
      [1.0, 1.0]
      ]
   outputs = [[] for i in range(len(inputs))]
   booleanFns = []

   # convert string to function
   for i in booleanFnStrs:
      booleanFns.append(getattr(operator, "__" + i + "__"))
   

   for i in range(len(inputs)):
      for j in booleanFns:
         # run boolean function using int representations of inputs, convert output to float and append to outputs array
         outputs[i].append(float(j(int(inputs[i][0]), int(inputs[i][1]))))

   return (inputs, outputs)
# def genBooleanTrainingData(booleanFnStrs)

# main
if __name__ == "__main__":
   # load config
   configName = sys.argv[0] if len(sys.argv) == N_ARGS_CONFIG else DEFAULT_CONFIG_NAME
   config = loadConfig(configName)
   tParams = config['training']['params'] # training parameters

   # create the network
   network = Network(config['shape']['nInputs'], config['shape']['nHidden'], config['shape']['nOutputs'])

   # input/output training data
   inputs, outputs = genBooleanTrainingData(config['training']['data']['booleanOperators'])

   # choose training vs running
   if config['trainNetwork']:
      # training
      network.train(inputs, outputs, tParams['maxIterations'], tParams['errorThreshold'], tParams['learningRate'])
   else:
      # running
      network.weights = config['weights']
      network.trainingInputs = inputs
      network.trainingOutputs = outputs

      network.runOverTrainingData()


   print("\nConfig File:", configName, "\n")
# if __name__ == "__main__"
