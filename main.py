"""
" Arjun Dixit
" September 10, 2021
"
" The main file used to run the network with parameters specified by the config file of CONFIG_FILE_NAME.
" Contains the following methods:
"
" loadConfig(fName)
" genBooleanTrainingData(booleanFnStrs)
"
"""
import json
from network import Network
import operator

CONFIG_FILE_NAME = "config.json"

"""
" Loads the config from a JSON file (.json), and
" returns a JSON object representing the contents of the file.
"
" fName specifies the name of the file from which the config should
"       be loaded. The provided filename must include the file extension.
"""
def loadConfig(fName):
   return json.load(open(fName,))

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
   config = loadConfig(CONFIG_FILE_NAME)
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

   print("\nConfig File:", CONFIG_FILE_NAME, "\n")
# if __name__ == "__main__"
