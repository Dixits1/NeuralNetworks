import json
# from network import Network
from network import Network
import operator

CONFIG_FILE_NAME = "config.json"

"""
Loads the config from a JSON file (.json), and
returns a JSON object representing the contents of the file.

fName specifies the name of the file from which the config should
      be loaded. The provided filename must include the file extension.
"""
def loadConfig(fName):
    return json.load(open(fName,))

"""
Generates training input and output data given the name of a boolean operator.
"""
def genBooleanTrainingData(booleanFn):
    inputs = [
        [0.0, 0.0], 
        [1.0, 0.0], 
        [0.0, 1.0], 
        [1.0, 1.0]
        ]
    outputs = []
    
    for i in inputs:
        outputs.append(float(getattr(operator, "__" + booleanFn + "__")(int(i[0]), int(i[1]))))

    return (inputs, outputs)

if __name__ == "__main__":
    config = loadConfig(CONFIG_FILE_NAME)

    network = Network(config['shape']['nInputs'], config['shape']['nHidden'])

    inputs, outputs = genBooleanTrainingData(config['training']['data']['booleanOperator'])

    if config['trainNetwork']:
        # network.weights = config['weights']        
        network.train(inputs, outputs, config['training']['params']['maxIterations'], config['training']['params']['learningRate'])
    else:
        network.weights = config['weights']
        network.trainingInputs = inputs
        network.trainingOutputs = outputs

        network.runOverTrainingData()

    print("\nConfig File:", CONFIG_FILE_NAME, "\n")
