import pickle

WEIGHTS_DIR = 'weights/'

"""
Loads in a set of weights from a pickle file (.p), and
returns those weights.

fName specifies the name of the file from which the weights should
      be loaded. The provided filename must include the file extension.
    
Returns the weights if they are successfully loaded from the file; otherwise,
returns None.
"""
def getWeightsFromFile(fName):
    try:
        return pickle.load(open(WEIGHTS_DIR + fName, "rb"))
    except:
        return None

"""
Saves the provided weights to a pickle file named fName.

fName specifies the name of the file from which the weights should
      be loaded. The provided filename must include the file extension.
weights specifies the weights array to be saved.

Returns true if the weights array is sucessfully saved to the file; otherwise,
prints the error and returns false.
"""
def saveWeightsToFile(fName, weights):
    try:
        pickle.dump(weights, open(WEIGHTS_DIR + fName, "wb"))
        return True
    except Exception as e:
        print(e)
        return False