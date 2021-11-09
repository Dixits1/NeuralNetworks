"""
" Arjun Dixit
" September 10, 2021
"
" A helper file containing helper methods for file I/O.
" Contains the following methods:
" 
" loadContentsFromFile(fName, dir, verif)
" saveContentsToFile(contents, fName, dir, verif)
" loadConfig(fName)
"""
import pickle
import json

CONFIGS_DIR = "configs/"   # configs irectory
WEIGHTS_DIR = "weights/"   # weigths directory
TRAINING_DIR = "training/" # training data directory

"""
" Loads in an object from a pickle file (.p), and returns that object 
" if the verification matches.
"
" fName specifies the name of the file from which the contents should
"       be loaded. The provided filename must include the file extension.
" dir specifies the directory to open the file from.
" verif specifies the object used to verify the contents of the file, stored       
"       as the first element of the tuple in the file.
"
" Returns the contents if they are successfully loaded from the file; otherwise,
" prints the error returns None.
"""
def loadContentsFromFile(fName, dir, verif):
   fileContents = None
   mainContents = None

   try:
      fileContents = pickle.load(open(dir + fName, "rb"))

      if fileContents[0] == verif:
         mainContents = fileContents[1]
      else:
         raise Exception("Verification failed when trying to get contents from " + fName + ".")

   except Exception as e:
      print("Error: could not get contents from " + fName + ".")
      print(e)

   return mainContents
# def loadContentsFromFile(fName, dir, verif)

"""
" Saves the provided contents to a pickle file named fName.
"
" contents specifies the object which should be saved to the file fName.
" fName specifies the name of the file to which the contents should be saved. 
"       The provided filename must include the file extension.
" dir specifies the directory at which the file should be saved.
" verif specifies the object stored as the first element of the tuple,
"       used to verify that the contents are for the right configuration
"       when reopening the file.
" 
"
" Returns True if the weights array is sucessfully saved to the file; otherwise,
" prints the error and returns False.
"""
def saveContentsToFile(contents, fName, dir, verif):
   success = True


   try:
      pickle.dump((verif, contents), open(dir + fName, "wb"))

   except Exception as e:
      print("Error: could not dump contents to " + fName + ".")
      print(e)
      success = False

   return success
# def saveContentsToFile(contents, fName, dir, verif)


"""
" Loads the config from a JSON file (.json), and
" returns a JSON object representing the contents of the file.
"
" fName specifies the name of the file from which the config should
"       be loaded. The provided filename must include the file extension.
"""
def loadConfig(fName):
   return json.load(open(CONFIGS_DIR + fName,))
