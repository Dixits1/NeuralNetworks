import pickle
import json

"""
" Loads in an object from a pickle file (.p), and
" returns that object if the verification matches.
"
" fName specifies the name of the file from which the contents should
"       be loaded. The provided filename must include the file extension.
" dir specifies the directory to import the file from.
" verif specifies the object used to verify the contents of the file, stored       
"       as the first element of the tuple in the file.
"
" Returns the contents if they are successfully loaded from the file; otherwise,
" prints the error returns None.
"""
def getContentsFromFile(fName, dir, verif):
   fileContents = None
   mainContents = None

   try:
      fileContents = pickle.load(open(dir + fName, "rb"))

      if file[0] == verif:
         mainContents = file[1]
      else:
         raise Exception("Verification failed when trying to get contents from " + fName + ".")

   except Exception as e:
      print("Error: could not get contents from " + fName + ".")

   return mainContents
# def getContentsFromFile(fName, dir, verif)

"""
" Saves the provided contents to a pickle file named fName.
"
" fName specifies the name of the file from which the weights should
"       be loaded. The provided filename must include the file extension.
" weights specifies the weights array to be saved.
"
" Returns true if the weights array is sucessfully saved to the file; otherwise,
" prints the error and returns false.
"""
def saveContentsToFile(contents, fName, dir, verif):
   success = True


   try:
      pickle.dump((verif, weights), open(dir + fName, "wb"))

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
   return json.load(open(fName,))