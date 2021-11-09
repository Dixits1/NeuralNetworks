import re

N_IND = 3

def getLinesFromFile(fName):
	return open(fName, "r").read().split("\n")

# remove spaces
def rmSp(line):
	return line.replace(" ", "")

# get indentation
def getInd(line):
	return len(line) - len(line.lstrip())

# i = index of line in lines array
def raiseExc(i, msg):
	raise Exception("line " + str(i + 1) + ": " + msg)

def isEmpty(line):
	return len(rmSp(line)) == 0

def isComment(line):
	rmln = rmSp(line)
	return rmln.startswith("#") or rmln.startswith("\"")

def isVarDef(line, checkConst=False):
	splits = line.split("=")

	return len(splits) == 2 and not (" " in splits[0].strip()) and re.search(('^[A-Z_][A-Z0-9_]*' if checkConst else '^[A-Za-z_][A-Za-z0-9_]*'), splits[0])

def isFnDef(line):
	return rmSp(line).startswith("def")

# ILC = in-line comment
def containsILC(line):
	return (not isComment(line)) and ("#" in line)

# TODO: test each of these features (comment alignment doesn't work)
# TODO: finish all of the other ones
# TODO: print out what the linter doesnt check for


def lint(fName):
	lines = getLinesFromFile(fName)

	# ~~~~ check that top of file contains block comment ~~~~
	if not rmSp(lines[0]).startswith("\"\"\""):
		raise Exception("No top-of-file comment.")

	cnt = 1
	while rmSp(lines[cnt]).startswith("\""):
		cnt += 1

	if not rmSp(lines[cnt - 1]).startswith("\"\"\""):
		raise Exception("Top level comment formatted incorrectly.")


	for i in range(len(lines)):
		# ~~~~ check block comments for functions are correct ~~~~
		if isFnDef(lines[i]):
			if not rmSp(lines[i - 1]).startswith("\"\"\""):
				raiseExc(i, "No block comment on the line above this function definition.")

			cnt = 1
			while rmSp(lines[i - 1 - cnt]).startswith("\""):
				cnt += 1

			if not rmSp(lines[i - cnt]).startswith("\"\"\""):
				raiseExc(i - 1 - cnt, "Block comment is formatted incorrectly.")

		# ~~~~ check for end-of-block comments ~~~~
		if (not rmSp(lines[i]).startswith("\"")) and rmSp(lines[i]).endswith(":"):
			
			cnt = 1
			while lines[i + cnt].startswith((getInd(lines[i]) + N_IND) * " ") or isEmpty(lines[i + cnt]):
				cnt += 1

			if cnt >= 11:
				if not isComment(lines[i + cnt]):
					raiseExc(i, "Block does not contain an end-of-block comment.")

				if getInd(lines[i + cnt]) != getInd(lines[i]):
					raiseExc(i + cnt, "End-of-block comment is not indented correctly.")

				if rmSp(lines[i])[:-1] != rmSp(lines[i + cnt])[1:]:
					raiseExc(i + cnt, "End-of-block comment does not match the block's starting line.")

		# ~~~~ check that indentation number of spaces is correct ~~~~
		if getInd(lines[i]) % N_IND != 0 and not isEmpty(lines[i]):
			raiseExc(i, "Number of spaces for indentation is wrong.")

		# ~~~~ check that mathematical operators are spaced ~~~~


		# ~~~~ check that consecutive in-line comments (skip empty lines) are aligned
		if containsILC(lines[i]):
			# get position of # in line
			cPos = lines[i].find("#")

			cnt = 1
			while containsILC(lines[i + cnt]):
				if (not isEmpty(lines[i + cnt])) and lines[i + cnt].find("#") != cPos:
					raiseExc(i + cnt, "In-line comment does not align with the other in-line comments.")
				cnt += 1

		# ~~~~ check that lines are less than 132 characters ~~~~
		if len(lines[i]) > 132:
			raiseExc(i, "Line is longer than 132 characters.")

		# ~~~~ check for magic numbers ~~~~
		if not isComment(lines[i]):
			for ch in lines[i].replace("1", "").replace("0", ""):
				if ch.isdigit() and not isVarDef(lines[i], checkConst=True):
					raiseExc(i, "Line contains a magic number.")

	print(fName + " meets coding standards!")

def printToC(fName):
	lines = getLinesFromFile(fName)
	toc = []

	for i in range(len(lines)):
		if isFnDef(lines[i]):
			toc.append(lines[i].strip()[4:-1].replace("self, ", "").replace("self", ""))

	print("\" " + "\n\" ".join(toc))


lint("network.py")
lint("main.py")
lint("fileio.py")


