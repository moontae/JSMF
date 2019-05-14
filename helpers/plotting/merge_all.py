# Merge results.

import os
import sys

prefix = 'result_K'
suffix = '.metrics'
for subdir, dirs, files in os.walk('.'):	
	if subdir.find('admmDR') < 0 and subdir.find('activeSet') < 0 and subdir.find('expGrad') < 0:
		continue	
	
	print(subdir)
	files = [file for file in files if file.find(prefix) >= 0 and file.find(suffix) >= 0]
	sortedFiles = sorted(files, key=lambda file: int(file[file.find('-') + 1:file.rfind('.')]))
	print(sortedFiles)

	mergedFile = open(subdir + '/result_all' + suffix, 'w')    
	for (i, sortedFile) in enumerate(sortedFiles):
		eachFile = open(subdir + '/' + sortedFile, 'r')		

		if i == 0:
			lines = eachFile.readlines()
		else:
			lines = eachFile.readlines()[1:]

		for line in lines:
			if len(line.strip()) > 0:
				mergedFile.write(line)

		eachFile.close()
	mergedFile.close()



