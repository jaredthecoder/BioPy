# run multiple.py
import os


def recursive_run(hidden_layers, learning_rate, exp, tests):
	if len(hidden_layers) == 11:
		return 0
	hlstr = ""
	for i in hidden_layers:
		hlstr += str(i) + " "
	for test in tests:
		if test == 'f':
			os.system("python run_tests.py -exp " +str(exp) + " -ttype " + test + " -hidden_layers " + hlstr + " --learning_rate " + str(learning_rate) + ' --ftrain training1.txt --ftest testing1.txt --fvalid validation1.txt')
			exp += 1
			os.system("python run_tests.py -exp " +str(exp) + " -ttype " + test + " -hidden_layers " + hlstr + " --learning_rate " + str(learning_rate) + ' --ftrain training2.txt --ftest testing2.txt --fvalid validation2.txt')
			exp += 1
		else:
			os.system("python run_tests.py -exp " +str(exp) + " -ttype " + test + " -hidden_layers " + hlstr + " --learning_rate " + str(learning_rate))
	learning_rate += .1

	if learning_rate == 1:
		learning_rate = .1
		if hidden_layers[len(hidden_layers)-1] == 100:
			for i in hidden_layers:
				i = 1
			hidden_layers.append(0)
		hidden_layers[len(hidden_layers)-1] += 1
   	if recursive_run(hidden_layers, learning_rate, exp, tests) == 0:
   		return 0
   	else:
   		return 1

recursive_run([1], .1,0, ['x', 'i', 'd', 'f'])
