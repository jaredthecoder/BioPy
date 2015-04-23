#driver.py
import os

def run(exp=0, l=20, N=30, G=10, pm=.033, pc=.6, NG=20):
		string = 'python .\geneticRun.py'\
		+ ' -exp ' + str(exp) \
		+ ' --num_bits '+ str(l) \
		+ ' --population_size ' + str(N)\
		+ ' --num_gens ' + str(G)\
		+ ' --pr_mutation ' + str(pm)\
		+ ' --pr_crossover ' + str(pc)\
		+ ' --num_learning_guesses ' + str(NG)
		print "DRIVER: %s" % string
		os.system(string)
def learn(exp=0, l=20, N=30, G=10, pm=.033, pc=.6, learn=True, NG=20):
		string = 'python .\geneticRun.py'\
		+ ' -exp ' + str(exp) \
		+ ' --num_bits '+ str(l) \
		+ ' --population_size ' + str(N)\
		+ ' --num_gens ' + str(G)\
		+ ' --pr_mutation ' + str(pm)\
		+ ' --pr_crossover ' + str(pc)\
		+ ' --learn_offspring ' + str(learn)\
		+ ' --num_learning_guesses ' + str(NG)
		print "DRIVER: %s" % string;
		os.system(string)
def CE(exp=0, l=20, N=30, G=10, pm=.033, pc=.6, learn=False, CE=True, NG=20, nruns=1, seed=123456):
		string = 'python .\geneticRun.py'\
		+ ' -exp ' + str(exp) \
		+ ' --num_bits '+ str(l) \
		+ ' --population_size ' + str(N)\
		+ ' --num_gens ' + str(G)\
		+ ' --pr_mutation ' + str(pm)\
		+ ' --pr_crossover ' + str(pc)\
		+ ' --learn_offspring ' + str(learn)\
		+ ' --change_environment ' + str(CE)\
		+ ' --num_learning_guesses ' + str(NG)
		print "DRIVER: %s" % string;
		os.system(string)

exp = 0;
pm = .033;
N = 30
print "DRIVER: EVALUATING POPULATION SIZE AND MUTATION PROBABILITY"
run(exp = exp, N = N, pm = 1/N)

while (N < 150):
	run(exp = exp, N = N, pm = 1/N)
	N+=10
	exp+=1
print "DRIVER: EVALUATING CROSSOVER PROBABLITY"
pc = .3
while (pc < 1):
	run(exp = exp, pc = pc)
	pc +=.1
	exp+=1
print "DRIVER: EVALUATING NUMBER OF GENERATIONS"
G=30
while (G <= 150):
	run(exp = exp, G = G);
	G+=10
	exp+=1