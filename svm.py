import pickle
import sys
from sklearn import svm
import numpy as np

X, Y = pickle.load( open( "svm.data", "rb" ))
X = np.array([np.array(xi) for xi in X])
Y = np.array([np.array(xi) for xi in Y])
sys.stdout.write("got data\n")
sys.stdout.flush()

def runSVM():
	sys.stdout.write("init\n")
	sys.stdout.flush()
	clf = svm.SVC(C=.8, cache_size=7000)

	rand_index = np.random.choice(len(X), size=100000)
	clf.fit(X[rand_index], Y[rand_index])
	pickle.dump(clf, open( "svm.model", "wb" ))
	sys.stdout.write("done\n")
	sys.stdout.flush()

#runSVM();

def getAccuracy(X, Y):
	clf = pickle.load( open( "svm.model", "rb" ))
	rand_index = np.random.choice(len(X), size=100000)
	X = X[rand_index]
	Y = Y[rand_index]
	ypred = clf.predict(X)
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	for i in range(0, 100000):
		if(ypred[i] == 0):
			if(Y[i] == 0):
				count1 += 1
			else:
				count2 += 1
		else:
			if(Y[i] == 0):
				count3 += 1
			else:
				count4 += 1
	sys.stdout.write("pred = 0, tar = 0: " + str(count1) + "\n")
	sys.stdout.write("pred = 0, tar = 1: " + str(count2) + "\n")
	sys.stdout.write("pred = 1, tar = 0: " + str(count3) + "\n")
	sys.stdout.write("pred = 1, tar = 1: " + str(count4) + "\n")
	sys.stdout.write("done\n")
	sys.stdout.flush()

getAccuracy(X, Y)

