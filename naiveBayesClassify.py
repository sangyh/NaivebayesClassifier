import numpy as np

def naiveBayesClassify(xTest, M, V, p):
	nb=[]
	no_classes=np.shape(M)[1]

	for row in xTest[:]:
		prod=[1,1,1,1,1]
		for feat in range(len(row)):
		    for c in range(no_classes):
		        prod[c]*=(1/np.sqrt(V[feat][c]))*np.exp(-0.5*((row[feat]-M[feat][c])**2)/V[feat][c])

		prod=np.multiply(prod,p)
		nb.append(np.argmax(prod)+1)
	return nb
