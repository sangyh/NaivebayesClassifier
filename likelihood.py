import numpy as np

def likelihood (xTrain, yTrain):
	no_samples=int(np.shape(xTrain)[0])
	no_features=int(np.shape(xTrain)[1])

	class1_feature_values,class2_feature_values,class3_feature_values,class4_feature_values,class5_feature_values=[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]

	for feature_index in range(no_features):
		for sample in range(no_samples):#no_samples
		    if yTrain[sample]==1:
		        class1_feature_values[feature_index].append(xTrain[sample][feature_index])
		    elif yTrain[sample]==2:
		        class2_feature_values[feature_index].append(xTrain[sample][feature_index])
		    elif yTrain[sample]==3:
		        class3_feature_values[feature_index].append(xTrain[sample][feature_index])
		    elif yTrain[sample]==4:
		        class4_feature_values[feature_index].append(xTrain[sample][feature_index])
		    elif yTrain[sample]==5:
		        class5_feature_values[feature_index].append(xTrain[sample][feature_index])
	M=np.zeros((5,5))
	V=np.zeros((5,5))
	for i in range(no_features):
		M[i][0]=(np.mean(class1_feature_values[i]))
		V[i][0]=(np.var(class1_feature_values[i]))
		M[i][1]=(np.mean(class2_feature_values[i]))
		V[i][1]=(np.var(class2_feature_values[i]))
		M[i][2]=(np.mean(class3_feature_values[i]))
		V[i][2]=(np.var(class3_feature_values[i]))
		M[i][3]=(np.mean(class4_feature_values[i]))
		V[i][3]=(np.var(class4_feature_values[i]))
		M[i][4]=(np.mean(class5_feature_values[i]))
		V[i][4]=(np.var(class5_feature_values[i]))

	    
	return M, V
