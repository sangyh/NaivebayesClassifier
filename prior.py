from collections import Counter


def prior(yTrain):
	p=[]
	global class_label,no_classes
	class_label=Counter(yTrain)
	no_classes=len(class_label)
	for item in class_label.values():
		p.append(float(item)/float(len(yTrain)))
	return p
