import sys
import numpy as np
import pandas


#If the number of arguments is less than 3, print an error and exit
if len(sys.argv) > 3:
    print("Too little arguments. Exiting...\n", file=sys.stderr)

#The arguments passed in should be:
trainCSV = sys.argv[1]
testCSV = sys.argv[2]

#Read the CSV files as Panda DataFrames
#Note that the files DO have headers, which is the default behavior
#Here's hoping that it works with just the arument passed in

train = pandas.DataFrame(data=pandas.read_csv(trainCSV))
test = pandas.DataFrame(data=pandas.read_csv(testCSV))

#Get the ground truth labels
#Grab the "ground" column from the training and testing data
train_ground = train.loc[:,'ground']
test_ground = test.loc[:,'ground']

#Drop the "ground" column from the training and testing data
train = train.drop(['ground'], axis=1)
test = test.drop(['ground'], axis=1)

#Drop the "Filename" column from the training and testing data
train = train.drop(['Filename'], axis=1)
test = test.drop(['Filename'], axis=1)

#Get the values (Numpy arrays) of the remaining training and testing data
train = train.values
test = test.values

#Create and fit ANY two classifiers from the scikit-learn library to the training data
#Ian's decision: Random Forest and AdaBoostClassifier
#Random Forest
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100)
forest.fit(train, train_ground)

#AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

adaBoost = AdaBoostClassifier(n_estimators=100)
ada_model = adaBoost.fit(train, train_ground)

#Using your two classifiers, predict the results for the training data
forest_pred_tr = forest.predict(train)
ada_pred_tr = ada_model.predict(train)

#Using your two classifiers, predict the results for the testing data
forest_pred_tst = forest.predict(test)
ada_pred_tst = ada_model.predict(test)

#For each classifier, print out the following (WITH TEXT THAT EXPLAINS WHICH NUMBER BELONGS TO WHICH STATISTIC, DATASET, AND CLASSIFIER)

print("Predictor 1: Random Forest Classifier")
#Training accuracy with classifer 1 (USE A DESCRIPTIVE NAME for the classifer, like "Adaboost with 100")
print("Forest Train Accuracy:",sklearn.metrics.accuracy_score(train_ground,forest_pred_tr))
#Training F1 score (average = "macro") with classifer 1
print("Forest Train F1:",sklearn.metrics.f1_score(train_ground,forest_pred_tr, average="macro"))
#Testing accuracy with classifer 1
print("Forest Test Accuracy:", sklearn.metrics.accuracy_score(test_ground,forest_pred_tst))
#Testing F1 score with classifier 1
print("Forest Test F1", sklearn.metrics.f1_score(test_ground, forest_pred_tst, average="macro"))

print("Predictor 2: Adaboost Classifier")
#Training accuracy with classifier 2
print("Adaboost Train Accuracy:", sklearn.metrics.accuracy_score(train_ground,ada_pred_tr))
#Training F1 score with classifier 2
print("Adaboost Train F1:", sklearn.metrics.f1_score(train_ground, ada_pred_tr, average="macro"))
#Testing accuracy with classifer 2
print("Adaboost Test Accuracy:", sklearn.metrics.accuracy_score(test_ground,ada_pred_tst))
#Testing F1 score with classifier 2
print("Adaboost Test F1:", sklearn.metrics.f1_score(test_ground,ada_pred_tst,average="macro"))
