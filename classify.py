import numpy as np
import sys
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
#import xgboost as xgboo


trainfile = sys.argv[1]
devfile = sys.argv[2]
testfile = sys.argv[3]


# define hyperparameters for XGBoost
#params = {
#    "objective": "binary:logistic",  # Binary classification objective
#    "eval_metric": "logloss",         # Logarithmic loss metric
#    "max_depth": 3,                  # Maximum depth of trees
#    "eta": 0.1,                      # Learning rate
#    "subsample": 0.8,                # Fraction of training data to use in each round
#    "colsample_bytree": 0.8,         # Fraction of features to use in each tree
#    "seed": 42                        # Random seed for reproducibility
#}



def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words)


def preprocess(file):
    x_out = []
    y_out = []

    for i, (label, words) in enumerate(read_from(file), 1):
        x_out.append(words)
        y_out.append(label)

    return (x_out, y_out)


# pre-process the data into TF-IDF form
x_train, y_train = preprocess(trainfile)
x_dev, y_dev = preprocess(devfile)


vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_dev = vectorizer.transform(x_dev)

x_test, y_test = preprocess(testfile)
x_test = vectorizer.transform(x_test)



# run an SVM classifier
svc = SVC(kernel='linear')
t_svc = time.time()
svc.fit(x_train, y_train)
y_dev_pred = svc.predict(x_dev)
svc_acc = accuracy_score(y_dev, y_dev_pred)
print(f"SVC Classifier Dev Error: {100-100*svc_acc:.1f}%, |w|: {len(vectorizer.get_feature_names_out())}, time: {(time.time() - t_svc):.1f} secs")

y_test_pred = svc.predict(x_test)
y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
print(f"SVC % Pos on Test data: {100*y_test_tot/1000:.1f}%")
print()


# run an MNB classifier
mnb = MultinomialNB()
t_mnb = time.time()
mnb.fit(x_train, y_train)
y_dev_pred = mnb.predict(x_dev)
mnb_acc = accuracy_score(y_dev, y_dev_pred)
print(f"MNB Classifier Dev Error: {100-100*mnb_acc:.1f}%, |w|: {len(vectorizer.get_feature_names_out())}, time: {(time.time() - t_mnb):.1f} secs")

y_test_pred = mnb.predict(x_test)
y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
print(f"MNB % Pos on Test data: {100*y_test_tot/1000:.1f}%")
print()


# run an XGBoost classifier
#y_train_xgb = [0 if label == -1 else 1 for label in y_train]
#xgb_train = xgboo.DMatrix(x_train, label=y_train_xgb)
#xgb_dev = xgboo.DMatrix(x_dev)
#t_xgb = time.time()
#xgb = xgboo.train(params, xgb_train, 100)
#y_dev_pred = xgb.predict(xgb_dev)
#y_dev_pred_bin = [1 if pred > 0.50 else -1 for pred in y_dev_pred]
#xgb_acc = accuracy_score(y_dev, y_dev_pred_bin)
#print(f"XGB Classifier Dev Error: {100-100*xgb_acc:.1f}%, |w|: {len(vectorizer.get_feature_names_out())}, time: {(time.time() - t_xgb):.1f} secs")
#
#xgb_test = xgboo.DMatrix(x_test)
#y_test_pred = xgb.predict(xgb_test)
#y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
#print(f"XGB % Pos on Test data: {100*y_test_tot/1000:.1f}%")
#print()


# run an MLP classifier
mlp = MLPClassifier()
t_mlp = time.time()
mlp.fit(x_train, y_train)
y_dev_pred = mlp.predict(x_dev)
mlp_acc = accuracy_score(y_dev, y_dev_pred)
print(f"MLP Classifier Dev Error: {100-100*mlp_acc:.1f}%, |w|: {len(vectorizer.get_feature_names_out())}, time: {(time.time() - t_mlp):.1f} secs")

y_test_pred = mlp.predict(x_test)
y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
print(f"MLP % Pos on Test data: {100*y_test_tot/1000:.1f}%")
print()



