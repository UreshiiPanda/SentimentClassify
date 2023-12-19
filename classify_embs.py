import numpy as np
import sys
import time
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVC




trainfile = sys.argv[1]
devfile = sys.argv[2]
testfile = sys.argv[3]


# load the embeddings
wv = KeyedVectors.load('embs_train.kv')



def parse_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words)


# process the embeddings
def wv_process(file):
    x_out = []
    y_out = []

    for i, (label, words) in enumerate(parse_from(file), 1):
        # get word embeddings for this sentence
        words = words.lower().split()
        word_embeddings = [wv[word] for word in words if word in wv]
        if len(word_embeddings) > 0:
            # take the mean of the wv's for this sentence
            sentence_embedding = np.mean(word_embeddings, axis=0)
            x_out.append(sentence_embedding)
        else:
            # if none of the words were in wv, generate a zero-vector
            empty_sentence = np.zeros(wv.vector_size)
            x_out.append(empty_sentence.tolist())
        y_out.append(label)

    return (x_out, y_out)


## pre-process the data for word embeddings for dense vectors
x_train, y_train = wv_process(sys.argv[1])
x_dev, y_dev = wv_process(sys.argv[2])
x_test, y_test = wv_process(sys.argv[3])


## scale the word-embeddings data
#scaler = MinMaxScaler()
#x_train = scaler.fit_transform(x_train)
#x_dev = scaler.transform(x_dev)
#x_test = scaler.transform(x_test)



tron = Perceptron()
t_tron = time.time()
tron.fit(x_train, y_train)
y_dev_pred = tron.predict(x_dev)
tron_acc = accuracy_score(y_dev, y_dev_pred)    # calc the best DEV accuracy
print(f"Sklearn Perceptron Dev Error: {100-100*tron_acc:.1f}%, time: {(time.time() - t_tron):.1f} secs")

y_test_pred = tron.predict(x_test)
y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
print(f"Sklearn Perceptron % Pos on Test data: {100*y_test_tot/1000:.1f}%")
print()



reg = LinearSVC()
t_reg = time.time()
reg.fit(x_train, y_train)
y_dev_pred = reg.predict(x_dev)
reg_acc = accuracy_score(y_dev, y_dev_pred)
print(f"LSVC Classifier Dev Error: {100-100*reg_acc:.1f}%, time: {(time.time() - t_reg):.1f} secs")

y_test_pred = reg.predict(x_test)
y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
print(f"LSVC % Pos on Test data: {100*y_test_tot/1000:.1f}%")
print()


# run an SVM classifier
svc = SVC(kernel='linear')
t_svc = time.time()
svc.fit(x_train, y_train)
y_dev_pred = svc.predict(x_dev)
svc_acc = accuracy_score(y_dev, y_dev_pred)
print(f"SVC Classifier Dev Error: {100-100*svc_acc:.1f}%, time: {(time.time() - t_svc):.1f} secs")

y_test_pred = svc.predict(x_test)
y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
print(f"SVC % Pos on Test data: {100*y_test_tot/1000:.1f}%")
print()


# run an GNB classifier
#gnb = GaussianNB()
#t_gnb = time.time()
#gnb.fit(x_train, y_train)
#y_dev_pred = gnb.predict(x_dev)
#gnb_acc = accuracy_score(y_dev, y_dev_pred)
#print(f"GNB Classifier Dev Error: {100-100*gnb_acc:.1f}%, time: {(time.time() - t_gnb):.1f} secs")
#
#y_test_pred = gnb.predict(x_test)
#y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
#print(f"GNB % Pos on Test data: {100*y_test_tot/1000:.1f}%")
#print()


# run an MNB classifier
#mnb = MultinomialNB()
#t_mnb = time.time()
#mnb.fit(x_train, y_train)
#y_dev_pred = mnb.predict(x_dev)
#mnb_acc = accuracy_score(y_dev, y_dev_pred)
#print(f"MNB Classifier Dev Error: {100-100*mnb_acc:.1f}%, time: {(time.time() - t_mnb):.1f} secs")
#
#y_test_pred = mnb.predict(x_test)
#y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
#print(f"MNB % Pos on Test data: {100*y_test_tot/1000:.1f}%")
#print()


# run an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(1000,), activation='identity', solver='adam', 
                    alpha=0.0001, batch_size='auto', learning_rate='constant', 
                    learning_rate_init=0.001, max_iter=1000, random_state=42)
t_mlp = time.time()
mlp.fit(x_train, y_train)
y_dev_pred = mlp.predict(x_dev)
mlp_acc = accuracy_score(y_dev, y_dev_pred)
print(f"MLP Classifier Dev Error: {100-100*mlp_acc:.1f}%, time: {(time.time() - t_mlp):.1f} secs")

y_test_pred = mlp.predict(x_test)
y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
print(f"MLP % Pos on Test data: {100*y_test_tot/1000:.1f}%")
print()



