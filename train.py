#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector
from gensim.models import KeyedVectors
from sklearn.linear_model import Perceptron
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer






#####  part 1


#print( wv.most_similar(positive=['sister', 'man'], negative=['woman'], topn=10))
#print( wv.most_similar(positive=['harder', 'fast'], negative=['hard'], topn=10))
#print( wv.most_similar(positive=['musketeers', 'woman', 'criminally'], topn=10))
#print( wv.most_similar(positive=['disgusting', 'melodramas'], negative=['sympathies', 'brainpower'], topn=10))
#print( wv.most_similar(positive=['nightmares', 'superhuman', 'fistfights'], negative=['extremists', 'paranormal'], topn=10))



##### setup data and process it


# load the embeddings
wv = KeyedVectors.load('embs_train.kv')



def parse_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words)


def preprocess(file):
    x_out = []
    y_out = []

    for i, (label, words) in enumerate(parse_from(file), 1):
        x_out.append(words)
        y_out.append(label)

    return (x_out, y_out)


def wv_process(file):
    x_out = []
    y_out = []

    for i, (label, words) in enumerate(parse_from(file), 1):
        # get word embeddings for this sentence
        words = words.lower().split()
        word_embeddings = [wv[word] for word in words if word in wv]
        # now we have a list of lists, where each inner list is the wv of that word 
        if len(word_embeddings) > 0:
            # take the mean of the wv's for this sentence
            sentence_embedding = np.mean(word_embeddings, axis=0)
            x_out.append(sentence_embedding.tolist())
        else:
            # if none of the words were in wv, generate a zero-vector
            empty_sentence = np.zeros(wv.vector_size)
            x_out.append(empty_sentence.tolist())
        y_out.append(label)

    return (x_out, y_out)






## pre-process the data for OneHot for sparse vectors
#x_train, y_train = preprocess(sys.argv[1])
#x_dev, y_dev = preprocess(sys.argv[2])

# use Vectorizers to simulate OneHot
#vectorizer = TfidfVectorizer()
#x_train = vectorizer.fit_transform(x_train)
#x_dev = vectorizer.transform(x_dev)

#cv = CountVectorizer()
#x_train = cv.fit_transform(x_train)
#x_dev = cv.transform(x_dev)



## pre-process the data for word embeddings for dense vectors
x_train, y_train = wv_process(sys.argv[1])
x_dev, y_dev = wv_process(sys.argv[2])
x_test, y_test = wv_process(sys.argv[3])


## scale the word-embeddings data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_dev = scaler.transform(x_dev)
x_test = scaler.transform(x_test)


# get most similar sentences to sentence 1 and 2
def get_similar(trainfile):
    for line in open(trainfile):
        label, sentence = line.strip().split("\t")
        sentences.append(sentence)


sentences = []
first_sentence = "it 's a tour de force , written and directed so quietly that it 's implosion rather than explosion you fear"
second_sentence = "places a slightly believable love triangle in a difficult to swallow setting , and then disappointingly moves the story into the realm of an improbable thriller"

get_similar(sys.argv[1])
similarities = [wv.n_similarity(second_sentence, sentence) for sentence in sentences[:1] + sentences[2:]]
most_similar_idx = similarities.index(max(similarities))
#print(f"Most Similar Sentence:  {sentences[most_similar_idx]}")




#####  part 2 - kNN


#dev_errs = []
#
## run on DEV data
#for i in range(100):
#    if not i % 2: 
#        continue
#    else:
#        kenene = KNeighborsClassifier(n_neighbors=i)
#        kenene.fit(x_train, y_train)
#        
#        start = time.time()   
#        y_dev_pred = kenene.predict(x_dev) 
#        end = time.time()
#
#        #print("k value is:  ", i)
#        #print(f"time elapsed: {end - start}")
#
#        pos = 0
#        for val in y_dev_pred:
#            if val: pos += 1
#
#        #print(f"percentage predicted pos: {pos}/1000 = {pos/1000}")
#        test_accuracy = accuracy_score(y_dev, y_dev_pred)     
#        #print(f"DEV Error: {1 - test_accuracy}")
#        #print()
#
#        dev_accuracy = accuracy_score(y_dev, y_dev_pred)    # calc the best DEV accuracy
#        dev_errs.append( (1-dev_accuracy, i) )
#
#        print(f"k: {i}, dev error: {100* (1 - test_accuracy):.2f}")
#
#print(f"dev accuracies arr: {dev_errs}")
#print()
#print(f"min dev error: {min(dev_errs)}")
#print()
#
#
#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
#print()







#####  part 2 - Perceptron


def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())


def make_vector(words):
    v = svector()
    v['im_bias'] = 1.0
    #v['im_bias'] = np.ones(wv.vector_size)     # add a bias feature, set it to 1, we now have d+1 dim
    word_embeddings = [wv[word] for word in words if word in wv]
    sentence_embedding = np.mean(word_embeddings, axis=0)
    #v[" ".join(words)] = sentence_embedding
    for word in words:
        v[word] = np.linalg.norm(sentence_embedding)
        #v[word] = sentence_embedding
    return v


def test(devfile, model, dev_errs):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        sent = make_vector(words)
        err += label * (model.dot(sent)) <= 0
        # get max/min errors on dev
        dev_errs.append((label, label * (model.dot(make_vector(words))), words))
    return err/i  # i is |D| now



def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    model['im_bias'] = 0.0    
    #model['im_bias'] = np.zeros(wv.vector_size)   
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent 
        dev_errs = []
        dev_err = test(devfile, model, dev_errs)
        best_err = min(best_err, dev_err)
        #print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("Naive Perceptron Dev Error %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))



def train_avg(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    model['im_bias'] = 0.0
    #model['im_bias'] = np.zeros(wv.vector_size)    
    w_aux = svector()
    count = 0
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:  # model made a mistake
                updates += 1
                model += label * sent
                w_aux += count * label * sent
            count += 1 
        dev_errs = []
        dev_err = test(devfile, (count * model) - w_aux, dev_errs)
        best_err = min(best_err, dev_err)



        #print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("Avg Perceptron Dev Error: %.1f%%, |w|: %d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))


    return model



#tron = Perceptron()
#t_tron = time.time()
#tron.fit(x_train, y_train)
#y_dev_pred = tron.predict(x_dev)
#tron_acc = accuracy_score(y_dev, y_dev_pred)    # calc the best DEV accuracy
#print(f"Sklearn Perceptron Dev Error: {100-100*tron_acc:.1f}%, time: {(time.time() - t_tron):.1f} secs")
#
#y_test_pred = tron.predict(x_test)
#y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
#print(f"Sklearn Perceptron % Pos on Test data: {100*y_test_tot/1000:.1f}%")
#print()



# predict on test data
def predict(testfile, model): 
    y_test_pred = []
    t = time.time()
    for i, (label, words) in enumerate(read_from(testfile), 1):
        pred = (model.dot(make_vector(words)))
        y_test_pred.append(1 if pred > 0 else -1)
    y_test_tot = sum( [1 for label in y_test_pred if label == 1] )
    print(f"Avg Perceptron % Pos on Test data: {100*y_test_tot/1000:.1f}%")
    print()


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], 10)
    model = train_avg(sys.argv[1], sys.argv[2], 10)
    #predict(sys.argv[3], model)
    print()


