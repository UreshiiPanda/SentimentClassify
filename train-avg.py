#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector


def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    v['im_bias'] = 1   # add a bias feature, set it to 1, we now have d+1 dim
    for word in words:
        v[word] += 1
    return v
   

def test(devfile, model, dev_errs):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
        # get max/min errors on dev
        dev_errs.append((label, label * (model.dot(make_vector(words))), words))
    return err/i  # i is |D| now
           

def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    model['im_bias'] = 0  # add a bias weight, set it to 0, we now have d+1 dim
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
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

 

def train_avg(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    model['im_bias'] = 0  # add a bias weight, set it to 0, we now have d+1 dim
    w_aux = svector()
    count = 0
    print() 
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

        
        # get the 5 worst dev errors for both pos/neg labels for each epoch

        # print()
        # dev_errs.sort(key=lambda x: x[1])
        # dev_errs_pos = [(x[1], x[2]) for x in dev_errs[:30] if x[0] == 1]
        # dev_errs_neg = [(x[1], x[2]) for x in dev_errs[:30] if x[0] == -1]
        # dev_errs_pos.sort(key=lambda x: x[0])
        # dev_errs_neg.sort(key=lambda x: x[0])
        # print(f"5 Worst Mistakes on Pos Dev Labels:")
        # for err, ex in dev_errs_pos[:5]:
        #     print(f"Mistake of Error {err} on Pos Dev Label  \n  {' '.join(ex)}")
        #     print()
        # print()
        # print()
        # print(f"5 Worst Mistakes on Neg Dev Labels:") 
        # for err, ex in dev_errs_neg[:5]:
        #     print(f"Mistake of Error {err} on Neg Dev Label  \n  {' '.join(ex)}")
        #     print()
        # print()



        # print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("Avg Perceptron Dev Error: %.1f%%, |w|: %d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))



    # get 20 most positive/negative features/words

    #print()
    #pos_features = sorted(model.items(), key=lambda item: item[1], reverse=True)[:20]
    #neg_features = sorted(model.items(), key=lambda item: item[1])[:20]  
    #for k, v in pos_features:
    #    print(f'Pos Word:  {k},   Pos Weight: {v}')
    #print()
    #for k, v in neg_features:
    #    print(f'Neg Word:  {k},   Neg Weight: {v}')
    #print()


    return model

 
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
    model = train_avg(sys.argv[1], sys.argv[2], 10)
    predict(sys.argv[3], model)
