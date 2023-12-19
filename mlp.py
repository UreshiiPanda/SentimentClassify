import numpy as np
import sys
from gensim.models import KeyedVectors
from sklearn.neural_network import MLPClassifier


trainfile = sys.argv[1]
testfile = sys.argv[2]


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
x_train, y_train = wv_process(trainfile)
x_test, y_test = wv_process(testfile)



# run an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(1000,), activation='identity', solver='adam', 
                    alpha=0.0001, batch_size='auto', learning_rate='constant', 
                    learning_rate_init=0.001, max_iter=1000, random_state=42)
mlp.fit(x_train, y_train)
y_test_pred = mlp.predict(x_test)


# produce output for test.txt.predicted
for pred, (label, words) in zip(y_test_pred, parse_from(testfile)):
    output = ("+" if str(pred) == "1" else "-") + "\t" + ''.join(words)
    print(output)


