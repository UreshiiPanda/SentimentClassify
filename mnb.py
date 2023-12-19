import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB


trainfile = sys.argv[1]
testfile = sys.argv[2]


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
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test, y_test = preprocess(testfile)
x_test = vectorizer.transform(x_test)


# run an MLP classifier
# mlp = MLPClassifier()
# mlp.fit(x_train, y_train)
# y_test_pred = mlp.predict(x_test)


# run an MNB classifier
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_test_pred = mnb.predict(x_test)



# produce output for test.txt.predicted
for pred, (label, words) in zip(y_test_pred, read_from(testfile)):
    output = ("+" if str(pred) == "1" else "-") + "\t" + ''.join(words)
    print(output)


