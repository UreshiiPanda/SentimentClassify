# SentimentClassify

### A sentiment classification ML program written in Python &amp; Bash


<a name="readme-top"></a>

<!-- sentiment classification gif -->
![classify](https://github.com/UreshiiPanda/SentimentClassify/assets/39992411/06cce4a9-4350-47bb-a127-4f12208152fd)


<!-- ABOUT THE PROJECT -->
## About The Project

This is an ML program built for the sentiment classification of movie review data. It utilizes
numpy, pandas, and sklearn libraries in order to compare movie review classification results
from various classifiers both from sklearn and from Perceptron algorithms implemented from scratch. 
Specifically, the classification results from k-NN, Vanilla Perceptron and Averaged Perceptron (avg) are compared, 
along with sklearn classifiers, including: (sklearn's) Perceptron (sklp), Multi-Layer Perceptron (mlp), 
Support Vector Machines (svc, lsvc), and Multinomial Naive Bayes (mnb). The data has been 
processed in 3 different ways for comparison purposes: OneHotEncoding for sparse data vectors (with
various scaling techniques), TF-IDF vectorization for sparse vectors with the word-count form the corpus
involved, and Word Embeddings generated from Google's Word2Vec model in order to yield dense data vectors. 
All models are plotted together in order to compare dev error results and % positive predictions on the 50/50 
split test data as more and more words are pruned from the data set in order to compare how the models perform with less and less words.

#### Results

The model with the lowest error on dev given the sparse TF-IDF vectors was mnb with a 23.0% rate. The model 
with the lowest error on dev given the word embeddings was mlp with a 22.8% rate. It can also be seen from the
graphs that the models which are able to introduce nonlinearity in their learning are able to handle changes in the
data much more resiliently, whereas the linear Perceptron algorithms tend to vary wildly as the data is manipulated. 
Models like mlp, scv, lsvc produced much more robust, consistent results across all tests relative to the simpler models. 
The mnb model is known to perform well on text classification and when combined with TF-IDF (which factors in not only how
frequently a word appears within its own data example but also its frequency within the text corpus as a whole) proved to
be a strong model on sparse vectors. The mlp and support vector machine models performed consistently well with word 
embeddings as they are most likely more capable of learning the more complex relationships between words that the word 
embeddings capture.



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This program can be run from any command line interpreter (with Python and the necessary libraries installed) or it can 
be run from a Jupyter Lab instance in order to provide graphical results. Both options are explained below. The training/
dev/test data have already been pre-normalized for consistency purposes.

###### NOTE: the mlp model can take very long to train, depending on your machine



### Installation / Execution Steps in a Shell only:

1. Clone the repo
   ```sh
      git clone https://github.com/UreshiiPanda/SentimentClassify.git
   ```
2. To run the models and see results for sparse OneHot/TF-IDF vectors: 
   ```sh
      ./tasks.sh
   ```
3. To run the models and see results for dense word embeddings: 
   ```sh
      ./tasks-embs.sh
   ```



### Installation / Execution Steps in Jupyter Lab:

1. Clone the repo
   ```sh
      git clone https://github.com/UreshiiPanda/SentimentClassify.git
   ```

2. To run the models and see results for sparse OneHot/TF-IDF vectors:
   1. Run the following bash script to run the models and generate a csv file of results: 
       ```sh
          ./tasks-graph.sh
       ```
   2. In a Jupyter Notebook, run both cells in the ```classify.ipynb``` file

3. To run the models and see results for dense word embeddings:
   1. Run the following bash script to run the models and generate a csv file of results: 
       ```sh
          ./tasks-embs-graph.sh
       ```
   2. In a Jupyter Notebook, run both cells in the ```classify-embs-graph.ipynb``` file



<p align="right">(<a href="#readme-top">back to top</a>)</p>

