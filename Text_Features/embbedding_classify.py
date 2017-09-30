from csv import DictReader, DictWriter
import csv
import random
import numpy as np
from numpy import array
import scipy
from scipy.sparse import hstack
import string

import gensim

from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import VarianceThreshold

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin


kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kTROPE = 'trope'


'''
Implemented different features extraction methods for the text classification problem.


Different transformers used to extract features
    1. CountVectorizer(): bag of words
    2. TfidfVectorizer(): tfidf
    3. NLTKPreprocessor(): preprocess document
    4. IMBD(): features from IMBD movie dataset
    5. Keyword(): movies key words in a sentence
    6. Trope(): trope of a sentence
'''

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class IMBD:
    '''
    Get the features from IMBD dataset
    '''
    def __init__(self):
        self.vector = CountVectorizer()

    def fit_transformer(self):
        self.data = list(DictReader(open("./movie_metadata.csv", 'r')))
        feavector = self.vector.fit_transform(x['movie_title']+x['genres'] for x in self.data)


    def train_feature(self,examples):
        imbd_feature = self.vector.transform(examples)/50
        imbd_count = imbd_feature.sum(axis=1)/100

        print ("imbd",type(imbd_count))
        return imbd_count

    def test_feature(self,examples):
        return self.train_feature(examples)

class Keyword:
    def __init__(self):
        self.keyword = CountVectorizer()

    def fit_transformer(self):
        self.data = list(DictReader(open("./movie_metadata.csv", 'r')))
        keyword = self.keyword.fit_transform(x['plot_keywords']+x['actor_1_name']+x['actor_2_name']
                                             + x['actor_3_name']for x in self.data)


    def train_feature(self,examples):
        imbd_keyword = self.keyword.transform(examples)

        return imbd_keyword.sum(axis=1)/100

    def test_feature(self,examples):
        return self.train_feature(examples)

class Trope:
    def __init__(self):
        self.vector = CountVectorizer()

    def train_feature(self,examples):
        trope_train = self.vector.fit_transform(examples)

        return trope_train

    def test_feature(self,examples):
        trope_test = self.vector.transform(examples)

        return trope_test


class Featurizer:
    def __init__(self):
        self.df = CountVectorizer( ngram_range=(1,5), stop_words='english')
        self.tfidf = TfidfTransformer()

    def train_feature(self, examples):
        tfidf_train = self.df.fit_transform(examples)
        tfidf_train = self.tfidf.fit_transform(tfidf_train)

        return tfidf_train

    def test_feature(self, examples):
        tfidf_test = self.df.transform(examples)
        tfidf_test = self.tfidf.transform(tfidf_test)

        return tfidf_test

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.df.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
            bad = np.argsort(np.absolute(classifier.coef_[0]))[:11000]
            with open("stopwords.csv", 'w') as stopwords:
                d = feature_names[bad].tolist()
                wr =csv.writer(stopwords)
                for i in d:
                    wr.writerow([i,])
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))



class Cross_validation():
    '''
    Use the test data to do k-fold cross validation
    '''

    def __init__(self, k):
        self.k = k # k-fold cross_validation
        train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
        self.data = list(DictReader(open("./movie_metadata.csv", 'r')))
        self.sum = train + list(DictReader(open("../data/spoilers/test.csv", 'r')))
        random.shuffle(train)
        self.train = train
        self.accuracy = 0
        self.set_number = int(len(self.train)/k)

    def cross_train(self):
        self.accuracy = 0
        labels = []
        for line in self.train:
            if not line[kTARGET_FIELD] in labels:
                labels.append(line[kTARGET_FIELD])

        tokens1 = NLTKPreprocessor(stopwords=['a']).fit_transform((x[kTEXT_FIELD]) for x in self.sum)
        tokens2 = NLTKPreprocessor().fit_transform(x['movie_title']+x['genres']
        +x['plot_keywords']+x['actor_1_name']+x['actor_2_name']+x['actor_1_name']for x in self.data)

        model = gensim.models.Word2Vec(tokens1+tokens2, size=100, window=5, min_count=5, workers=2)
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

        for i in range(self.k):
            train = self.train[0:i*self.set_number] + self.train[(i+1)*self.set_number:]
            test = self.train[i*self.set_number:(i+1)*self.set_number]
            tokens_train = NLTKPreprocessor().fit_transform((x[kTEXT_FIELD]) for x in train)
            tokens_test = NLTKPreprocessor().fit_transform((x[kTEXT_FIELD]) for x in test)

            feat = Featurizer()
            movie = IMBD()
            init = movie.fit_transformer()
            trope = Trope()
            keyword = Keyword()
            init2 = keyword.fit_transformer()

            x_train = feat.train_feature((x[kTEXT_FIELD]) for x in train)
            imbd_train = movie.train_feature((x[kTEXT_FIELD]+x[kTROPE]) for x in train)
            trope_train = trope.train_feature(x[kTROPE] for x in train)
            keyword_train = keyword.train_feature((x[kTEXT_FIELD]+x[kTROPE]) for x in train)
            print ("imbd feature shape",imbd_train.shape,"orignal data",x_train.shape)

            embedding_train = MeanEmbeddingVectorizer(w2v).fit_transform(x for x in tokens_train)
            print ('embedding size',embedding_train.shape)

            x_train = hstack([x_train,
                            imbd_train,
                              embedding_train,
                              trope_train,keyword_train])

            print ("train data shape",x_train.shape)

            x_test = feat.test_feature((x[kTEXT_FIELD]) for x in test)
            imbd_test = movie.test_feature((x[kTEXT_FIELD]+x[kTROPE]) for x in test)
            trope_test = trope.test_feature(x[kTROPE] for x in test)
            keyword_test = keyword.test_feature((x[kTEXT_FIELD]+x[kTROPE]) for x in test)

            embedding_test = MeanEmbeddingVectorizer(w2v).fit_transform(x for x in tokens_test)

            x_test = hstack([x_test,
                            imbd_test,
                             embedding_test,
                             trope_test,keyword_test])

            print ("x_test type", type(x_test))


            # train and test labels are numbers and should be consistent
            y_train = array(list(labels.index(x[kTARGET_FIELD])
                                 for x in train))
            y_test = array(list(labels.index(x[kTARGET_FIELD])
                                 for x in test))

            lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
            lr.fit(x_train, y_train)
            # feat.show_top10(lr,labels)
            self.accuracy = self.accuracy + lr.score(x_test,y_test)

        self.accuracy = self.accuracy/self.k

        return self.accuracy




if __name__ == "__main__":
    # Generate the test data
    
    # # Cast to list to keep it all in memory
    # train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    # test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    #
    # feat = Featurizer()
    # movie = IMBD()
    # init = movie.fit_transformer()
    # trope = Trope()
    #
    # labels = []
    # for line in train:
    #     if not line[kTARGET_FIELD] in labels:
    #         labels.append(line[kTARGET_FIELD])
    #
    # print("Label set: %s" % str(labels))
    #
    #
    # x_train = feat.train_feature((x[kTEXT_FIELD]) for x in train)
    # imbd_train = movie.train_feature((x[kTEXT_FIELD] + x[kTROPE]) for x in train)
    # trope_train = trope.train_feature(x[kTROPE] for x in train)
    #
    # x_train = hstack([x_train, imbd_train, trope_train])
    #
    #
    # x_test = feat.test_feature((x[kTEXT_FIELD]) for x in test)
    # imbd_test = movie.test_feature((x[kTEXT_FIELD] + x[kTROPE]) for x in test)
    # trope_test = trope.test_feature(x[kTROPE] for x in test)
    # x_test = hstack([x_test, imbd_test, trope_test])
    #
    # y_train = array(list(labels.index(x[kTARGET_FIELD])
    #                      for x in train))
    #
    # print(len(train), len(y_train))
    # print(set(y_train))
    #
    # # Train classifier
    # lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    # lr.fit(x_train, y_train)
    #
    # # feat.show_top10(lr, labels)
    #
    # predictions = lr.predict(x_test)
    # o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    # o.writeheader()
    # for ii, pp in zip([x['id'] for x in test], predictions):
    #     d = {'id': ii, 'cat': labels[pp]}
    #     o.writerow(d)


    # Cross validation
    data = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    true = 0
    false = 0
    for x in data:
        if x[kTARGET_FIELD] == "True":
            true += 1
        else:
            false += 1
    # print (type(x[kTARGET_FIELD]))
    print ("True sample",true,"percent",true/(true+false))
    print ("False sample",false,"percent",false/(true+false))
    crosstest = Cross_validation(6)
    accuracy = crosstest.cross_train()
    print (accuracy)



