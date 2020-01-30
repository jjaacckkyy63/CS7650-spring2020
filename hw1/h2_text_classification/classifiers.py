import numpy as np
import math
###################
from tqdm import tqdm
import json
import os
###################
# You need to build your own model here instead of using well-built python packages such as sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZeor(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Store dict of dict of words for each class (In the project, hate / not hate)
        self.count = {}
        # hate / not hate
        self.classes = None

    def fit(self, X, Y):

        self.classes = set(Y.tolist())

        for _class in self.classes:
            self.count[_class] = {}
            self.count['total_sen'] = len(X) 
            for i in range(len(X[0])):
                self.count[_class][i] = 0
            self.count[_class]['total_word'] = 0
            self.count[_class]['total_sen'] = 0
        
        for i in range(len(X)):
            for j in range(len(X[0])):
                self.count[Y[i]][j] += X[i][j]
                self.count[Y[i]]['total_word'] += X[i][j]
            self.count[Y[i]]['total_sen'] += 1


    def _log_prob(self, _class, sen):
        log_prob_cls = np.log(self.count[_class]['total_sen']) - np.log(self.count['total_sen'])
        total_words = len(sen)
        for i in range(len(sen)):
            # add test into corpus
            #current_word_prob = sen[i] * (np.log(self.count[_class][i]+1)-np.log(self.count[_class]['total_word']+total_words))
            # not add test into corpus
            current_word_prob = sen[i] * (np.log(self.count[_class][i]+math.exp(-10))-np.log(self.count[_class]['total_word']))
            log_prob_cls += current_word_prob
        
        return log_prob_cls

    def predict(self, X):
        res = []
        for i in range(len(X)):
            pred = []
            for cls in self.classes:
                pred.append(self._log_prob(cls, X[i]))

            res.append(pred.index(max(pred)))

        return res

        


# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")
        

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")
        
    
    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")
