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

        # give the ratio of different class of words
        self.prob_ratio = {}
        for i in range(len(X[0])):
            pos_prob = self.count[0][i] / self.count[0]['total_word']
            neg_prob = self.count[1][i] / self.count[1]['total_word']
            self.prob_ratio[i] = [pos_prob/neg_prob]
        
        print(self.prob_ratio)
        print("Top-10", sorted(self.prob_ratio.items(),key=lambda kv: kv[1])[-10:])
        print("Top-10-reverse", sorted(self.prob_ratio.items(),key=lambda kv: kv[1])[:10])



    def _log_prob(self, _class, sen):
        log_prob_cls = np.log(self.count[_class]['total_sen']) - np.log(self.count['total_sen'])
        total_words = len(sen)
        for i in range(len(sen)):
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
        self.lr = 0.01
        self.batch_size = 16
        self.n_iterations = 50
        self.sigmoid = lambda z : 1 / (1 + np.exp(-z))
        self.reg_coefficient = 0.01

        self.train_loss = []
        self.test_loss = []
        

    
    def gradient_descent(self, X, y, weight, lr):
        y = y.reshape(-1, 1)
        gradients = ((np.dot(X.T, self.sigmoid(np.dot(X, weight.T)) - y)) - 2 * self.reg_coefficient * weight.T) / len(y)
        # gradients = ((np.dot(X.T, self.sigmoid(np.dot(X, weight.T)) - y))) / len(y)
        new_weight = weight - lr * gradients.T

        return new_weight
    
    def prepare_batches(self, X, y, batch_size):
        X_batch = []
        y_batch = []
        
        for i in range(len(y) // batch_size):
            X_batch.append(np.array(X[i*batch_size : i* batch_size+batch_size, :]))
            y_batch.append(np.array(y[i*batch_size : i* batch_size+batch_size]))

        if len(y) % batch_size > 0:
            X_batch.append(np.array(X[len(y) // batch_size * batch_size :, :]))
            y_batch.append(np.array(y[len(y) // batch_size * batch_size :]))
            
        return np.array(X_batch), np.array(y_batch)

    def fit(self, X, Y):
        self.weight = np.random.random(X.shape[1]).reshape(1, -1)

        # self.logloss = lambda y_hat, y : np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))  / len(y_hat)
        self.logloss = lambda y_hat, y : np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) + self.reg_coefficient*np.dot(self.weight, self.weight.T) / len(y_hat)

        X_batch, Y_batch = self.prepare_batches(X, Y, self.batch_size)
        n_batch = len(Y_batch)
        n_iter = 0

        while n_iter < self.n_iterations:
            print("Iteration:",n_iter)
            iter_correct = 0
            for i in range(n_batch):
                if i % 10 == 0:
                    print("batch:",i,"/",n_batch)
                X_mini = X_batch[i]
                Y_mini = Y_batch[i]

                self.weight = self.gradient_descent(X_mini, Y_mini, self.weight, self.lr)
                y_preds = self.sigmoid(np.squeeze(np.dot(X_mini, self.weight.T)))
                self.train_loss.append(self.logloss(y_preds, Y_mini) / len(Y_mini))
                print(self.logloss(y_preds, Y_mini).shape)


                y_preds_digits = lambda X: (X > .5) * 1
                batch_correct = sum(y_preds_digits(y_preds)==Y_mini)
                iter_correct += batch_correct
            train_iter_acc = iter_correct / len(Y)
        
            print("Iteration:", n_iter, "Acc:", train_iter_acc)
            n_iter += 1
            
    
    def predict(self, X):
        
        y_preds_digits = lambda X: (X > .5) * 1
        y_preds = self.sigmoid(np.squeeze(np.dot(X, self.weight.T)))
        
        return y_preds_digits(y_preds)

