import math, random
from typing import List, Tuple

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return ['~'] * n

Pair = Tuple[str, str]
Ngrams = List[Pair]
def ngrams(n, text:str) -> Ngrams:
    text=text.strip().split()
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    pass

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.count = {}

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        pass

    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        pass

    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''
        pass

    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
#         random.seed(1)
        pass

    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        pass

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        pass

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text:str):
        pass

    def prob(self, context:str, word:str):
        pass

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass