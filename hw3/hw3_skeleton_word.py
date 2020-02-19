import math, random
from typing import List, Tuple
from collections import defaultdict

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
    storage = []
    padded_text = start_pad(n) + text
    for i in range(n,len(padded_text)):
        storage.append((' '.join(padded_text[i-n:i]), padded_text[i]))
    return storage

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
        self.ngram_record = defaultdict(int)
        self.context_record = defaultdict(int)
        self.vocab = set()

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        return self.vocab

    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        n_grams = ngrams(self.n, text)

        for i in range(len(n_grams)):
            ctex = n_grams[i][0]
            word = n_grams[i][1]
            self.vocab.add(word)
            self.ngram_record[(ctex, word)] += 1
            self.context_record[ctex] += 1

    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''

        if context not in self.context_record:
            return 1/len(self.vocab)
        numerator = self.ngram_record.get((context, word), 0) + self.k
        denominator = self.context_record.get(context, 0) + self.k * len(self.get_vocab())
        return numerator / denominator

    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
        
        order_vocab = sorted(list(self.vocab))
        r = random.random()
        accu_prob = 0
        idx = 0

        while idx < len(self.vocab) and accu_prob < r:
            accu_prob += self.prob(context, order_vocab[idx])
            idx += 1
        if accu_prob < r:
            print("Error")
        return order_vocab[idx-1]
    
    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        context = start_pad(self.n)
        generated_text = context
        
        for i in range(length):
            next_word = self.random_word(' '.join(context))
            context = context[1:] + [next_word]
            generated_text.append(next_word)
        
        return ' '.join(generated_text[self.n:])

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        length = len(text.strip().split())
        generated_text = start_pad(self.n) + text.strip().split()
        accu_prob = 0
        for i in range(self.n,len(generated_text)):
            context = generated_text[i-self.n:i]
            context = ' '.join(context)
            prob = self.prob(context, generated_text[i])
            if prob == 0:
                return float('inf')
            accu_prob += math.log(prob, 2)
        ll = accu_prob / len(text)
        
        return math.pow(2, -ll)



################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super(NgramModelWithInterpolation, self).__init__(n, k)
        self.list_lambda = [1/(n+1)] * (n+1)
        self.word_count = defaultdict(int)

    def get_vocab(self):
        return self.vocab

    def update(self, text:str):
        # update vocab
        for i in range(self.n+1):
            n_grams = ngrams(i, text)
            for i in range(len(n_grams)):
                ctex = n_grams[i][0]
                word = n_grams[i][1]
                self.vocab.add(word)
                self.ngram_record[(ctex, word)] += 1
                self.context_record[ctex] += 1
                self.word_count[word] += 1
                
    def prob(self, context:str, word:str):
        context = tuple(context.strip().split())
        p_in = 0
        for i in range(0, self.n+1):
            context_in = context[i:]
            if i == self.n:
                context_in = tuple()
            numerator = self.ngram_record.get((context_in, word), 0) + self.k
            denominator = self.context_record.get(context_in, 0) + self.k * len(self.get_vocab())
            if self.k == 0 and denominator == 0: # avoid divide by zero
                continue
            p_in += self.list_lambda[i] * numerator / denominator
        return p_in

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass