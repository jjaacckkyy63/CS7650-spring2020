import math, random
from collections import defaultdict
################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    storage = []
    padded_text = start_pad(n) + text
    for i in range(n,len(padded_text)):
        storage.append((padded_text[i-n:i] , padded_text[i]))
    return storage

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
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
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''

        n_grams = ngrams(self.n, text)
        
        for i in range(len(n_grams)):
            ctex = n_grams[i][0]
            char = n_grams[i][1]
            self.vocab.add(char)
            self.ngram_record[(ctex, char)] += 1
            self.context_record[ctex] += 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        
        if context not in self.context_record:
            return 1/len(self.vocab)
        numerator = self.ngram_record.get((context, char), 0) + self.k
        denominator = self.context_record.get(context, 0) + self.k * len(self.get_vocab())

        return numerator / denominator

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        # random.seed(1)
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
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        generated_context = []
        context = start_pad(self.n)
        for i in range(length):
            next_char = self.random_char(context)
            generated_context.append(next_char)
            if self.n > 0:
                context = context[1:] + next_char
        return ''.join(generated_context)

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        generated_text = start_pad(self.n) + text
        accu_prob = 0
        for i in range(self.n,len(generated_text)):
            context = generated_text[i-self.n:i]
            prob = self.prob(context, generated_text[i])
            if prob == 0:
                return float('inf')
            accu_prob += math.log(prob, 2)
        l = accu_prob / len(text)
        
        return math.pow(2, -l)

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super(NgramModelWithInterpolation, self).__init__(n, k)
        self.list_lambda = [1/(n+1)] * (n+1)

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        # update vocab
        for i in range(self.n+1):
            n_grams = ngrams(i, text)
            for i in range(len(n_grams)):
                ctex = n_grams[i][0]
                char = n_grams[i][1]
                self.vocab.add(char)
                self.ngram_record[(ctex, char)] += 1
                self.context_record[ctex] += 1

    def prob(self, context, char):
        
        p_in = 0        
        for i in range(0, self.n+1):
            context_in = context[i:]
            numerator = self.ngram_record.get((context_in, char), 0) + self.k
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