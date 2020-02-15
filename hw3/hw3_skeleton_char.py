import math, random

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
        self.counts = {}
        self.vocab = set()

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        # update vocab
        for c in text:
            self.vocab.add(c)

        n_grams = ngrams(self.n, text)

        for n_gram in n_grams:
            context = n_gram[0]
            char = n_gram[1]
            if context in self.counts:
                if char in self.counts[context]:
                    self.counts[context][char] += 1
                else:
                    self.counts[context][char] = 1
            else:
                self.counts[context] = {}
                self.counts[context][char] = 1


    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        candidates = self.counts.get(context, None)

        if not candidates:
            return 1 / len(self.vocab)

        num_char = self.counts[context].get(char, 0)
        num_sum = sum(self.counts[context].values())

        return num_char / num_sum

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        # random.seed(1)
        r = random.random()

        if context:
            if context in self.counts:
                order_char = sorted(self.counts[context].keys())
            else:
                order_char = sorted(list(self.vocab))    
        else:
            order_char = sorted(list(self.vocab))

        order_char_prob = [self.prob(context, key) for key in order_char]
        accu_prob = [sum(order_char_prob[:i]) for i in range(len(order_char_prob)+1)]

        for i in range(1, len(order_char)+1):
            if accu_prob[i-1] <= r and r < accu_prob[i]:
                return order_char[i-1]


        

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        generated_context = ''
        for i in range(length):
            s = self.random_char(generated_context)
            generated_context += s

        return generated_context

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

    def update(self, text):
        pass

    def prob(self, context, char):
        pass

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass